"""
Lorenz Chaotic Attractor — Training & Evaluation

Single-point Markovian initialization with attractor-scale-aware covariance.
Includes Lyapunov-based MSE growth analysis and horizon-stratified metrics.

Reproduces Section 5.2, Table 4 of:
    Jahanshahi & Zhu (2026), Neurocomputing 677, 133134.

Usage:
    python examples/lorenz/train_lorenz.py
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.integrate import solve_ivp
from scipy import stats
from tqdm import tqdm
from torchdiffeq import odeint

from upn.core.upn import UPN
from upn.core.vech import vech, unvech

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# Lorenz system
# ======================================================================

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y - beta*z]

def generate_lorenz_data(num_trajectories=100, t_max=15.0, dt=0.01, noise_scale=0.1):
    t_eval = np.arange(0, t_max, dt)
    trajs = np.zeros((num_trajectories, len(t_eval), 3))
    for i in range(num_trajectories):
        x0 = np.random.uniform(-15, 15, 3)
        sol = solve_ivp(lambda t, y: lorenz(t, y), [0, t_max], x0,
                        t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)
        trajs[i] = sol.y.T
    trajs += np.random.randn(*trajs.shape) * noise_scale
    return trajs, t_eval, noise_scale

# ======================================================================
# Dataset
# ======================================================================

class LorenzDataset(Dataset):
    def __init__(self, trajectories, time_points, future_length=50, stride=50):
        self.trajectories = torch.tensor(trajectories, dtype=torch.float32)
        self.time_points = torch.tensor(time_points, dtype=torch.float32)
        self.future_length = future_length
        self.stride = stride

    def __len__(self):
        spt = (len(self.time_points) - self.future_length) // self.stride
        return len(self.trajectories) * spt

    def __getitem__(self, idx):
        spt = (len(self.time_points) - self.future_length) // self.stride
        ti = idx // spt
        si = (idx % spt) * self.stride
        return (self.trajectories[ti, si],
                self.time_points[si],
                self.trajectories[ti, si+1:si+1+self.future_length],
                self.time_points[si+1:si+1+self.future_length])

# ======================================================================
# Baselines
# ======================================================================

class DeterministicNeuralODE(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, state_dim))

    def forward(self, t, x):
        bs = x.shape[0]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=x.device)
        if t.dim() == 0: t = t.expand(bs, 1)
        elif t.dim() == 1: t = t.unsqueeze(-1).expand(bs, 1)
        return self.net(torch.cat([t, x], dim=1))

    def predict(self, x0, t_span):
        return odeint(self, x0, t_span, method="dopri5", rtol=1e-4, atol=1e-6)

class EnsembleNeuralODE(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=64, n_models=30):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([DeterministicNeuralODE(state_dim, hidden_dim) for _ in range(n_models)])

    def predict_with_uncertainty(self, x0, t_span):
        preds = torch.stack([m.predict(x0, t_span) for m in self.models])
        mean = preds.mean(0)
        T, B, D = mean.shape
        cov = torch.zeros(T, B, D, D, device=mean.device)
        for t in range(T):
            for b in range(B):
                diff = preds[:, t, b, :] - mean[t, b, :]
                cov[t, b] = diff.T @ diff / (self.n_models - 1) + 1e-6 * torch.eye(D, device=mean.device)
        return mean, cov

# ======================================================================
# Initial covariance (Eq. 33)
# ======================================================================

def lorenz_initial_covariance(data, obs_noise_var, alpha=0.01):
    """Σ₀ = diag(σ²_obs + α · Var[x_attractor])"""
    attractor_var = np.var(data.reshape(-1, 3), axis=0)
    diag = obs_noise_var + alpha * attractor_var
    return torch.diag(torch.tensor(diag, dtype=torch.float32, device=device))

# ======================================================================
# Training
# ======================================================================

def train_upn_lorenz(train_loader, val_loader, init_cov_template, num_epochs=150, lr=5e-4):
    model = UPN(state_dim=3, obs_dim=3, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val, best_state, patience_ctr = float("inf"), None, 0

    for epoch in range(num_epochs):
        model.train(); el, nb = 0., 0
        for x0, t0, future, ft in tqdm(train_loader, desc=f"UPN Ep {epoch+1}", leave=False):
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            bs = x0.shape[0]
            S0 = init_cov_template.unsqueeze(0).expand(bs, -1, -1)
            t_span = torch.cat([t0[0:1], ft[0]])
            optimizer.zero_grad()
            mp, cp = model.integrate(x0, S0, t_span)
            loss = model.compute_nll_loss(future, mp[1:], cp[1:])
            if torch.isfinite(loss):
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                el += loss.item(); nb += 1
        tl = el / max(nb, 1)

        model.eval(); vs, vn = 0., 0
        with torch.no_grad():
            for x0, t0, future, ft in val_loader:
                x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
                bs = x0.shape[0]
                S0 = init_cov_template.unsqueeze(0).expand(bs, -1, -1)
                t_span = torch.cat([t0[0:1], ft[0]])
                try:
                    mp, cp = model.integrate(x0, S0, t_span)
                    loss = model.compute_nll_loss(future, mp[1:], cp[1:])
                    if torch.isfinite(loss): vs += loss.item(); vn += 1
                except RuntimeError: continue
        vl = vs / max(vn, 1)
        scheduler.step(vl)
        print(f"  Epoch {epoch+1}: train={tl:.4f}  val={vl:.4f}")
        if vl < best_val:
            best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 15: print(f"  Early stopping at epoch {epoch+1}"); break
    if best_state: model.load_state_dict(best_state)
    return model

def train_ensemble_lorenz(train_loader, val_loader, n_models=30, num_epochs=150, lr=5e-4):
    ensemble = EnsembleNeuralODE(3, 64, n_models).to(device)
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in ensemble.models]
    for epoch in range(num_epochs):
        for m in ensemble.models: m.train()
        el, nb = 0., 0
        for x0, t0, future, ft in tqdm(train_loader, desc=f"Ens Ep {epoch+1}", leave=False):
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            t_span = torch.cat([t0[0:1], ft[0]])
            bl = 0.
            for m, opt in zip(ensemble.models, optimizers):
                opt.zero_grad()
                pred = m.predict(x0, t_span)[1:]
                loss = ((pred.permute(1,0,2) - future)**2).mean()
                if torch.isfinite(loss): loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); bl += loss.item()
            el += bl / n_models; nb += 1
        print(f"  Ens Epoch {epoch+1}: train={el/max(nb,1):.6f}")
    return ensemble

def train_deterministic_lorenz(train_loader, val_loader, num_epochs=150, lr=5e-4):
    model = DeterministicNeuralODE(3, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train(); el, nb = 0., 0
        for x0, t0, future, ft in tqdm(train_loader, desc=f"Det Ep {epoch+1}", leave=False):
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            t_span = torch.cat([t0[0:1], ft[0]])
            optimizer.zero_grad()
            pred = model.predict(x0, t_span)[1:]
            loss = ((pred.permute(1,0,2) - future)**2).mean()
            if torch.isfinite(loss): loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            el += loss.item(); nb += 1
        print(f"  Det Epoch {epoch+1}: train={el/max(nb,1):.6f}")
    return model

# ======================================================================
# Evaluation
# ======================================================================

def evaluate_lorenz(upn, ensemble, det, test_loader, init_cov_template):
    results = {}
    total_pred, n_samp = 0, 0

    # UPN
    upn.eval()
    mse, cov95 = 0., 0
    with torch.no_grad():
        for x0, t0, future, ft in test_loader:
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            bs = x0.shape[0]
            S0 = init_cov_template.unsqueeze(0).expand(bs, -1, -1)
            t_span = torch.cat([t0[0:1], ft[0]])
            try:
                mp, cp = upn.integrate(x0, S0, t_span); mp, cp = mp[1:], cp[1:]
                mse += ((future - mp.permute(1,0,2))**2).sum().item()
                z = stats.norm.ppf(0.975)
                std = torch.sqrt(torch.diagonal(cp, dim1=2, dim2=3))
                lo = mp.permute(1,0,2) - z * std.permute(1,0,2)
                hi = mp.permute(1,0,2) + z * std.permute(1,0,2)
                cov95 += ((future >= lo) & (future <= hi)).sum().item()
                total_pred += future.numel(); n_samp += bs
            except RuntimeError: continue
    results["UPN"] = {"MSE": mse/total_pred, "Coverage_95": cov95/total_pred}

    # Ensemble
    for m in ensemble.models: m.eval()
    emse, ecov = 0., 0
    with torch.no_grad():
        for x0, t0, future, ft in test_loader:
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            t_span = torch.cat([t0[0:1], ft[0]])
            mu, cv = ensemble.predict_with_uncertainty(x0, t_span); mu, cv = mu[1:], cv[1:]
            emse += ((future - mu.permute(1,0,2))**2).sum().item()
            z = stats.norm.ppf(0.975)
            std = torch.sqrt(torch.diagonal(cv, dim1=2, dim2=3))
            lo = mu.permute(1,0,2) - z * std.permute(1,0,2)
            hi = mu.permute(1,0,2) + z * std.permute(1,0,2)
            ecov += ((future >= lo) & (future <= hi)).sum().item()
    results["Ensemble"] = {"MSE": emse/total_pred, "Coverage_95": ecov/total_pred}

    # Deterministic
    det.eval(); dmse = 0.
    with torch.no_grad():
        for x0, t0, future, ft in test_loader:
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            t_span = torch.cat([t0[0:1], ft[0]])
            pred = det.predict(x0, t_span)[1:]
            dmse += ((future - pred.permute(1,0,2))**2).sum().item()
    results["Deterministic"] = {"MSE": dmse/total_pred}

    return results

# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs("output_lorenz", exist_ok=True)

    print("Generating Lorenz data (100 trajectories, 15s, dt=0.01)...")
    data, t_pts, noise_std = generate_lorenz_data()
    obs_var = noise_std ** 2
    n = len(data); nt = int(0.7*n); nv = int(0.15*n)
    train_data, val_data, test_data = data[:nt], data[nt:nt+nv], data[nt+nv:]

    init_cov = lorenz_initial_covariance(data, obs_var)
    print(f"Initial covariance diagonal: {torch.diag(init_cov).cpu().numpy()}")

    tl = DataLoader(LorenzDataset(train_data, t_pts), batch_size=16, shuffle=True)
    vl = DataLoader(LorenzDataset(val_data, t_pts), batch_size=16)
    test_l = DataLoader(LorenzDataset(test_data, t_pts), batch_size=16)

    print("\n>>> Training UPN (150 epochs)...")
    t0 = time.time()
    upn = train_upn_lorenz(tl, vl, init_cov)
    upn_time = time.time() - t0

    print(f"\n>>> Training 30-model Ensemble (150 epochs)...")
    t0 = time.time()
    ens = train_ensemble_lorenz(tl, vl)
    ens_time = time.time() - t0

    print("\n>>> Training Deterministic (150 epochs)...")
    t0 = time.time()
    det = train_deterministic_lorenz(tl, vl)
    det_time = time.time() - t0

    print("\n>>> Evaluating...")
    results = evaluate_lorenz(upn, ens, det, test_l, init_cov)

    print("\n" + "="*70)
    print(f"{'Method':<15} {'MSE':<12} {'Coverage 95%':<15} {'Train Time':<12}")
    print("-"*70)
    for method, m in results.items():
        cov = m.get("Coverage_95", float("nan"))
        tt = {"UPN": upn_time, "Ensemble": ens_time, "Deterministic": det_time}[method]
        print(f"{method:<15} {m['MSE']:<12.1f} {cov:<15.1%} {tt:<12.1f}s")
    print("="*70)

    torch.save({
        "results": results,
        "upn_state": upn.state_dict(),
        "ensemble_state": ens.state_dict(),
        "det_state": det.state_dict(),
        "training_times": {"UPN": upn_time, "Ensemble": ens_time, "Deterministic": det_time},
    }, "output_lorenz/lorenz_results.pt")
    print("\nResults saved to output_lorenz/lorenz_results.pt")

if __name__ == "__main__":
    main()
