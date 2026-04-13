"""
Non-Chaotic Dynamical Systems — Training & Evaluation

Four canonical systems with single-point Markovian initialization:
  1. Damped Harmonic Oscillator  (k=1, m=1, c=0.1)
  2. Van der Pol Oscillator       (μ=0.5)
  3. Linear 2D System             (A=[[-0.1,0.5],[-0.5,-0.1]])
  4. Damped Pendulum              (g=9.81, l=1, c=0.1)

Reproduces Section 5.1, Table 3 of:
    Jahanshahi & Zhu (2026), Neurocomputing 677, 133134.

Usage:
    python examples/dynamical_systems/train_dynamical.py
"""

import os, json, warnings
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
# Dynamical systems
# ======================================================================

def linear_oscillator(t, state, k=1.0, m=1.0, c=0.1):
    x, v = state
    return [v, -(k/m)*x - (c/m)*v]

def van_der_pol(t, state, mu=0.5):
    x, y = state
    return [y, mu*(1 - x**2)*y - x]

def linear_system(t, state):
    A = np.array([[-0.1, 0.5], [-0.5, -0.1]])
    return A @ state

def damped_pendulum(t, state, g=9.81, l=1.0, c=0.1):
    theta, omega = state
    return [omega, -(g/l)*np.sin(theta) - (c/l)*omega]

SYSTEMS = [
    {"name": "Linear Oscillator", "func": linear_oscillator,
     "state_names": ["Position (m)", "Velocity (m/s)"],
     "initial_vis": [1.0, 0.0],
     "params": {"num_trajectories": 50, "t_max": 20.0, "dt": 0.1, "noise_scale": 0.05}},
    {"name": "Van der Pol", "func": van_der_pol,
     "state_names": ["x", "y"],
     "initial_vis": [0.5, 0.5],
     "params": {"num_trajectories": 75, "t_max": 20.0, "dt": 0.1, "noise_scale": 0.05}},
    {"name": "Linear System", "func": linear_system,
     "state_names": ["State 1", "State 2"],
     "initial_vis": [0.8, -0.5],
     "params": {"num_trajectories": 50, "t_max": 20.0, "dt": 0.1, "noise_scale": 0.1}},
    {"name": "Damped Pendulum", "func": damped_pendulum,
     "state_names": ["Angle (rad)", "Angular Velocity (rad/s)"],
     "initial_vis": [0.5, 0.0],
     "params": {"num_trajectories": 50, "t_max": 20.0, "dt": 0.1, "noise_scale": 0.05,
                "initial_range": (-np.pi/4, np.pi/4)}},
]

# ======================================================================
# Data generation
# ======================================================================

def generate_data(system_func, num_trajectories=50, t_max=20.0, dt=0.1,
                  noise_scale=0.05, state_dim=2, initial_range=(-1, 1)):
    t_eval = np.arange(0, t_max, dt)
    trajs = np.zeros((num_trajectories, len(t_eval), state_dim))
    for i in range(num_trajectories):
        x0 = np.random.uniform(initial_range[0], initial_range[1], state_dim)
        sol = solve_ivp(system_func, [0, t_max], x0, t_eval=t_eval,
                        method="RK45", rtol=1e-8, atol=1e-10)
        trajs[i] = sol.y.T
    trajs += np.random.randn(*trajs.shape) * noise_scale
    return trajs, t_eval, noise_scale

# ======================================================================
# Dataset
# ======================================================================

class MarkovianDataset(Dataset):
    """Single-point initialisation dataset (Section 5.1.2)."""
    def __init__(self, trajectories, time_points, future_length=20, stride=20):
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
    def __init__(self, state_dim, hidden_dim=64):
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
    def __init__(self, state_dim, hidden_dim=64, n_models=10):
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
# Training
# ======================================================================

def train_upn_system(train_loader, val_loader, obs_noise_var, state_dim=2,
                     hidden_dim=64, num_epochs=60, lr=1e-3):
    model = UPN(state_dim=state_dim, obs_dim=state_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    best_val, best_state, patience_ctr = float("inf"), None, 0
    init_cov_scale = max(obs_noise_var, 0.01)

    for epoch in range(num_epochs):
        model.train(); ep_loss, nb = 0., 0
        for x0, t0, future, ft in tqdm(train_loader, desc=f"UPN Ep {epoch+1}", leave=False):
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            bs = x0.shape[0]
            mu0 = x0
            S0 = init_cov_scale * torch.eye(state_dim, device=device).unsqueeze(0).expand(bs, -1, -1)
            t_span = torch.cat([t0[0:1], ft[0]])
            optimizer.zero_grad()
            mp, cp = model.integrate(mu0, S0, t_span)
            loss = model.compute_nll_loss(future, mp[1:], cp[1:])
            if torch.isfinite(loss):
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                ep_loss += loss.item(); nb += 1
        tl = ep_loss / max(nb, 1)

        model.eval(); vl_sum, vnb = 0., 0
        with torch.no_grad():
            for x0, t0, future, ft in val_loader:
                x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
                bs = x0.shape[0]
                S0 = init_cov_scale * torch.eye(state_dim, device=device).unsqueeze(0).expand(bs, -1, -1)
                t_span = torch.cat([t0[0:1], ft[0]])
                try:
                    mp, cp = model.integrate(x0, S0, t_span)
                    loss = model.compute_nll_loss(future, mp[1:], cp[1:])
                    if torch.isfinite(loss): vl_sum += loss.item(); vnb += 1
                except RuntimeError: continue
        vl = vl_sum / max(vnb, 1)
        scheduler.step(vl)
        print(f"  Epoch {epoch+1}: train={tl:.4f}  val={vl:.4f}")
        if vl < best_val:
            best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 10: print(f"  Early stopping at epoch {epoch+1}"); break
    if best_state: model.load_state_dict(best_state)
    return model

def train_ensemble_system(train_loader, val_loader, state_dim=2, hidden_dim=64, n_models=10, num_epochs=60, lr=1e-3):
    ensemble = EnsembleNeuralODE(state_dim, hidden_dim, n_models).to(device)
    optimizers = [optim.Adam(m.parameters(), lr=lr) for m in ensemble.models]
    for epoch in range(num_epochs):
        for m in ensemble.models: m.train()
        el, nb = 0., 0
        for x0, t0, future, ft in tqdm(train_loader, desc=f"Ens Ep {epoch+1}", leave=False):
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            t_span = torch.cat([t0[0:1], ft[0]])
            for m, opt in zip(ensemble.models, optimizers):
                opt.zero_grad()
                pred = m.predict(x0, t_span)[1:]
                loss = ((pred.permute(1,0,2) - future)**2).mean()
                if torch.isfinite(loss): loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            el += loss.item(); nb += 1
        print(f"  Ens Epoch {epoch+1}: train={el/max(nb,1):.6f}")
    return ensemble

def train_deterministic_system(train_loader, val_loader, state_dim=2, hidden_dim=64, num_epochs=60, lr=1e-3):
    model = DeterministicNeuralODE(state_dim, hidden_dim).to(device)
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

def evaluate(upn, ensemble, det, test_loader, obs_noise_var, state_dim=2):
    init_cov_scale = max(obs_noise_var, 0.01)
    results = {}

    # UPN
    upn.eval()
    upn_mse, upn_nll, upn_cov95, total_pred, n_samp = 0., 0., 0, 0, 0
    with torch.no_grad():
        for x0, t0, future, ft in test_loader:
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            bs = x0.shape[0]
            S0 = init_cov_scale * torch.eye(state_dim, device=device).unsqueeze(0).expand(bs, -1, -1)
            t_span = torch.cat([t0[0:1], ft[0]])
            try:
                mp, cp = upn.integrate(x0, S0, t_span)
                mp, cp = mp[1:], cp[1:]
                upn_mse += ((future - mp.permute(1,0,2))**2).sum().item()
                z = stats.norm.ppf(0.975)
                std = torch.sqrt(torch.diagonal(cp, dim1=2, dim2=3))
                lo = mp.permute(1,0,2) - z * std.permute(1,0,2)
                hi = mp.permute(1,0,2) + z * std.permute(1,0,2)
                upn_cov95 += ((future >= lo) & (future <= hi)).sum().item()
                total_pred += future.numel(); n_samp += bs
            except RuntimeError: continue
    results["UPN"] = {"MSE": upn_mse/total_pred, "Coverage_95": upn_cov95/total_pred}

    # Ensemble
    for m in ensemble.models: m.eval()
    ens_mse, ens_cov95 = 0., 0
    with torch.no_grad():
        for x0, t0, future, ft in test_loader:
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            t_span = torch.cat([t0[0:1], ft[0]])
            mu, cov = ensemble.predict_with_uncertainty(x0, t_span)
            mu, cov = mu[1:], cov[1:]
            ens_mse += ((future - mu.permute(1,0,2))**2).sum().item()
            z = stats.norm.ppf(0.975)
            std = torch.sqrt(torch.diagonal(cov, dim1=2, dim2=3))
            lo = mu.permute(1,0,2) - z * std.permute(1,0,2)
            hi = mu.permute(1,0,2) + z * std.permute(1,0,2)
            ens_cov95 += ((future >= lo) & (future <= hi)).sum().item()
    results["Ensemble"] = {"MSE": ens_mse/total_pred, "Coverage_95": ens_cov95/total_pred}

    # Deterministic
    det.eval(); det_mse = 0.
    with torch.no_grad():
        for x0, t0, future, ft in test_loader:
            x0, t0, future, ft = x0.to(device), t0.to(device), future.to(device), ft.to(device)
            t_span = torch.cat([t0[0:1], ft[0]])
            pred = det.predict(x0, t_span)[1:]
            det_mse += ((future - pred.permute(1,0,2))**2).sum().item()
    results["Deterministic"] = {"MSE": det_mse/total_pred}

    return results

# ======================================================================
# Main
# ======================================================================

def main():
    os.makedirs("output", exist_ok=True)
    all_results = {}

    for sys in SYSTEMS:
        print(f"\n{'='*60}\nSYSTEM: {sys['name']}\n{'='*60}")
        data, t_pts, noise_std = generate_data(sys["func"], **sys["params"])
        obs_var = noise_std ** 2
        n = len(data); nt = int(0.7*n); nv = int(0.15*n)

        train_ds = MarkovianDataset(data[:nt], t_pts)
        val_ds = MarkovianDataset(data[nt:nt+nv], t_pts)
        test_ds = MarkovianDataset(data[nt+nv:], t_pts)
        tl = DataLoader(train_ds, batch_size=16, shuffle=True)
        vl = DataLoader(val_ds, batch_size=16)
        test_l = DataLoader(test_ds, batch_size=16)

        print("\n>>> Training UPN...")
        upn = train_upn_system(tl, vl, obs_var)
        print("\n>>> Training Ensemble (10 models)...")
        ens = train_ensemble_system(tl, vl, n_models=10)
        print("\n>>> Training Deterministic...")
        det = train_deterministic_system(tl, vl)

        print("\n>>> Evaluating...")
        results = evaluate(upn, ens, det, test_l, obs_var)
        all_results[sys["name"]] = results

        print(f"\n{'Method':<15} {'MSE':<12} {'Coverage_95':<12}")
        print("-"*40)
        for method, m in results.items():
            cov = m.get("Coverage_95", float("nan"))
            print(f"{method:<15} {m['MSE']:<12.6f} {cov:<12.3f}")

    # Save
    torch.save(all_results, "output/non_chaotic_results.pt")
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    upn_mse = np.mean([all_results[s]["UPN"]["MSE"] for s in all_results])
    upn_cov = np.mean([all_results[s]["UPN"]["Coverage_95"] for s in all_results])
    ens_cov = np.mean([all_results[s]["Ensemble"]["Coverage_95"] for s in all_results])
    print(f"UPN avg MSE: {upn_mse:.6f}, avg coverage: {upn_cov:.3f}")
    print(f"Ensemble avg coverage: {ens_cov:.3f}")
    print(f"\nResults saved to output/non_chaotic_results.pt")

if __name__ == "__main__":
    main()
