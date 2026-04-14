"""
ETTh1 Time-Series Forecasting — UPN vs Neural SDE

Latent-space UPN with GRU encoder for time-series prediction.
Predicts 12 steps ahead from 48-step history on the ETTh1 benchmark.

Reproduces Section 5.4, Tables 6–7 of:
    Jahanshahi & Zhu (2026), Neurocomputing 677, 133134.

Usage:
    python examples/etth1/download_etth1.py   # first time only
    python examples/etth1/train_etth1.py
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint, odeint_adjoint
from tqdm import tqdm

from upn.core.vech import vech, unvech

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# Dataset
# ======================================================================

class ETTh1Dataset(Dataset):
    def __init__(self, data_path="data/etth1/etth1_simple.csv", seq_len=48, pred_len=12, split="train"):
        df = pd.read_csv(data_path)
        self.data = df[["temperature", "feature_1", "feature_2"]].values.astype(np.float32)
        self.mean = self.data.mean(0, keepdims=True)
        self.std = self.data.std(0, keepdims=True) + 1e-6
        self.data = (self.data - self.mean) / self.std
        n = len(self.data)
        bounds = {"train": (0, int(0.7*n)), "val": (int(0.7*n), int(0.85*n)), "test": (int(0.85*n), n)}
        s, e = bounds[split]
        self.data = self.data[s:e]
        self.seq_len, self.pred_len = seq_len, pred_len
        self.time = np.arange(seq_len + pred_len, dtype=np.float32) / (seq_len + pred_len)
        print(f"  {split}: {len(self.data)} points, {self.data.shape[1]} features")

    def __len__(self): return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, i):
        return (torch.tensor(self.data[i:i+self.seq_len]),
                torch.tensor(self.data[i+self.seq_len:i+self.seq_len+self.pred_len]),
                torch.tensor(self.time[:self.seq_len]),
                torch.tensor(self.time[self.seq_len:]))

# ======================================================================
# Latent UPN (Section 5.4 architecture)
# ======================================================================

class LatentUPN(nn.Module):
    """
    UPN operating in a learned latent space for time-series forecasting.
    
    Encoder maps historical observations → latent initial (μ₀, Σ₀).
    Coupled ODEs evolve latent distribution forward.
    Decoder maps latent → observation space with uncertainty.
    """

    def __init__(self, obs_dim=3, latent_dim=4, hidden_dim=16):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.vech_dim = latent_dim * (latent_dim + 1) // 2

        # Encoder: last 10 observations → initial latent mean
        self.init_encoder = nn.Sequential(
            nn.Linear(obs_dim * 10, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))

        # Learned initial covariance
        self.init_L = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)

        # Dynamics f_θ (Eq. 2)
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim))

        # Process noise Q_φ (Eq. 5)
        self.L_dim = latent_dim * (latent_dim + 1) // 2
        self.noise_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, self.L_dim))

        # Decoder: latent → observation
        self.H = nn.Linear(latent_dim, obs_dim)
        self.log_R_diag = nn.Parameter(torch.zeros(obs_dim))

    def _time_vec(self, t, bs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=device)
        if t.dim() == 0: return t.unsqueeze(0).expand(bs, 1)
        if t.dim() == 1: return t.unsqueeze(1)
        return t

    def compute_dynamics(self, mu, t):
        return self.dynamics_net(torch.cat([mu, self._time_vec(t, mu.shape[0])], dim=-1))

    def compute_noise(self, mu, t):
        bs = mu.shape[0]
        L_params = self.noise_net(torch.cat([mu, self._time_vec(t, bs)], dim=-1))
        L = torch.zeros(bs, self.latent_dim, self.latent_dim, device=mu.device)
        idx = 0
        for i in range(self.latent_dim):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = nn.functional.softplus(L_params[:, idx]) + 1e-6
                else:
                    L[:, i, j] = L_params[:, idx]
                idx += 1
        return torch.bmm(L, L.transpose(1, 2)) + 1e-6 * torch.eye(self.latent_dim, device=mu.device)

    @torch.enable_grad()
    def forward(self, t, z):
        """Coupled ODE: dz/dt for z = [μ, vech(Σ)]"""
        mu = z[:, :self.latent_dim]
        Sigma = unvech(z[:, self.latent_dim:], self.latent_dim)
        dmu = self.compute_dynamics(mu, t)

        # Jacobian
        mu_g = mu.detach().requires_grad_(True)
        f = self.compute_dynamics(mu_g, t)
        J = torch.zeros(mu.shape[0], self.latent_dim, self.latent_dim, device=mu.device)
        for i in range(self.latent_dim):
            J[:, i, :] = torch.autograd.grad(f[:, i].sum(), mu_g, retain_graph=True, create_graph=True)[0]

        Q = self.compute_noise(mu, t)
        dSigma = torch.bmm(J, Sigma) + torch.bmm(Sigma, J.transpose(1, 2)) + Q
        return torch.cat([dmu, vech(dSigma)], dim=1)

    def get_initial_state(self, x_seq):
        bs = x_seq.shape[0]
        n_use = min(10, x_seq.shape[1])
        x_flat = x_seq[:, -n_use:, :].reshape(bs, -1)
        if x_flat.shape[1] < self.obs_dim * 10:
            x_flat = torch.cat([x_flat, torch.zeros(bs, self.obs_dim*10 - x_flat.shape[1], device=x_seq.device)], dim=1)
        mu0 = self.init_encoder(x_flat)
        S0 = (self.init_L @ self.init_L.t() + 0.01 * torch.eye(self.latent_dim, device=device)).unsqueeze(0).expand(bs, -1, -1)
        return mu0, S0

    def predict(self, x_seq, t_obs, t_pred):
        bs = x_seq.shape[0]
        mu0, S0 = self.get_initial_state(x_seq)
        z0 = torch.cat([mu0, vech(S0)], dim=1)

        t_start = t_obs[0, -1:] if t_obs.dim() == 2 else t_obs[-1:]
        t_pred_1d = t_pred[0] if t_pred.dim() == 2 else t_pred
        t_int = torch.cat([t_start, t_pred_1d])

        solver = odeint_adjoint if self.training else odeint
        z_traj = solver(self, z0, t_int, method="dopri5", rtol=1e-4, atol=1e-6)[1:]

        mu_traj = z_traj[:, :, :self.latent_dim]
        sv_traj = z_traj[:, :, self.latent_dim:]

        # Decode to observation space
        y_mean = torch.stack([self.H(mu_traj[t]) for t in range(mu_traj.shape[0])])
        R = torch.diag(torch.exp(self.log_R_diag))
        Hw = self.H.weight
        y_cov = []
        for t in range(sv_traj.shape[0]):
            St = unvech(sv_traj[t], self.latent_dim)
            HSH = Hw.unsqueeze(0) @ St @ Hw.t().unsqueeze(0)
            y_cov.append(HSH + R.unsqueeze(0).expand(bs, -1, -1))
        y_cov = torch.stack(y_cov)
        return y_mean.permute(1, 0, 2), y_cov.permute(1, 0, 2, 3)

    def compute_nll_loss(self, y_true, y_pred, y_cov):
        B, T, D = y_true.shape
        total = 0.0
        for t in range(T):
            c = y_cov[:, t] + 1e-4 * torch.eye(D, device=y_cov.device)
            try:
                total += -torch.distributions.MultivariateNormal(y_pred[:, t], c).log_prob(y_true[:, t]).mean()
            except Exception:
                s = torch.sqrt(torch.diagonal(c, dim1=1, dim2=2) + 1e-6)
                total += -torch.distributions.Normal(y_pred[:, t], s).log_prob(y_true[:, t]).sum(1).mean()
        return total / T

# ======================================================================
# Neural SDE baseline
# ======================================================================

class NeuralSDE(nn.Module):
    def __init__(self, obs_dim=3, hidden_dim=16):
        super().__init__()
        self.obs_dim = obs_dim
        self.init_encoder = nn.Sequential(
            nn.Linear(obs_dim*10, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim))
        self.drift_net = nn.Sequential(
            nn.Linear(obs_dim+1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, obs_dim))
        self.L_dim = obs_dim * (obs_dim + 1) // 2
        self.diff_net = nn.Sequential(
            nn.Linear(obs_dim+1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, self.L_dim))

    def get_initial_state(self, x_seq):
        bs = x_seq.shape[0]; n = min(10, x_seq.shape[1])
        x = x_seq[:, -n:, :].reshape(bs, -1)
        if x.shape[1] < self.obs_dim*10:
            x = torch.cat([x, torch.zeros(bs, self.obs_dim*10 - x.shape[1], device=x.device)], 1)
        return self.init_encoder(x)

    def _tv(self, t, bs):
        if not isinstance(t, torch.Tensor): t = torch.tensor(t, dtype=torch.float32, device=device)
        if t.dim() == 0: return t.unsqueeze(0).expand(bs, 1)
        return t.unsqueeze(1) if t.dim() == 1 else t

    def drift(self, x, t): return self.drift_net(torch.cat([x, self._tv(t, x.shape[0])], -1))

    def diffusion(self, x, t):
        bs = x.shape[0]
        Lp = self.diff_net(torch.cat([x, self._tv(t, bs)], -1))
        L = torch.zeros(bs, self.obs_dim, self.obs_dim, device=x.device)
        idx = 0
        for i in range(self.obs_dim):
            for j in range(i+1):
                L[:, i, j] = nn.functional.softplus(Lp[:, idx]) + 1e-6 if i == j else Lp[:, idx]
                idx += 1
        return L

    def sde_integrate(self, x0, t_span, n_samples=100):
        bs = x0.shape[0]; all_s = []
        for _ in range(n_samples):
            traj = []; xc = x0.clone()
            for i in range(len(t_span)-1):
                dt = (t_span[i+1]-t_span[i]).item()
                dW = torch.randn(bs, self.obs_dim, device=device) * dt**0.5
                xc = xc + self.drift(xc, t_span[i])*dt + torch.bmm(self.diffusion(xc, t_span[i]), dW.unsqueeze(-1)).squeeze(-1)
                traj.append(xc.clone())
            all_s.append(torch.stack(traj))
        return torch.stack(all_s)

    def predict(self, x_seq, t_obs, t_pred, n_samples=100):
        x0 = self.get_initial_state(x_seq)
        ts = t_obs[0, -1:] if t_obs.dim() == 2 else t_obs[-1:]
        tp = t_pred[0] if t_pred.dim() == 2 else t_pred
        t_int = torch.cat([ts, tp])
        samples = self.sde_integrate(x0, t_int, n_samples)
        mean = samples.mean(0)
        T, B, D = mean.shape
        cov = torch.zeros(T, B, D, D, device=device)
        for t in range(T):
            for b in range(B):
                d = samples[:, t, b, :] - mean[t, b]
                cov[t, b] = d.t() @ d / (n_samples-1) + 1e-4 * torch.eye(D, device=device)
        return mean.permute(1,0,2), cov.permute(1,0,2,3), samples

    def compute_nll_from_samples(self, samples, targets):
        ns, T, B, D = samples.shape
        tgt = targets.permute(1, 0, 2)
        total = 0.0
        for t in range(T):
            m = samples[:, t].mean(0)
            c = torch.zeros(B, D, D, device=device)
            for b in range(B):
                d = samples[:, t, b] - m[b]
                c[b] = d.t() @ d / (ns-1) + 1e-4 * torch.eye(D, device=device)
            try:
                total += -torch.distributions.MultivariateNormal(m, c).log_prob(tgt[t]).mean()
            except Exception:
                s = torch.sqrt(torch.diagonal(c, dim1=1, dim2=2) + 1e-6)
                total += -torch.distributions.Normal(m, s).log_prob(tgt[t]).sum(1).mean()
        return total / T

# ======================================================================
# Training
# ======================================================================

def train_model(model, train_loader, val_loader, model_type="UPN",
                num_epochs=50, lr=5e-4, patience=5, n_train_samples=50):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)
    best_val, best_state, pctr = float("inf"), None, 0
    train_losses, val_losses, times = [], [], []

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train(); tl, nb = 0., 0
        for batch in tqdm(train_loader, desc=f"{model_type} Ep {epoch+1}", leave=False):
            x, y, tx, ty = [b.to(device) for b in batch]
            optimizer.zero_grad()
            if model_type == "UPN":
                yp, yc = model.predict(x, tx, ty)
                loss = model.compute_nll_loss(y, yp, yc)
            else:
                ym, yc, samps = model.predict(x, tx, ty, n_samples=n_train_samples)
                loss = model.compute_nll_from_samples(samps, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tl += loss.item(); nb += 1
        tl /= nb; train_losses.append(tl)

        model.eval(); vl_s, vn = 0., 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, tx, ty = [b.to(device) for b in batch]
                if model_type == "UPN":
                    yp, yc = model.predict(x, tx, ty)
                    loss = model.compute_nll_loss(y, yp, yc)
                else:
                    ym, yc, samps = model.predict(x, tx, ty, n_samples=100)
                    loss = model.compute_nll_from_samples(samps, y)
                vl_s += loss.item(); vn += 1
        vl = vl_s / vn; val_losses.append(vl)
        et = time.time() - t0; times.append(et)
        scheduler.step(vl)
        print(f"  {model_type} Ep {epoch+1}: train={tl:.4f}  val={vl:.4f}  ({et:.1f}s)")

        if vl < best_val - 1e-4:
            best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pctr = 0
        else:
            pctr += 1
            if pctr >= patience: print(f"  Early stopping at epoch {epoch+1}"); break
    if best_state: model.load_state_dict(best_state)
    return model, train_losses, val_losses, times

# ======================================================================
# Evaluation
# ======================================================================

def evaluate(model, test_loader, model_type="UPN", n_samples=100):
    model.eval()
    mse_t, nll_t, c95, c90, c80, cnt, inf_t = 0., 0., 0, 0, 0, 0, 0.
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {model_type}"):
            x, y, tx, ty = [b.to(device) for b in batch]
            t0 = time.time()
            if model_type == "UPN":
                yp, yc = model.predict(x, tx, ty)
                nll = model.compute_nll_loss(y, yp, yc)
            else:
                yp, yc, samps = model.predict(x, tx, ty, n_samples=n_samples)
                nll = model.compute_nll_from_samples(samps, y)
            inf_t += time.time() - t0
            B, T, D = yp.shape
            mse_t += ((y - yp)**2).mean().item() * B
            nll_t += nll.item() * B
            std = torch.zeros(B, T, D, device=device)
            for b in range(B):
                for t in range(T):
                    std[b, t] = torch.sqrt(torch.diagonal(yc[b, t]))
            for z, name in [(1.96, "95"), (1.645, "90"), (1.28, "80")]:
                w = ((y >= yp - z*std) & (y <= yp + z*std)).sum().item()
                if name == "95": c95 += w
                elif name == "90": c90 += w
                else: c80 += w
            cnt += y.numel()
    N = len(test_loader.dataset)
    return {"mse": mse_t/N, "nll": nll_t/N, "coverage_95": c95/cnt,
            "coverage_90": c90/cnt, "coverage_80": c80/cnt,
            "inference_time": inf_t/len(test_loader)}

def sample_tradeoff(sde_model, test_loader):
    results = {}
    for ns in [10, 25, 50, 100]:
        t0 = time.time()
        r = evaluate(sde_model, test_loader, "SDE", n_samples=ns)
        results[ns] = {"mse": r["mse"], "coverage_95": r["coverage_95"], "time": time.time()-t0}
        print(f"  {ns} samples: MSE={r['mse']:.4f}, Cov={r['coverage_95']:.2%}, Time={results[ns]['time']:.1f}s")
    return results

# ======================================================================
# Main
# ======================================================================

def main():
    data_path = "data/etth1/etth1_simple.csv"
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Run: python examples/etth1/download_etth1.py")
        return

    print("Loading ETTh1 dataset...")
    tds = ETTh1Dataset(data_path, split="train")
    vds = ETTh1Dataset(data_path, split="val")
    teds = ETTh1Dataset(data_path, split="test")
    obs_dim = tds[0][0].shape[1]

    tl = DataLoader(tds, batch_size=32, shuffle=True)
    vl = DataLoader(vds, batch_size=32)
    tel = DataLoader(teds, batch_size=32)

    # Train UPN
    print("\n" + "="*60 + "\nTRAINING UPN\n" + "="*60)
    upn = LatentUPN(obs_dim=obs_dim, latent_dim=4, hidden_dim=16).to(device)
    print(f"  Parameters: {sum(p.numel() for p in upn.parameters()):,}")
    upn, upn_tl, upn_vl, upn_times = train_model(upn, tl, vl, "UPN")

    # Train Neural SDE
    print("\n" + "="*60 + "\nTRAINING NEURAL SDE\n" + "="*60)
    sde = NeuralSDE(obs_dim=obs_dim, hidden_dim=16).to(device)
    print(f"  Parameters: {sum(p.numel() for p in sde.parameters()):,}")
    sde, sde_tl, sde_vl, sde_times = train_model(sde, tl, vl, "SDE", n_train_samples=50)

    # Evaluate
    print("\n" + "="*60 + "\nEVALUATION\n" + "="*60)
    ur = evaluate(upn, tel, "UPN")
    sr = evaluate(sde, tel, "SDE", n_samples=100)

    print("\n" + "="*80)
    print(f"{'Metric':<25} {'UPN':<15} {'Neural SDE':<15}")
    print("-"*80)
    print(f"{'MSE':<25} {ur['mse']:<15.3f} {sr['mse']:<15.3f}")
    print(f"{'NLL':<25} {ur['nll']:<15.3f} {sr['nll']:<15.3f}")
    print(f"{'95% Coverage':<25} {ur['coverage_95']:<15.3f} {sr['coverage_95']:<15.3f}")
    print(f"{'90% Coverage':<25} {ur['coverage_90']:<15.3f} {sr['coverage_90']:<15.3f}")
    print(f"{'80% Coverage':<25} {ur['coverage_80']:<15.3f} {sr['coverage_80']:<15.3f}")
    print(f"{'Train Time/epoch (s)':<25} {np.mean(upn_times):<15.1f} {np.mean(sde_times):<15.1f}")
    print(f"{'Inference Time/batch (s)':<25} {ur['inference_time']:<15.3f} {sr['inference_time']:<15.3f}")
    print(f"{'Relative Speed':<25} {'1x':<15} {sr['inference_time']/ur['inference_time']:.1f}x")
    print("="*80)

    # Sample trade-off
    print("\nNeural SDE sample-size trade-off:")
    st = sample_tradeoff(sde, tel)
    print(f"\n{'Samples':<10} {'MSE':<12} {'95% Cov':<12} {'Rel. Time':<12}")
    print("-"*46)
    for ns in sorted(st):
        print(f"{ns:<10} {st[ns]['mse']:<12.3f} {st[ns]['coverage_95']:<12.3f} {st[ns]['time']/st[10]['time']:<12.1f}x")
    print(f"{'UPN':<10} {ur['mse']:<12.3f} {ur['coverage_95']:<12.3f} {'-':<12}")

    # Save
    os.makedirs("output_etth1", exist_ok=True)
    torch.save({"upn": ur, "sde": sr, "sample_tradeoff": st,
                "upn_state": upn.state_dict(), "sde_state": sde.state_dict()},
               "output_etth1/etth1_results.pt")
    print("\nResults saved to output_etth1/etth1_results.pt")

if __name__ == "__main__":
    main()
