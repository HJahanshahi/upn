"""
CubeSat Trajectory Prediction — Training Script

Trains UPN and baseline models (Latent ODE, Ensemble Neural ODE, Neural SDE)
on CubeSat free-floating dynamics data.

Reproduces Section 5.3 of:
    Jahanshahi & Zhu (2026), Neurocomputing 677, 133134.

Usage:
    python train_cubesat.py --data_path /path/to/experimental_data
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from upn import UPN
from upn.baselines import LatentODE, EnsembleNeuralODE, NeuralSDE

# ======================================================================
# Reproducibility
# ======================================================================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Data utilities
# ======================================================================


def load_cubesat_data(data_path: str):
    """Load CubeSat CSV files.  Each file has columns: time, theta_deg, x, y."""
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")

    trajectories = []
    for f in csv_files:
        df = pd.read_csv(f, header=None)
        time = df.iloc[:, 0].values
        theta_rad = np.deg2rad(df.iloc[:, 1].values)
        x = df.iloc[:, 2].values
        y = df.iloc[:, 3].values
        trajectories.append({"time": time, "states": np.stack([x, y, theta_rad], axis=1)})
    return trajectories


class CubeSatDataset(Dataset):
    """Sliding-window dataset over CubeSat trajectories."""

    def __init__(self, trajectories, history_length=10, future_length=20):
        self.windows = []
        for traj_idx, traj in enumerate(trajectories):
            T = len(traj["time"])
            total = history_length + future_length
            if T < total:
                continue
            for s in range(T - total + 1):
                self.windows.append({
                    "trajectory_id": traj_idx,
                    "history_states": traj["states"][s : s + history_length],
                    "history_time": traj["time"][s : s + history_length],
                    "future_states": traj["states"][s + history_length : s + total],
                    "future_time": traj["time"][s + history_length : s + total],
                })
        if not self.windows:
            raise ValueError("No valid windows could be created")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return (
            torch.tensor(w["history_states"], dtype=torch.float32),
            torch.tensor(w["history_time"], dtype=torch.float32),
            torch.tensor(w["future_states"], dtype=torch.float32),
            torch.tensor(w["future_time"], dtype=torch.float32),
        )


def initialize_state_with_velocity(history_states, history_time):
    """Construct 6-D Markovian state [position, velocity] from history."""
    position = history_states[:, -1, :]
    if history_states.shape[1] >= 2:
        dt = (history_time[:, -1] - history_time[:, -2]).unsqueeze(-1).clamp(min=1e-6)
        velocity = (history_states[:, -1, :] - history_states[:, -2, :]) / dt
    else:
        velocity = torch.zeros_like(position)
    return torch.cat([position, velocity], dim=-1)


# ======================================================================
# Training loops
# ======================================================================


def _training_loop(model, train_loader, val_loader, forward_fn, num_epochs, lr, weight_decay, patience=15):
    """Generic train/val loop with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    best_val, best_state, counter = float("inf"), None, 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss = forward_fn(model, *batch, training=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        train_losses.append(epoch_loss / n)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                vl += forward_fn(model, *batch, training=False).item()
                vn += 1
        avg_vl = vl / vn
        val_losses.append(avg_vl)
        scheduler.step(avg_vl)

        print(f"  Epoch {epoch+1}: train={train_losses[-1]:.6f}  val={avg_vl:.6f}")
        if avg_vl < best_val:
            best_val = avg_vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
            print(f"  ✓ New best (val={best_val:.6f})")
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


# --- UPN ---

def _upn_forward(model, hist_s, hist_t, fut_s, fut_t, *, training, state_dim=6, update_frequency=5):
    batch_size = hist_s.shape[0]
    initial_mean = initialize_state_with_velocity(hist_s, hist_t)
    initial_cov = torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1) * 0.01
    full_time = torch.cat([hist_t[0, -1:], fut_t[0]])
    mean_pred, cov_pred = model.integrate(
        initial_mean, initial_cov, full_time,
        observations=fut_s, update_frequency=update_frequency,
    )
    return model.compute_nll_loss(fut_s, mean_pred[1:], cov_pred[1:])


def train_upn(train_loader, val_loader, state_dim=6, hidden_dim=32, update_frequency=5, **kw):
    model = UPN(state_dim=state_dim, obs_dim=3, hidden_dim=hidden_dim).to(device)
    fwd = lambda m, *args, **kwargs: _upn_forward(m, *args, state_dim=state_dim, update_frequency=update_frequency, **kwargs)
    return _training_loop(model, train_loader, val_loader, fwd, **kw)


# --- Latent ODE ---

def _latent_forward(model, hist_s, hist_t, fut_s, fut_t, *, training):
    obs_mean, obs_std = model(hist_s, hist_t, fut_t)
    obs_mean, obs_std = obs_mean.permute(1, 0, 2), obs_std.permute(1, 0, 2)
    return -torch.distributions.Normal(obs_mean, obs_std).log_prob(fut_s).mean()


def train_latent_ode(train_loader, val_loader, state_dim=6, hidden_dim=32, **kw):
    model = LatentODE(obs_dim=3, latent_dim=state_dim, hidden_dim=hidden_dim).to(device)
    return _training_loop(model, train_loader, val_loader, _latent_forward, **kw)


# --- Ensemble ---

def _ensemble_forward(model, hist_s, hist_t, fut_s, fut_t, *, training, state_dim=6, model_idx=0):
    initial_state = initialize_state_with_velocity(hist_s, hist_t)
    t_span = torch.cat([hist_t[0, -1:], fut_t[0]])
    def ode_func(t, s):
        return model.forward(t, s, model_idx=model_idx)
    traj = torch.zeros(1)  # placeholder
    from torchdiffeq import odeint as _odeint
    traj = _odeint(ode_func, initial_state, t_span, method="dopri5", rtol=1e-4, atol=1e-6)
    pred = traj[1:, :, :3].permute(1, 0, 2)
    return nn.MSELoss()(pred, fut_s)


def train_ensemble(train_loader, val_loader, state_dim=6, hidden_dim=32, n_ensemble=5, **kw):
    model = EnsembleNeuralODE(state_dim=state_dim, hidden_dim=hidden_dim, n_ensemble=n_ensemble).to(device)
    results = {}
    all_train, all_val = [], []
    for idx in range(n_ensemble):
        print(f"  Training ensemble member {idx+1}/{n_ensemble}")
        fwd = lambda m, *args, mi=idx, **kwargs: _ensemble_forward(m, *args, state_dim=state_dim, model_idx=mi, **kwargs)
        _, tl, vl = _training_loop(model, train_loader, val_loader, fwd, **kw)
        all_train.append(tl)
        all_val.append(vl)
    return model, all_train, all_val


# --- Neural SDE ---

def _sde_forward(model, hist_s, hist_t, fut_s, fut_t, *, training, state_dim=6, n_samples=20):
    initial_state = initialize_state_with_velocity(hist_s, hist_t)
    t_span = torch.cat([hist_t[0, -1:], fut_t[0]])
    samples = model.sde_integrate(initial_state, t_span, n_samples=n_samples if training else 10, dt=0.01)
    mean_pred = samples[:, 1:, :, :3].mean(dim=0).permute(1, 0, 2)
    return nn.MSELoss()(mean_pred, fut_s)


def train_neural_sde(train_loader, val_loader, state_dim=6, hidden_dim=32, n_samples=20, **kw):
    model = NeuralSDE(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    fwd = lambda m, *args, **kwargs: _sde_forward(m, *args, state_dim=state_dim, n_samples=n_samples, **kwargs)
    return _training_loop(model, train_loader, val_loader, fwd, **kw)


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/cubesat")
    parser.add_argument("--save_dir", type=str, default="trained_models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading CubeSat data...")
    trajectories = load_cubesat_data(args.data_path)
    n = len(trajectories)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    train_ds = CubeSatDataset(trajectories[:n_train])
    val_ds = CubeSatDataset(trajectories[n_train : n_train + n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    common = dict(num_epochs=args.epochs, lr=args.lr, weight_decay=1e-4)

    print("\n[1/4] Training UPN")
    m, tl, vl = train_upn(train_loader, val_loader, hidden_dim=args.hidden_dim, **common)
    torch.save({"model_state_dict": m.state_dict(), "train_losses": tl, "val_losses": vl},
               os.path.join(args.save_dir, "upn_model.pt"))

    print("\n[2/4] Training Latent ODE")
    m, tl, vl = train_latent_ode(train_loader, val_loader, hidden_dim=args.hidden_dim, **common)
    torch.save({"model_state_dict": m.state_dict(), "train_losses": tl, "val_losses": vl},
               os.path.join(args.save_dir, "latent_ode_model.pt"))

    print("\n[3/4] Training Ensemble ODE")
    m, tl, vl = train_ensemble(train_loader, val_loader, hidden_dim=args.hidden_dim, n_ensemble=5, **common)
    torch.save({"model_state_dict": m.state_dict(), "train_losses": tl, "val_losses": vl},
               os.path.join(args.save_dir, "ensemble_model.pt"))

    print("\n[4/4] Training Neural SDE")
    m, tl, vl = train_neural_sde(train_loader, val_loader, hidden_dim=args.hidden_dim, **common)
    torch.save({"model_state_dict": m.state_dict(), "train_losses": tl, "val_losses": vl},
               os.path.join(args.save_dir, "sde_model.pt"))

    print("\n✓ All models saved to", args.save_dir)


if __name__ == "__main__":
    main()
