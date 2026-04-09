"""
CubeSat Trajectory Prediction — Evaluation Script

Evaluates trained UPN and baseline models, computes metrics (MSE, coverage),
and generates visualisation figures (Figs. 27–31 in the paper).

Usage:
    python evaluate_cubesat.py --data_path /path/to/experimental_data
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset

from upn import UPN
from upn.baselines import LatentODE, EnsembleNeuralODE, NeuralSDE

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Data (same helpers as training script)
# ======================================================================

def load_cubesat_data(data_path):
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")
    trajs = []
    for f in csv_files:
        df = pd.read_csv(f, header=None)
        trajs.append({
            "time": df.iloc[:, 0].values,
            "states": np.stack([df.iloc[:, 2].values, df.iloc[:, 3].values,
                                np.deg2rad(df.iloc[:, 1].values)], axis=1),
        })
    return trajs


class CubeSatDataset(Dataset):
    def __init__(self, trajectories, history_length=10, future_length=20):
        self.windows = []
        for ti, t in enumerate(trajectories):
            T = len(t["time"]); total = history_length + future_length
            if T < total: continue
            for s in range(T - total + 1):
                self.windows.append({
                    "trajectory_id": ti,
                    "history_states": t["states"][s:s+history_length],
                    "history_time": t["time"][s:s+history_length],
                    "future_states": t["states"][s+history_length:s+total],
                    "future_time": t["time"][s+history_length:s+total],
                })

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        return (torch.tensor(w["history_states"], dtype=torch.float32),
                torch.tensor(w["history_time"], dtype=torch.float32),
                torch.tensor(w["future_states"], dtype=torch.float32),
                torch.tensor(w["future_time"], dtype=torch.float32),
                w["trajectory_id"])

    def get_diverse_samples(self, n=4):
        seen, out = set(), []
        for i, w in enumerate(self.windows):
            if w["trajectory_id"] not in seen:
                out.append(i); seen.add(w["trajectory_id"])
                if len(out) >= n: break
        return out


def init_state(hist_s, hist_t):
    pos = hist_s[:, -1, :]
    if hist_s.shape[1] >= 2:
        dt = (hist_t[:, -1] - hist_t[:, -2]).unsqueeze(-1).clamp(min=1e-6)
        vel = (hist_s[:, -1, :] - hist_s[:, -2, :]) / dt
    else:
        vel = torch.zeros_like(pos)
    return torch.cat([pos, vel], dim=-1)


# ======================================================================
# Evaluation helpers
# ======================================================================


def evaluate_upn(model, loader, dataset, state_dim=6, update_freq=5, use_updates=True, n_viz=4):
    model.eval()
    mse_tot, nll_tot, n_samp = 0., 0., 0
    cov_count, cov_total = 0, 0

    with torch.no_grad():
        for hs, ht, fs, ft, _ in loader:
            hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
            bs = hs.shape[0]
            mu0 = init_state(hs, ht)
            S0 = torch.eye(state_dim, device=device).unsqueeze(0).expand(bs,-1,-1)*0.01
            t_full = torch.cat([ht[0,-1:], ft[0]])

            obs = fs if use_updates else None
            mp, cp = model.integrate(mu0, S0, t_full, observations=obs,
                                     update_frequency=update_freq if use_updates else 999)
            mp, cp = mp[1:], cp[1:]

            pred_pos = mp[:,:,:3].permute(1,0,2)
            mse_tot += ((pred_pos - fs)**2).mean().item() * bs

            # coverage
            std_pos = torch.sqrt(torch.diagonal(cp[:,:,:3,:3], dim1=2, dim2=3)).permute(1,0,2)
            within = (fs >= pred_pos - 2*std_pos) & (fs <= pred_pos + 2*std_pos)
            cov_count += within.sum().item()
            cov_total += within.numel()
            n_samp += bs

    # viz examples
    viz = []
    indices = dataset.get_diverse_samples(n_viz)
    subset = Subset(dataset, indices)
    vl = DataLoader(subset, batch_size=1)
    with torch.no_grad():
        for hs, ht, fs, ft, tid in vl:
            hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
            mu0 = init_state(hs, ht)
            S0 = torch.eye(state_dim, device=device).unsqueeze(0)*0.01
            t_full = torch.cat([ht[0,-1:], ft[0]])
            obs = fs if use_updates else None
            mp, cp = model.integrate(mu0, S0, t_full, observations=obs,
                                     update_frequency=update_freq if use_updates else 999)
            mp, cp = mp[1:], cp[1:]
            viz.append({
                "history": hs[0].cpu().numpy(),
                "future": fs[0].cpu().numpy(),
                "pred_mean": mp[:,:,:3].permute(1,0,2)[0].cpu().numpy(),
                "pred_std": torch.sqrt(torch.diagonal(cp[:,:,:3,:3],dim1=2,dim2=3)).permute(1,0,2)[0].cpu().numpy(),
                "time": torch.cat([ht[0], ft[0]]).cpu().numpy(),
                "trajectory_id": tid[0].item(),
                "uses_updates": use_updates,
                "update_frequency": update_freq if use_updates else None,
            })
            if len(viz) >= n_viz: break

    return mse_tot/n_samp, cov_count/cov_total, viz


def evaluate_latent_ode(model, loader, dataset, n_viz=4):
    model.eval()
    mse_tot, n_samp, cov_count, cov_total = 0., 0, 0, 0
    with torch.no_grad():
        for hs, ht, fs, ft, _ in loader:
            hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
            om, os_ = model(hs, ht, ft)
            om, os_ = om.permute(1,0,2), os_.permute(1,0,2)
            mse_tot += ((om-fs)**2).mean().item() * hs.shape[0]
            within = (fs >= om-2*os_) & (fs <= om+2*os_)
            cov_count += within.sum().item(); cov_total += within.numel()
            n_samp += hs.shape[0]
    # viz
    viz = []
    for hs, ht, fs, ft, tid in DataLoader(Subset(dataset, dataset.get_diverse_samples(n_viz)), batch_size=1):
        hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
        with torch.no_grad():
            om, os_ = model(hs, ht, ft)
            om, os_ = om.permute(1,0,2), os_.permute(1,0,2)
        viz.append({"history": hs[0].cpu().numpy(), "future": fs[0].cpu().numpy(),
                    "pred_mean": om[0].cpu().numpy(), "pred_std": os_[0].cpu().numpy(),
                    "time": torch.cat([ht[0],ft[0]]).cpu().numpy(), "trajectory_id": tid[0].item()})
        if len(viz) >= n_viz: break
    return mse_tot/n_samp, cov_count/cov_total, viz


def evaluate_ensemble(model, loader, dataset, state_dim=6, n_viz=4):
    model.eval()
    mse_tot, n_samp, cov_count, cov_total = 0., 0, 0, 0
    with torch.no_grad():
        for hs, ht, fs, ft, _ in loader:
            hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
            s0 = init_state(hs, ht)
            ts = torch.cat([ht[0,-1:], ft[0]])
            preds = model.predict_ensemble(s0, ts)[:, 1:, :, :3]
            m, s = model.get_mean_and_std(preds)
            mp = m.permute(1,0,2)
            sp = s.permute(1,0,2)
            mse_tot += ((mp-fs)**2).mean().item() * hs.shape[0]
            within = (fs >= mp-2*sp) & (fs <= mp+2*sp)
            cov_count += within.sum().item(); cov_total += within.numel()
            n_samp += hs.shape[0]
    viz = []
    for hs, ht, fs, ft, tid in DataLoader(Subset(dataset, dataset.get_diverse_samples(n_viz)), batch_size=1):
        hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
        with torch.no_grad():
            s0 = init_state(hs, ht)
            ts = torch.cat([ht[0,-1:], ft[0]])
            preds = model.predict_ensemble(s0, ts)[:, 1:, :, :3]
            m, s = model.get_mean_and_std(preds)
        viz.append({"history": hs[0].cpu().numpy(), "future": fs[0].cpu().numpy(),
                    "pred_mean": m.permute(1,0,2)[0].cpu().numpy(),
                    "pred_std": s.permute(1,0,2)[0].cpu().numpy(),
                    "time": torch.cat([ht[0],ft[0]]).cpu().numpy(), "trajectory_id": tid[0].item()})
        if len(viz) >= n_viz: break
    return mse_tot/n_samp, cov_count/cov_total, viz


def evaluate_neural_sde(model, loader, dataset, state_dim=6, n_samples=100, n_viz=4):
    model.eval()
    mse_tot, n_samp, cov_count, cov_total = 0., 0, 0, 0
    with torch.no_grad():
        for hs, ht, fs, ft, _ in loader:
            hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
            s0 = init_state(hs, ht)
            ts = torch.cat([ht[0,-1:], ft[0]])
            samps = model.sde_integrate(s0, ts, n_samples=n_samples)[:, 1:, :, :3]
            m, s = model.get_statistics(samps)
            mp = m.permute(1,0,2); sp = s.permute(1,0,2)
            mse_tot += ((mp-fs)**2).mean().item() * hs.shape[0]
            within = (fs >= mp-2*sp) & (fs <= mp+2*sp)
            cov_count += within.sum().item(); cov_total += within.numel()
            n_samp += hs.shape[0]
    viz = []
    for hs, ht, fs, ft, tid in DataLoader(Subset(dataset, dataset.get_diverse_samples(n_viz)), batch_size=1):
        hs, ht, fs, ft = [x.to(device) for x in [hs, ht, fs, ft]]
        with torch.no_grad():
            s0 = init_state(hs, ht)
            ts = torch.cat([ht[0,-1:], ft[0]])
            samps = model.sde_integrate(s0, ts, n_samples=n_samples)[:, 1:, :, :3]
            m, s = model.get_statistics(samps)
        viz.append({"history": hs[0].cpu().numpy(), "future": fs[0].cpu().numpy(),
                    "pred_mean": m.permute(1,0,2)[0].cpu().numpy(),
                    "pred_std": s.permute(1,0,2)[0].cpu().numpy(),
                    "time": torch.cat([ht[0],ft[0]]).cpu().numpy(), "trajectory_id": tid[0].item()})
        if len(viz) >= n_viz: break
    return mse_tot/n_samp, cov_count/cov_total, viz


def evaluate_baseline(loader):
    mse_tot, n = 0., 0
    for hs, ht, fs, ft, _ in loader:
        bs = hs.shape[0]
        if hs.shape[1] >= 2:
            vel = hs[:,-1,:] - hs[:,-2,:]
            dt = (ht[:,-1] - ht[:,-2]).unsqueeze(-1).unsqueeze(-1)
            preds = []
            for i in range(fs.shape[1]):
                fdt = (ft[:,i] - ht[:,-1]).unsqueeze(-1).unsqueeze(-1)
                preds.append(hs[:,-1,:].unsqueeze(1) + vel.unsqueeze(1) * (fdt / dt))
            mse_tot += ((torch.cat(preds, dim=1) - fs)**2).mean().item() * bs
            n += bs
    return mse_tot / n


# ======================================================================
# Visualisation
# ======================================================================


def plot_method(viz_examples, method_name, color, save_path=None):
    """Plot 4 trajectories × 3 states for one method."""
    names = ["X Position (m)", "Y Position (m)", "Yaw Angle (rad)"]
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    for ei, ex in enumerate(viz_examples[:4]):
        hl = len(ex["history"])
        ht = ex["time"][:hl]; ft = ex["time"][hl:]
        for si, sn in enumerate(names):
            ax = axes[ei, si]
            ax.plot(ht, ex["history"][:,si], "b-", lw=2.5, alpha=0.8,
                    label="History" if ei==0 else "")
            ax.plot(ft, ex["future"][:,si], "k-", lw=2.5, label="Ground Truth" if ei==0 else "")
            ax.plot(ft, ex["pred_mean"][:,si], color=color, lw=2.5,
                    label=method_name if ei==0 else "")
            ax.fill_between(ft, ex["pred_mean"][:,si]-2*ex["pred_std"][:,si],
                            ex["pred_mean"][:,si]+2*ex["pred_std"][:,si],
                            color=color, alpha=0.25, label="95% CI" if ei==0 else "")
            if ex.get("uses_updates") and ex.get("update_frequency"):
                for ui in range(0, len(ft), ex["update_frequency"]):
                    if ui < len(ft):
                        ax.axvline(ft[ui], color="orange", ls=":", alpha=0.8, lw=1.5,
                                   label="Updates" if ei==0 and ui==0 else "")
            ax.axvline(ht[-1], color="r", ls="--", alpha=0.7, lw=2)
            ax.grid(True, alpha=0.3)
            if ei==0: ax.set_title(sn, fontweight="bold")
            if si==0: ax.set_ylabel(f"Trajectory {ex.get('trajectory_id',ei)+1}", fontweight="bold")
            if ei==3: ax.set_xlabel("Time (s)")
            if ei==0 and si==0: ax.legend(fontsize=10)
    plt.suptitle(f"{method_name}", fontsize=20, fontweight="bold")
    plt.tight_layout(pad=2.0, h_pad=2.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved {save_path}")
    plt.close()


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/cubesat")
    parser.add_argument("--model_dir", type=str, default="trained_models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    trajs = load_cubesat_data(args.data_path)
    n = len(trajs); nt = int(0.7*n); nv = int(0.15*n)
    test_ds = CubeSatDataset(trajs[nt+nv:])
    test_loader = DataLoader(test_ds, batch_size=16)

    sd, hd = 6, 32  # state_dim, hidden_dim

    # Load models
    upn = UPN(state_dim=sd, obs_dim=3, hidden_dim=hd).to(device)
    upn.load_state_dict(torch.load(os.path.join(args.model_dir, "upn_model.pt"), weights_only=False)["model_state_dict"])

    latent = LatentODE(obs_dim=3, latent_dim=sd, hidden_dim=hd).to(device)
    latent.load_state_dict(torch.load(os.path.join(args.model_dir, "latent_ode_model.pt"), weights_only=False)["model_state_dict"])

    ens = EnsembleNeuralODE(state_dim=sd, hidden_dim=hd, n_ensemble=5).to(device)
    ens.load_state_dict(torch.load(os.path.join(args.model_dir, "ensemble_model.pt"), weights_only=False)["model_state_dict"])

    sde = NeuralSDE(state_dim=sd, hidden_dim=hd).to(device)
    sde.load_state_dict(torch.load(os.path.join(args.model_dir, "sde_model.pt"), weights_only=False)["model_state_dict"])

    # Evaluate
    print("Baseline...")
    bl_mse = evaluate_baseline(test_loader)

    print("UPN (no updates)...")
    upn_mse, upn_cov, upn_viz = evaluate_upn(upn, test_loader, test_ds, use_updates=False)

    print("Latent ODE...")
    lat_mse, lat_cov, lat_viz = evaluate_latent_ode(latent, test_loader, test_ds)

    print("Ensemble...")
    ens_mse, ens_cov, ens_viz = evaluate_ensemble(ens, test_loader, test_ds)

    print("Neural SDE...")
    sde_mse, sde_cov, sde_viz = evaluate_neural_sde(sde, test_loader, test_ds)

    print("UPN (with updates)...")
    upn_mse_u, upn_cov_u, upn_viz_u = evaluate_upn(upn, test_loader, test_ds, use_updates=True)

    # Print results table
    print("\n" + "="*90)
    print(f"{'Method':<20} {'MSE':<12} {'95% Coverage':<15}")
    print("-"*90)
    print(f"{'UPN (no upd.)':<20} {upn_mse:<12.6f} {upn_cov:<15.1%}")
    print(f"{'UPN (with upd.)':<20} {upn_mse_u:<12.6f} {upn_cov_u:<15.1%}")
    print(f"{'Latent ODE':<20} {lat_mse:<12.6f} {lat_cov:<15.1%}")
    print(f"{'Ensemble':<20} {ens_mse:<12.6f} {ens_cov:<15.1%}")
    print(f"{'Neural SDE':<20} {sde_mse:<12.6f} {sde_cov:<15.1%}")
    print(f"{'Baseline':<20} {bl_mse:<12.6f} {'N/A':<15}")
    print(f"\nUPN MSE reduction from updates: {(upn_mse-upn_mse_u)/upn_mse*100:.1f}%")
    print("="*90)

    # Generate figures
    plot_method(upn_viz, "UPN - Without Updates", "#1f77b4", os.path.join(args.results_dir, "upn_no_updates.png"))
    plot_method(upn_viz_u, "UPN - With Updates", "#1f77b4", os.path.join(args.results_dir, "upn_with_updates.png"))
    plot_method(lat_viz, "Latent ODE", "#2ca02c", os.path.join(args.results_dir, "latent_ode.png"))
    plot_method(ens_viz, "Ensemble", "#9467bd", os.path.join(args.results_dir, "ensemble.png"))
    plot_method(sde_viz, "Neural SDE", "#ff7f0e", os.path.join(args.results_dir, "neural_sde.png"))

    print("\n✓ Evaluation complete. Results in", args.results_dir)


if __name__ == "__main__":
    main()
