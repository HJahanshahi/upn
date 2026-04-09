"""
Baseline models for comparison with UPN.

- LatentODE: RNN encoder + latent ODE dynamics + Gaussian decoder.
- EnsembleNeuralODE: Ensemble of independently trained Neural ODEs.
- NeuralSDE: Neural SDE with Euler-Maruyama integration.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..core.dynamics import DynamicsNetwork


# ======================================================================
# Latent ODE
# ======================================================================


class LatentODE(nn.Module):
    """Latent ODE with GRU encoder and Gaussian output."""

    def __init__(self, obs_dim: int = 3, latent_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder_rnn = nn.GRU(
            input_size=obs_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True
        )
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder_mean = nn.Linear(latent_dim, obs_dim)
        self.decoder_logvar = nn.Linear(latent_dim, obs_dim)

    def encode(self, history_states):
        _, hidden = self.encoder_rnn(history_states)
        hidden = hidden[-1]
        return self.encoder_mean(hidden), self.encoder_logvar(hidden)

    def ode_func(self, t, z):
        batch_size = z.shape[0]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=z.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1) if t.shape[0] == batch_size else t.expand(batch_size, 1)
        return self.dynamics_net(torch.cat([z, t], dim=-1))

    def decode(self, z):
        obs_mean = self.decoder_mean(z)
        obs_std = torch.exp(0.5 * self.decoder_logvar(z)).clamp(min=1e-4)
        return obs_mean, obs_std

    def forward(self, history_states, history_time, future_time):
        z_mean, _ = self.encode(history_states)
        z0 = z_mean

        t0 = history_time[0, -1]
        t_eval = torch.cat([t0.unsqueeze(0), future_time[0]])

        z_traj = odeint(self.ode_func, z0, t_eval, method="dopri5", rtol=1e-4, atol=1e-6)
        z_traj = z_traj[1:]

        obs_mean_list, obs_std_list = [], []
        for t_idx in range(z_traj.shape[0]):
            m, s = self.decode(z_traj[t_idx])
            obs_mean_list.append(m)
            obs_std_list.append(s)

        return torch.stack(obs_mean_list), torch.stack(obs_std_list)


# ======================================================================
# Ensemble Neural ODE
# ======================================================================


class EnsembleNeuralODE(nn.Module):
    """Ensemble of independently trained Neural ODEs."""

    def __init__(self, state_dim: int = 6, hidden_dim: int = 64, n_ensemble: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.n_ensemble = n_ensemble
        self.ensemble_models = nn.ModuleList(
            [DynamicsNetwork(state_dim, hidden_dim) for _ in range(n_ensemble)]
        )

    def forward(self, t, state, model_idx=0):
        return self.ensemble_models[model_idx](state, t)

    def predict_ensemble(self, initial_state, t_span):
        all_preds = []
        for idx in range(self.n_ensemble):
            def ode_func(t, state, _idx=idx):
                return self.forward(t, state, model_idx=_idx)

            traj = odeint(ode_func, initial_state, t_span, method="dopri5", rtol=1e-4, atol=1e-6)
            all_preds.append(traj)
        return torch.stack(all_preds)

    @staticmethod
    def get_mean_and_std(predictions):
        return predictions.mean(dim=0), predictions.std(dim=0)


# ======================================================================
# Neural SDE
# ======================================================================


class NeuralSDE(nn.Module):
    """Neural SDE with diagonal diffusion and Euler-Maruyama integration."""

    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim

        self.drift_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def _time_input(self, t, y):
        batch_size = y.shape[0]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=y.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        return torch.cat([y, t], dim=-1)

    def f(self, t, y):
        return self.drift_net(self._time_input(t, y))

    def g(self, t, y):
        sigma = torch.sigmoid(self.diffusion_net(self._time_input(t, y))) * 0.3
        return torch.diag_embed(sigma)

    def sde_integrate(self, y0, t_span, n_samples: int = 100, dt: float = 0.01):
        """Euler-Maruyama integration producing *n_samples* trajectories."""
        batch_size = y0.shape[0]
        device = y0.device
        all_samples = []

        for _ in range(n_samples):
            trajectory = [y0.clone()]
            current_y = y0.clone()

            for i in range(len(t_span) - 1):
                t_cur = t_span[i]
                dt_actual = (t_span[i + 1] - t_span[i]).item()
                n_sub = max(1, int(dt_actual / dt))
                dt_sub = dt_actual / n_sub

                for _ in range(n_sub):
                    drift = self.f(t_cur, current_y)
                    diffusion = self.g(t_cur, current_y)
                    dW = torch.randn(batch_size, self.state_dim, device=device) * (dt_sub ** 0.5)
                    current_y = current_y + drift * dt_sub + torch.bmm(diffusion, dW.unsqueeze(-1)).squeeze(-1)
                    t_cur = t_cur + dt_sub

                trajectory.append(current_y.clone())
            all_samples.append(torch.stack(trajectory))

        return torch.stack(all_samples)

    @staticmethod
    def get_statistics(samples):
        return samples.mean(dim=0), samples.std(dim=0)
