"""
Uncertainty Propagation Network (UPN).

Simultaneously evolves mean μ(t) and covariance Σ(t) through coupled
differential equations (Eqs. 2–3), with optional Kalman measurement
updates (Section 3.4).
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

from .dynamics import DynamicsNetwork
from .noise import ProcessNoiseNetwork
from .vech import vech, unvech


class UPN(nn.Module):
    """
    Uncertainty Propagation Network.

    Supports two operational modes:
        - Pure prediction (Scenario 1): integrate forward without observations.
        - Adaptive filtering (Scenario 2): incorporate sparse measurements via
          Kalman updates when observations arrive during prediction.

    Args:
        state_dim: Dimension of the full Markovian state.
        obs_dim: Dimension of the observation vector.
        hidden_dim: Hidden units for dynamics and noise networks.
        obs_indices: Which state indices are directly observed.
            Defaults to the first obs_dim indices.
        init_R_diag: Initial diagonal of measurement noise covariance R.
    """

    def __init__(
        self,
        state_dim: int = 6,
        obs_dim: int = 3,
        hidden_dim: int = 64,
        obs_indices: list | None = None,
        init_R_diag: float = 0.01,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.vech_dim = state_dim * (state_dim + 1) // 2

        # Learnable components
        self.dynamics_net = DynamicsNetwork(state_dim, hidden_dim)
        self.noise_net = ProcessNoiseNetwork(state_dim, hidden_dim)

        # Observation model: y = H x + v,  v ~ N(0, R)
        if obs_indices is None:
            obs_indices = list(range(obs_dim))
        H = torch.zeros(obs_dim, state_dim)
        for i, j in enumerate(obs_indices):
            H[i, j] = 1.0
        self.register_buffer("H", H)

        # Learnable measurement noise (log-parameterised for positivity)
        self.log_R_diag = nn.Parameter(
            torch.log(torch.ones(obs_dim) * init_R_diag)
        )

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def compute_jacobian(self, mu: torch.Tensor, t) -> torch.Tensor:
        """Compute J_f = ∂f_θ/∂μ via automatic differentiation."""
        batch_size = mu.shape[0]
        mu_grad = mu.detach().clone().requires_grad_(True)
        f = self.dynamics_net(mu_grad, t)

        J = torch.zeros(batch_size, self.state_dim, self.state_dim, device=mu.device)
        for i in range(self.state_dim):
            grad = torch.autograd.grad(
                outputs=f[:, i].sum(),
                inputs=mu_grad,
                retain_graph=True,
                create_graph=True,
            )[0]
            J[:, i, :] = grad
        return J

    # ------------------------------------------------------------------
    # Kalman update  (Section 3.4.3, Eqs. 11–13)
    # ------------------------------------------------------------------

    def kalman_update(
        self,
        mu_pred: torch.Tensor,
        Sigma_pred: torch.Tensor,
        observation: torch.Tensor,
    ):
        """
        Kalman measurement update.

        Args:
            mu_pred: Predicted mean [batch, state_dim].
            Sigma_pred: Predicted covariance [batch, state_dim, state_dim].
            observation: Observation vector [batch, obs_dim].

        Returns:
            (mu_post, Sigma_post) after incorporating the observation.
        """
        batch_size = mu_pred.shape[0]
        H = self.H.unsqueeze(0).expand(batch_size, -1, -1)
        R = torch.diag(torch.exp(self.log_R_diag)).to(mu_pred.device)
        R = R.unsqueeze(0).expand(batch_size, -1, -1)

        # Innovation covariance S = H Σ H^T + R
        S = torch.bmm(torch.bmm(H, Sigma_pred), H.transpose(1, 2)) + R

        # Kalman gain K = Σ H^T S^{-1}  (Eq. 11)
        try:
            S_inv = torch.inverse(S)
        except Exception:
            S_reg = S + 1e-4 * torch.eye(self.obs_dim, device=S.device).unsqueeze(0)
            S_inv = torch.inverse(S_reg)

        K = torch.bmm(torch.bmm(Sigma_pred, H.transpose(1, 2)), S_inv)

        # State update  (Eq. 12)
        innovation = observation - torch.bmm(H, mu_pred.unsqueeze(-1)).squeeze(-1)
        mu_post = mu_pred + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)

        # Covariance update  (Eq. 13)
        I = torch.eye(self.state_dim, device=K.device).unsqueeze(0).expand(batch_size, -1, -1)
        Sigma_post = torch.bmm(I - torch.bmm(K, H), Sigma_pred)

        return mu_post, Sigma_post

    # ------------------------------------------------------------------
    # Coupled ODE right-hand side  (Eq. 8)
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def forward(self, t, z: torch.Tensor) -> torch.Tensor:
        """
        ODE function for the augmented state z = [μ, vech(Σ)].

        This is called by the ODE solver at each integration step.
        """
        mu = z[:, : self.state_dim]
        Sigma = unvech(z[:, self.state_dim :], self.state_dim)

        mu_dot = self.dynamics_net(mu, t)
        J = self.compute_jacobian(mu, t)
        Q = self.noise_net(mu, t)

        # dΣ/dt = J Σ + Σ J^T + Q   (Eq. 3)
        Sigma_dot = torch.bmm(J, Sigma) + torch.bmm(Sigma, J.transpose(1, 2)) + Q

        return torch.cat([mu_dot, vech(Sigma_dot)], dim=1)

    # ------------------------------------------------------------------
    # Integration with optional measurement updates
    # ------------------------------------------------------------------

    def integrate(
        self,
        initial_mean: torch.Tensor,
        initial_cov: torch.Tensor,
        time_points: torch.Tensor,
        observations: torch.Tensor | None = None,
        update_frequency: int = 5,
        method: str = "dopri5",
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        """
        Integrate coupled mean–covariance ODEs with optional Kalman updates.

        Args:
            initial_mean: [batch, state_dim].
            initial_cov: [batch, state_dim, state_dim].
            time_points: 1-D tensor of evaluation times (first entry = t₀).
            observations: [batch, T-1, obs_dim] ground-truth observations
                at future time points.  ``None`` for pure prediction mode.
            update_frequency: Apply Kalman update every *k* steps
                (only when *observations* is provided).
            method: ODE solver method.
            rtol, atol: Solver tolerances.

        Returns:
            (mean_trajectory, cov_trajectory) each of shape
            [T, batch, state_dim] and [T, batch, state_dim, state_dim].
        """
        mean_traj = [initial_mean]
        cov_traj = [initial_cov]

        current_mean = initial_mean
        current_cov = initial_cov

        for i in range(1, len(time_points)):
            t_span = time_points[i - 1 : i + 1]

            z0 = torch.cat([current_mean, vech(current_cov)], dim=1)
            z_out = odeint(self, z0, t_span, method=method, rtol=rtol, atol=atol)

            z_pred = z_out[-1]
            mu_pred = z_pred[:, : self.state_dim]
            Sigma_pred = unvech(z_pred[:, self.state_dim :], self.state_dim)

            # Kalman update when observations are available
            if observations is not None and (i % update_frequency == 0):
                obs_i = observations[:, i - 1, :]
                mu_pred, Sigma_pred = self.kalman_update(mu_pred, Sigma_pred, obs_i)

            current_mean = mu_pred
            current_cov = Sigma_pred

            mean_traj.append(current_mean)
            cov_traj.append(current_cov)

        return torch.stack(mean_traj), torch.stack(cov_traj)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_nll_loss(
        self,
        true_obs: torch.Tensor,
        pred_mean: torch.Tensor,
        pred_cov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gaussian negative log-likelihood in observation space.

        Args:
            true_obs: [batch, T, obs_dim].
            pred_mean: [T, batch, state_dim].
            pred_cov: [T, batch, state_dim, state_dim].

        Returns:
            Scalar loss.
        """
        T, batch_size, _ = pred_mean.shape
        H = self.H  # [obs_dim, state_dim]
        eps = 1e-4
        I_obs = torch.eye(self.obs_dim, device=pred_cov.device)

        total_nll = 0.0
        for t in range(T):
            Hb = H.unsqueeze(0).expand(batch_size, -1, -1)
            obs_mean = torch.bmm(Hb, pred_mean[t].unsqueeze(-1)).squeeze(-1)
            obs_cov = torch.bmm(torch.bmm(Hb, pred_cov[t]), H.t().unsqueeze(0).expand(batch_size, -1, -1))
            obs_cov = obs_cov + eps * I_obs

            try:
                dist = torch.distributions.MultivariateNormal(obs_mean, obs_cov)
                total_nll = total_nll + (-dist.log_prob(true_obs[:, t, :])).mean()
            except Exception:
                std = torch.sqrt(torch.diagonal(obs_cov, dim1=1, dim2=2))
                dist = torch.distributions.Normal(obs_mean, std)
                total_nll = total_nll + (-dist.log_prob(true_obs[:, t, :]).sum(dim=1)).mean()

        return total_nll / T
