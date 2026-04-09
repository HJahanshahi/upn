"""
Neural network for state-dependent process noise covariance Q_φ(μ, t).

Parameterizes the process noise in Eq. (3) via Cholesky factorisation (Eq. 5):
    Q_φ(μ, t) = L_φ(μ, t) L_φ(μ, t)^T + εI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessNoiseNetwork(nn.Module):
    """
    Neural network producing a positive-definite process noise covariance.

    Outputs a lower-triangular matrix L whose outer product L L^T
    guarantees positive semi-definiteness (Eq. 5 in the paper).

    Args:
        state_dim: Dimension of the state vector.
        hidden_dim: Number of hidden units per layer.
        eps: Small constant added to the diagonal for numerical stability.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.state_dim = state_dim
        self.eps = eps
        self.L_dim = state_dim * (state_dim + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.L_dim),
        )

    def forward(self, mu: torch.Tensor, t) -> torch.Tensor:
        """
        Compute Q_φ(μ, t) = L L^T + εI.

        Args:
            mu: Current mean state [batch, state_dim].
            t: Current time.

        Returns:
            Process noise covariance [batch, state_dim, state_dim].
        """
        batch_size = mu.shape[0]

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=mu.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1) if t.shape[0] == batch_size else t.expand(batch_size, 1)

        L_params = self.net(torch.cat([mu, t], dim=-1))

        # Build lower-triangular L with positive diagonal (softplus)
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=mu.device)
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = F.softplus(L_params[:, idx]) + 1e-6
                else:
                    L[:, i, j] = L_params[:, idx]
                idx += 1

        Q = torch.bmm(L, L.transpose(1, 2))
        Q = Q + self.eps * torch.eye(self.state_dim, device=Q.device).unsqueeze(0)
        return Q
