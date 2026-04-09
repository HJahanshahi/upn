"""
Neural network for mean dynamics f_θ(μ, t).

Parameterizes the right-hand side of Eq. (2):
    dμ(t)/dt = f_θ(μ(t), t)
"""

import torch
import torch.nn as nn


class DynamicsNetwork(nn.Module):
    """
    Neural network that maps (state, time) → state derivative.
    
    Used as the drift function in UPN's coupled ODE system.
    Default architecture: 2-layer MLP with Tanh activations.
    
    Args:
        state_dim: Dimension of the state vector.
        hidden_dim: Number of hidden units per layer.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, t) -> torch.Tensor:
        """
        Compute dμ/dt = f_θ(μ, t).

        Args:
            state: Current mean state [batch, state_dim].
            t: Current time (scalar, 0-d tensor, or [batch] tensor).

        Returns:
            State derivative [batch, state_dim].
        """
        batch_size = state.shape[0]

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=state.device)

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1) if t.shape[0] == batch_size else t.expand(batch_size, 1)

        return self.net(torch.cat([state, t], dim=-1))
