import torch
import torch.nn as nn

class DynamicsNetwork(nn.Module):
    """Neural network for modeling the drift function f(μ, t)"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, t, x):
        # Handle scalar time input - expand to match batch size
        batch_size = x.shape[0]
        
        # Convert scalar time to tensor if needed
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=x.device)
            
        # Ensure time has proper shape [batch_size, 1]
        if t.dim() == 0:  # scalar
            t = t.expand(batch_size).unsqueeze(-1)  # [batch_size, 1]
        elif t.dim() == 1 and t.numel() == 1:  # [1]
            t = t.expand(batch_size).unsqueeze(-1)  # [batch_size, 1]
        elif t.dim() == 1 and t.numel() == batch_size:  # [batch_size]
            t = t.unsqueeze(-1)  # [batch_size, 1]
            
        # Concatenate time and state
        tx = torch.cat([t, x], dim=1)
        return self.net(tx)