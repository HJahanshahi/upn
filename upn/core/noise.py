import torch
import torch.nn as nn

class ProcessNoiseNetwork(nn.Module):
    """Neural network for modeling the process noise covariance Q(μ, t)"""
    def __init__(self, state_dim, hidden_dim=64, use_diagonal=True):
        super().__init__()
        self.state_dim = state_dim
        self.use_diagonal = use_diagonal
        
        if use_diagonal:
            # Diagonal covariance (just need to output diagonal elements)
            output_dim = state_dim
        else:
            # Full covariance (need lower triangular elements for Cholesky factor)
            output_dim = state_dim * (state_dim + 1) // 2
            
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
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
        noise_params = self.net(tx)
        
        if self.use_diagonal:
            # Create a diagonal covariance matrix with positive values
            diag_values = torch.exp(noise_params) + 1e-6
            return torch.diag_embed(diag_values)
        else:
            # Create a lower triangular matrix for Cholesky factor
            batch_size = x.shape[0]
            L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)
            
            # Fill in the lower triangular part
            tril_indices = torch.tril_indices(self.state_dim, self.state_dim)
            for b in range(batch_size):
                L[b, tril_indices[0], tril_indices[1]] = noise_params[b]
                
            # Make diagonal elements positive
            batch_indices = torch.arange(batch_size, device=x.device)
            diag_indices = torch.arange(self.state_dim, device=x.device)
            L[batch_indices[:, None], diag_indices, diag_indices] = torch.exp(
                L[batch_indices[:, None], diag_indices, diag_indices]) + 1e-6
                
            # Compute Q = L * L^T
            Q = torch.bmm(L, torch.transpose(L, 1, 2))
            return Q