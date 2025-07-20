import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from .dynamics import DynamicsNetwork
from .noise import ProcessNoiseNetwork
from .ode_solver import compute_jacobian_batched, vech, unvech

class UPNBase(nn.Module):
    """
    Uncertainty Propagation Network (UPN) for Dynamical Systems.
    
    This class implements the main UPN framework which models evolution of 
    both state mean and covariance using coupled differential equations.
    """
    def __init__(self, state_dim, dynamics_net=None, noise_net=None, hidden_dim=64,
                 use_diagonal_cov=True, adjoint=False):
        """
        Initialize a UPN model.
        
        Args:
            state_dim: Dimension of the state vector
            dynamics_net: Neural network for modeling the drift function (if None, create a new one)
            noise_net: Neural network for modeling the process noise (if None, create a new one)
            hidden_dim: Dimension of hidden layers in networks (if creating new ones)
            use_diagonal_cov: Whether to use diagonal approximation for covariance
            adjoint: Whether to use adjoint method for backpropagation
        """
        super().__init__()
        self.state_dim = state_dim
        self.use_diagonal_cov = use_diagonal_cov
        self.adjoint = adjoint
        
        # If networks not provided, create them
        if dynamics_net is None:
            self.dynamics_net = DynamicsNetwork(state_dim, hidden_dim)
        else:
            self.dynamics_net = dynamics_net
            
        if noise_net is None:
            self.noise_net = ProcessNoiseNetwork(state_dim, hidden_dim, use_diagonal=use_diagonal_cov)
        else:
            self.noise_net = noise_net
    
    def forward(self, t, states):
        """
        ODE function for the UPN dynamics.
        
        Args:
            t: Current time
            states: Combined state vector [batch_size, state_dim + cov_dim]
                   where cov_dim depends on use_diagonal_cov
                   
        Returns:
            Derivative of the states (drift)
        """
        batch_size = states.shape[0]
        
        # Extract mean and covariance components
        if self.use_diagonal_cov:
            # For diagonal approximation
            mu = states[:, :self.state_dim]
            sigma_diag = states[:, self.state_dim:]
            
            # Compute drift for mean using dynamics network
            mu_drift = self.dynamics_net(t, mu)
            
            # Compute Jacobian of dynamics with respect to state
            J = compute_jacobian_batched(self.dynamics_net, mu, t)
            
            # Compute process noise covariance
            Q = self.noise_net(t, mu)
            Q_diag = torch.diagonal(Q, dim1=1, dim2=2)
            
            # Compute drift for diagonal covariance
            # For diagonal approximation: dΣ_ii/dt ≈ 2*J_ii*Σ_ii + Q_ii
            J_diag = torch.diagonal(J, dim1=1, dim2=2)
            sigma_drift = 2 * J_diag * sigma_diag + Q_diag
            
            # Combine mean and covariance drifts
            return torch.cat([mu_drift, sigma_drift], dim=1)
        else:
            # For full covariance representation
            mu = states[:, :self.state_dim]
            
            # Get dimension of vectorized covariance
            cov_dim = self.state_dim * (self.state_dim + 1) // 2
            sigma_vech = states[:, self.state_dim:self.state_dim + cov_dim]
            
            # Reconstruct full covariance matrix
            sigma = unvech(sigma_vech, self.state_dim)
            
            # Compute drift for mean
            mu_drift = self.dynamics_net(t, mu)
            
            # Compute Jacobian of dynamics
            J = compute_jacobian_batched(self.dynamics_net, mu, t)
            
            # Compute process noise
            Q = self.noise_net(t, mu)
            
            # Compute drift for covariance: dΣ/dt = JΣ + ΣJ^T + Q
            sigma_drift_full = torch.bmm(J, sigma)
            sigma_drift_full += torch.bmm(sigma, J.transpose(1, 2))
            sigma_drift_full += Q
            
            # Vectorize covariance drift
            sigma_drift_vech = vech(sigma_drift_full)
            
            # Combine mean and vectorized covariance drifts
            return torch.cat([mu_drift, sigma_drift_vech], dim=1)
    
    def integrate(self, z0, t):
        """
        Integrate the UPN dynamics from initial state z0 over time points t.
        
        Args:
            z0: Initial state [batch_size, state_dim + cov_dim]
            t: Time points [time_steps] - must be one-dimensional for torchdiffeq
            
        Returns:
            Solution at each time point [time_steps, batch_size, state_dim + cov_dim]
        """
        from torchdiffeq import odeint, odeint_adjoint
        
        # Ensure t is one-dimensional - torchdiffeq requires this
        if t.ndim > 1:
            # If t has multiple dimensions, use just the first element's time points
            t = t[0]
        
        if self.adjoint:
            return odeint_adjoint(self, z0, t, method='dopri5')
        else:
            return odeint(self, z0, t, method='dopri5')

    def predict(self, initial_state, initial_cov, t_span):
        """
        Make predictions with uncertainty at specified time points.
        
        Args:
            initial_state: Initial state mean [batch_size, state_dim]
            initial_cov: Initial state covariance [batch_size, state_dim, state_dim]
                        or [batch_size, state_dim] if using diagonal approximation
            t_span: Time points to predict at [time_steps] or [batch_size, time_steps]
                    (Only the first batch's time points will be used)
            
        Returns:
            mean_pred: Predicted means [time_steps, batch_size, state_dim]
            cov_pred: Predicted covariances [time_steps, batch_size, state_dim, state_dim]
        """
        batch_size = initial_state.shape[0]
        
        # Ensure t_span is 1D as required by ODE solver
        if t_span.ndim > 1:
            # If time has multiple dimensions, use the first batch element's times
            t_span = t_span[0]
        
        if self.use_diagonal_cov:
            # Prepare initial state for ODE integration
            if initial_cov.dim() == 3:  # If covariance is a full matrix
                cov_diag_0 = torch.diagonal(initial_cov, dim1=1, dim2=2)
            else:  # If already diagonal
                cov_diag_0 = initial_cov
                
            # Concatenate initial state and covariance
            z0 = torch.cat([initial_state, cov_diag_0], dim=1)
            
            # Integrate forward
            z_all = self.integrate(z0, t_span)
            
            # Extract mean and covariance at each time point
            mean_pred = z_all[:, :, :self.state_dim]
            cov_diag_pred = z_all[:, :, self.state_dim:]
            cov_pred = torch.diag_embed(cov_diag_pred)
            
            return mean_pred, cov_pred
        else:
            # For full covariance implementation
            # Vectorize initial covariance
            cov_vech_0 = vech(initial_cov)
            
            # Concatenate initial state and vectorized covariance
            z0 = torch.cat([initial_state, cov_vech_0], dim=1)
            
            # Integrate forward
            z_all = self.integrate(z0, t_span)
            
            # Extract mean and vectorized covariance
            mean_pred = z_all[:, :, :self.state_dim]
            
            # Get dimension of vectorized covariance
            cov_dim = self.state_dim * (self.state_dim + 1) // 2
            cov_vech_pred = z_all[:, :, self.state_dim:self.state_dim + cov_dim]
            
            # Reconstruct full covariance matrices
            time_steps = z_all.shape[0]
            cov_pred = torch.zeros(time_steps, batch_size, self.state_dim, self.state_dim,
                                device=z_all.device)
            
            for t in range(time_steps):
                for b in range(batch_size):
                    cov_pred[t, b] = unvech(cov_vech_pred[t, b], self.state_dim)
            
            return mean_pred, cov_pred

    def compute_loss(self, true_traj, pred_mean, pred_cov):
        """
        Compute negative log likelihood loss.
        
        Args:
            true_traj: Ground truth trajectory [time_steps, batch_size, state_dim]
            pred_mean: Predicted means [time_steps, batch_size, state_dim]
            pred_cov: Predicted covariances [time_steps, batch_size, state_dim, state_dim]
            
        Returns:
            Negative log likelihood loss
        """
        time_steps, batch_size, state_dim = pred_mean.shape
        
        # Reshape tensors for easier processing
        true_traj_flat = true_traj.reshape(time_steps * batch_size, state_dim)
        pred_mean_flat = pred_mean.reshape(time_steps * batch_size, state_dim)
        pred_cov_flat = pred_cov.reshape(time_steps * batch_size, state_dim, state_dim)
        
        # Compute negative log likelihood for each prediction
        nll = torch.zeros(time_steps * batch_size, device=pred_mean.device)
        
        for i in range(time_steps * batch_size):
            try:
                # Use multivariate normal distribution
                dist = MultivariateNormal(
                    pred_mean_flat[i], pred_cov_flat[i] + 1e-6 * torch.eye(state_dim, device=pred_mean.device)
                )
                nll[i] = -dist.log_prob(true_traj_flat[i])
            except:
                # Handle numerical issues (e.g., non-PSD covariance)
                # Use a simple MSE as fallback
                nll[i] = ((true_traj_flat[i] - pred_mean_flat[i]) ** 2).sum() * 0.5
                
        return nll.mean()