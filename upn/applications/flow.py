import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.datasets import make_moons, make_swiss_roll, make_blobs

from upn.core import UPN, DynamicsNetwork, ProcessNoiseNetwork

class UPNFlow(nn.Module):
    """UPN-based Normalizing Flow model for density estimation with uncertainty"""
    def __init__(self, dim, hidden_dim=64, use_diagonal_cov=True, time_length=0.5):
        super().__init__()
        self.dim = dim
        self.use_diagonal_cov = use_diagonal_cov
        self.time_length = time_length
        
        # Create UPN dynamics networks
        self.dynamics_net = DynamicsNetwork(state_dim=dim, hidden_dim=hidden_dim)
        self.noise_net = ProcessNoiseNetwork(state_dim=dim, hidden_dim=hidden_dim, 
                                           use_diagonal=use_diagonal_cov)
        
        # Base distribution parameters (learnable)
        self.register_parameter('base_mean', 
                               nn.Parameter(torch.zeros(dim)))
        self.register_parameter('base_log_var', 
                               nn.Parameter(torch.zeros(dim)))
    
    def forward(self, t, states):
        """ODE function for the UPN flow dynamics"""
        batch_size = states.shape[0]
        
        # Extract mean and vectorized covariance
        if self.use_diagonal_cov:
            mu = states[:, :self.dim]
            sigma_diag = states[:, self.dim:]
            
            # Compute drift for mean using dynamics network
            mu_drift = self.dynamics_net(t, mu)
            
            # Compute Jacobian of dynamics with respect to state
            J = torch.zeros(batch_size, self.dim, self.dim, device=states.device)
            for i in range(self.dim):
                with torch.enable_grad():
                    mu_i = mu.clone().detach().requires_grad_(True)
                    drift_i = self.dynamics_net(t, mu_i)
                    grad = torch.autograd.grad(drift_i[:, i].sum(), mu_i, create_graph=True)[0]
                    J[:, i, :] = grad
            
            # Compute process noise covariance
            Q = self.noise_net(t, mu)
            Q_diag = torch.diagonal(Q, dim1=1, dim2=2)
            
            # Compute drift for diagonal covariance
            # For diagonal approximation: dΣ_ii/dt ≈ 2*J_ii*Σ_ii + Q_ii
            sigma_drift = 2 * torch.diagonal(J, dim1=1, dim2=2) * sigma_diag + Q_diag
            
            # Compute log-density drift using the instantaneous change formula
            # ∂log p(z(t))/∂t = -tr(J) + 1/2 * tr(Σ^-1 * Q)
            log_density_drift = -torch.sum(torch.diagonal(J, dim1=1, dim2=2), dim=1) 
            log_density_drift += 0.5 * torch.sum(Q_diag / (sigma_diag + 1e-6), dim=1)
            
            # Combine mean and covariance drifts
            return torch.cat([mu_drift, sigma_drift], dim=1), log_density_drift
        else:
            # For full covariance implementation
            raise NotImplementedError("Full covariance not yet implemented for flows")

    def integrate(self, z0, t, return_log_density=False):
        """Integrate the UPN flow dynamics from initial state z0 over time points t"""
        from torchdiffeq import odeint
        
        # Initialize log-density accumulator
        t_size = t.size(0)
        batch_size = z0.size(0)
        log_density = torch.zeros(batch_size, device=z0.device)
        
        def ode_func(t, states):
            drift, log_density_drift = self.forward(t, states)
            return drift
        
        # Use a more stable solver with stricter tolerances
        z_all = odeint(ode_func, z0, t, method='dopri5', 
                    rtol=1e-5, atol=1e-5,  # Tighter tolerances
                    options={'max_num_steps': 10000})  # Allow more steps
        
        # If we need the log density, we integrate the log density drift
        if return_log_density:
            # We need to numerically integrate the log density drift
            for i in range(1, t_size):
                dt = t[i] - t[i-1]
                # Get states at time t[i-1]
                states_i = z_all[i-1]
                # Compute log density drift at time t[i-1]
                _, log_density_drift_i = self.forward(t[i-1], states_i)
                # Add numerical stability clip
                log_density_drift_i = torch.clamp(log_density_drift_i, min=-100, max=100)
                # Update log density (Euler integration)
                log_density += log_density_drift_i * dt
                
        if return_log_density:
            return z_all, log_density
        else:
            return z_all

    def transform_base_to_target(self, z0, return_log_density=False):
        """Transform samples from base distribution to target distribution"""
        # Generate integration time points
        t = torch.linspace(0, self.time_length, 2, device=z0.device)
        
        # Integrate the flow
        if return_log_density:
            z_all, log_density_change = self.integrate(z0, t, return_log_density=True)
            z_final = z_all[-1]
            return z_final, log_density_change
        else:
            z_all = self.integrate(z0, t)
            z_final = z_all[-1]
            return z_final
    
    def transform_target_to_base(self, z1, return_log_density=False):
        """Transform samples from target distribution back to base distribution"""
        # Generate reversed integration time points
        t = torch.linspace(self.time_length, 0, 2, device=z1.device)
        
        # Integrate the flow backward
        if return_log_density:
            z_all, log_density_change = self.integrate(z1, t, return_log_density=True)
            z_final = z_all[-1]
            return z_final, -log_density_change  # Note the negative sign for reverse flow
        else:
            z_all = self.integrate(z1, t)
            z_final = z_all[-1]
            return z_final
    
    def sample(self, n_samples=1000, with_uncertainty=True):
        """Sample from the flow-based model"""
        # Sample from base distribution
        if self.use_diagonal_cov:
            # For diagonal covariance, sample from independent Gaussians
            base_std = torch.exp(0.5 * self.base_log_var)
            eps = torch.randn(n_samples, self.dim, device=self.base_mean.device)
            z0_mean = self.base_mean.unsqueeze(0).expand(n_samples, -1)
            z0_samples = z0_mean + eps * base_std.unsqueeze(0)
            
            # Set initial covariance
            z0_cov = torch.exp(self.base_log_var).unsqueeze(0).expand(n_samples, -1)
            
            # Concatenate mean and covariance for ODE integration
            z0 = torch.cat([z0_samples, z0_cov], dim=1)
            
            # Transform to target distribution
            z_final = self.transform_base_to_target(z0)
            
            # Extract mean and variance
            z_mean = z_final[:, :self.dim]
            z_var = z_final[:, self.dim:]
            
            if with_uncertainty:
                return z_mean, z_var
            else:
                return z_mean
        else:
            # Implementation for full covariance
            raise NotImplementedError("Full covariance sampling not implemented")

    def log_prob(self, x):
        """Compute log probability of samples under the flow model"""
        # Shape check
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        if self.use_diagonal_cov:
            # Initialize with test point as mean and very small covariance
            init_cov = torch.ones(batch_size, self.dim, device=x.device) * 1e-6
            z1 = torch.cat([x, init_cov], dim=1)
            
            try:
                # Transform back to base distribution with more integration steps
                z0, log_density_change = self.transform_target_to_base(z1, return_log_density=True)
                
                # Add numerical stability
                log_density_change = torch.clamp(log_density_change, min=-1e6, max=1e6)
                
                # Extract mean and variance
                z0_mean = z0[:, :self.dim]
                
                # Compute log probability under base distribution
                base_mean = self.base_mean.unsqueeze(0).expand(batch_size, -1)
                base_var = torch.exp(self.base_log_var).unsqueeze(0).expand(batch_size, -1)
                base_var = torch.clamp(base_var, min=1e-6, max=1e6)  # Clamp for stability
                
                # Log prob under base distribution (independent Gaussians)
                log_terms = ((z0_mean - base_mean) ** 2) / base_var + self.base_log_var + np.log(2 * np.pi)
                log_terms = torch.clamp(log_terms, min=-1e6, max=1e6)  # Clamp for stability
                base_log_prob = -0.5 * torch.sum(log_terms, dim=1)
                
                # Add log determinant of Jacobian
                log_prob = base_log_prob + log_density_change
                
                # Clamp for stability
                log_prob = torch.clamp(log_prob, min=-1e6, max=1e6)
                
                return log_prob
            except:
                # Fallback for numerical issues
                return -1e6 * torch.ones(batch_size, device=x.device)
        else:
            # Implementation for full covariance
            raise NotImplementedError("Full covariance log_prob not implemented")          
    
    def compute_loss(self, x):
        """Compute negative log likelihood loss"""
        return -torch.mean(self.log_prob(x))


def generate_flow_data(dataset_type="moons", n_samples=1000, noise=0.1):
    """Generate synthetic 2D datasets for flow experiments"""
    if dataset_type == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise)
    elif dataset_type == "swiss_roll":
        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        # Use only 2 dimensions
        X = X[:, [0, 2]]
    elif dataset_type == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=5, cluster_std=noise)
    elif dataset_type == "circles":
        # Generate concentric circles
        theta = np.random.uniform(0, 2*np.pi, n_samples)
        r1 = np.random.normal(1.0, noise, n_samples)
        r2 = np.random.normal(2.0, noise, n_samples)
        
        # Half samples for inner circle, half for outer
        mask = np.random.choice([True, False], n_samples)
        r = np.where(mask, r1, r2)
        
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Convert to PyTorch tensor
    X = torch.tensor(X, dtype=torch.float32)
    
    # Scale to reasonable range
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    
    return X


def train_upn_flow(flow_data, val_split=0.2, batch_size=128, hidden_dim=64, 
                  num_epochs=300, lr=1e-4, device=None):
    """Train UPN Flow model for density estimation with improved numerical stability"""
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split into train and validation sets
    n_samples = flow_data.shape[0]
    n_val = int(n_samples * val_split)
    
    # Shuffle data
    indices = torch.randperm(n_samples)
    train_indices = indices[:-n_val]
    val_indices = indices[-n_val:]
    
    train_data = flow_data[train_indices]
    val_data = flow_data[val_indices]
    
    # Create data loaders
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Get data dimension
    dim = flow_data.shape[1]
    
    # Create UPN Flow model with shorter time_length for stability
    flow_model = UPNFlow(dim=dim, hidden_dim=hidden_dim, 
                        use_diagonal_cov=True, time_length=0.5).to(device)
    
    # Optimizer with lower learning rate
    optimizer = optim.Adam(flow_model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler with more patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 30  # Early stopping patience
    
    for epoch in range(num_epochs):
        # Training
        flow_model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            
            try:
                # Compute loss (negative log likelihood)
                loss = flow_model.compute_loss(x)
                
                # Check if loss is valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected (value: {loss.item()}), skipping batch")
                    continue
                
                # If loss is extremely large, use log-loss
                if loss > 1e5:
                    loss = torch.log(loss + 1)
                    print(f"Using log-loss due to large value: {loss.item()}")
                
                # Update model
                loss.backward()
                
                # Gradient clipping - more aggressive for stability
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
                
                # Check for exploding gradients
                valid_grads = True
                for param in flow_model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_grads = False
                            break
                
                if valid_grads:
                    optimizer.step()
                    epoch_loss += loss.item() * x.shape[0]
                    valid_batches += 1
                else:
                    print("Warning: NaN or Inf gradients detected, skipping update")
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        # Skip epoch if no valid batches
        if valid_batches == 0:
            print("No valid batches in epoch, skipping to next epoch")
            continue
            
        train_loss = epoch_loss / (valid_batches * batch_size)
        train_losses.append(train_loss)
        
        # Validation
        flow_model.eval()
        val_loss = 0.0
        valid_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                
                try:
                    # Compute loss
                    loss = flow_model.compute_loss(x)
                    
                    # Check if loss is valid
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                        
                    val_loss += loss.item() * x.shape[0]
                    valid_val_batches += 1
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Skip if no valid validation batches
        if valid_val_batches == 0:
            print("No valid validation batches, using previous validation loss")
            if val_losses:
                val_losses.append(val_losses[-1])
            else:
                val_losses.append(float('inf'))
            continue
            
        val_loss /= (valid_val_batches * batch_size)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = flow_model.state_dict().copy()
            print(f"New best model with validation loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping after {epoch+1} epochs due to no improvement")
            break
    
    # Load best model
    if best_model_state is not None:
        flow_model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model found, using final model")
    
    return flow_model, train_losses, val_losses


def visualize_flow_density(flow_model, data, grid_size=100, extent=(-4, 4, -4, 4), 
                          title="Flow Density", save_path=None):
    """Visualize the density learned by the flow model"""
    flow_model.eval()
    
    # Create grid for density evaluation
    x = np.linspace(extent[0], extent[1], grid_size)
    y = np.linspace(extent[2], extent[3], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid points
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(flow_model.base_mean.device)
    
    # Evaluate log probability at each grid point
    log_probs = []
    
    with torch.no_grad():
        # Evaluate in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(grid_tensor), batch_size):
            batch = grid_tensor[i:i+batch_size]
            batch_log_probs = flow_model.log_prob(batch)
            log_probs.append(batch_log_probs)
    
    # Combine batches
    log_probs = torch.cat(log_probs, dim=0)
    
    # Convert to probabilities
    probs = torch.exp(log_probs).cpu().numpy()
    
    # Reshape to grid
    density = probs.reshape(grid_size, grid_size)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot density
    plt.contourf(X, Y, density, cmap='viridis', levels=50)
    plt.colorbar(label='Probability Density')
    
    # Plot data points
    plt.scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), 
                s=10, color='red', alpha=0.5, label='Data')
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def visualize_flow_samples(flow_model, data, n_samples=1000, title="Flow Samples", save_path=None):
    """Visualize samples from the flow model with uncertainty"""
    flow_model.eval()
    
    # Generate samples with uncertainty
    with torch.no_grad():
        z_mean, z_var = flow_model.sample(n_samples=n_samples)
    
    # Convert to numpy
    samples = z_mean.cpu().numpy()
    variances = z_var.cpu().numpy()
    
    # Compute average uncertainty for each sample
    uncertainty = np.mean(variances, axis=1)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot true data points
    plt.scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), 
                s=10, color='red', alpha=0.5, label='True Data')
    
    # Plot samples with uncertainty-based coloring
    scatter = plt.scatter(samples[:, 0], samples[:, 1], 
                c=uncertainty, s=20, cmap='viridis', alpha=0.7, label='Flow Samples')
    
    # Add colorbar for uncertainty
    cbar = plt.colorbar(scatter)
    cbar.set_label('Uncertainty (Avg. Variance)')
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def visualize_flow_transformation(flow_model, n_grid=20, n_time_steps=10, 
                                title="Flow Transformation", save_path=None):
    """Visualize how the flow transforms a grid of points over time"""
    flow_model.eval()
    
    # Create grid of points in base space
    x = np.linspace(-2, 2, n_grid)
    y = np.linspace(-2, 2, n_grid)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid points
    grid_points = np.column_stack([X.flatten(), Y.flatten()])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(flow_model.base_mean.device)
    
    # Add small initial covariance
    init_cov = torch.ones(grid_tensor.shape[0], 2, device=grid_tensor.device) * 1e-4
    z0 = torch.cat([grid_tensor, init_cov], dim=1)
    
    # Create time steps
    time_length = flow_model.time_length
    t = torch.linspace(0, time_length, n_time_steps, device=grid_tensor.device)
    
    # Integrate forward
    with torch.no_grad():
        z_all = flow_model.integrate(z0, t)
    
    # Extract means at each time step
    transformed_grids = []
    for i in range(n_time_steps):
        z_mean = z_all[i, :, :2].cpu().numpy()
        transformed_grids.append(z_mean.reshape(n_grid, n_grid, 2))
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Plot transformation at each time step
    for i in range(n_time_steps):
        ax = axes[i]
        grid_i = transformed_grids[i]
        
        # Plot transformed grid
        for j in range(n_grid):
            ax.plot(grid_i[j, :, 0], grid_i[j, :, 1], 'b-', alpha=0.3)
            ax.plot(grid_i[:, j, 0], grid_i[:, j, 1], 'b-', alpha=0.3)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title(f't = {t[i].item():.2f}')
        ax.grid(alpha=0.3)
        
        # Add axis labels only to left and bottom plots
        if i >= 5:
            ax.set_xlabel('x1')
        if i % 5 == 0:
            ax.set_ylabel('x2')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig