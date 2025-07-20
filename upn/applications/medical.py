import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import norm
from tqdm import tqdm

from upn.core import UPN, DynamicsNetwork, ProcessNoiseNetwork

class TemporalEncoder(nn.Module):
    """Encoder network for irregular time series"""
    def __init__(self, input_dim, hidden_dim=64, latent_dim=8, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # GRU for sequence encoding - note we add 1 to input_dim for time gap
        self.gru = nn.GRU(
            input_size=input_dim + 1,  # +1 for time gap
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output layers for initial state distribution parameters
        self.mean_mapper = nn.Linear(hidden_dim, latent_dim)
        self.logvar_mapper = nn.Linear(hidden_dim, latent_dim)
        
        # Initial covariance output layer
        self.init_cov_mapper = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus()  # Ensures positive values
        )
    
    def forward(self, x, t, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            t: Time points [batch_size, seq_len]
            mask: Mask tensor for valid observations [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute time gaps between observations
        time_gaps = torch.zeros_like(t)
        time_gaps[:, 1:] = t[:, 1:] - t[:, :-1]
        
        # Create time gap feature
        time_gaps = time_gaps.unsqueeze(-1)
        
        # Combine input with time gap feature
        x_with_time = torch.cat([x, time_gaps], dim=-1)
        
        # Apply mask if provided (replace masked values with zeros)
        if mask is not None:
            x_with_time = x_with_time * mask.unsqueeze(-1)
        
        # Pass through GRU
        _, h_n = self.gru(x_with_time)
        
        # Use last layer's hidden state
        h_final = h_n[-1]
        
        # Map to distribution parameters
        mean = self.mean_mapper(h_final)
        logvar = self.logvar_mapper(h_final)
        init_cov_diag = self.init_cov_mapper(h_final)
        
        return mean, logvar, init_cov_diag


class EmissionModel(nn.Module):
    """Emission model for mapping latent state to observations"""
    def __init__(self, latent_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Neural network for emission model
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        # Output layers for mean and logvar
        self.mean_mapper = nn.Linear(hidden_dim, output_dim)
        self.logvar_mapper = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Hardtanh(min_val=-10.0, max_val=2.0)  # Constrain logvar for numerical stability
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent state tensor [batch_size, seq_len, latent_dim]
        """
        h = self.net(z)
        mean = self.mean_mapper(h)
        logvar = self.logvar_mapper(h)
        
        return mean, logvar


class TimeSeriesUPN(nn.Module):
    """UPN model for time series with irregular observations"""
    def __init__(self, input_dim, latent_dim=8, hidden_dim=64, use_diagonal_cov=True):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_diagonal_cov = use_diagonal_cov
        
        # Encoder network (recognition model)
        self.encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        # UPN dynamics model
        dynamics_net = DynamicsNetwork(state_dim=latent_dim, hidden_dim=hidden_dim)
        noise_net = ProcessNoiseNetwork(state_dim=latent_dim, hidden_dim=hidden_dim, use_diagonal=use_diagonal_cov)
        
        self.dynamics = UPN(
            state_dim=latent_dim,
            dynamics_net=dynamics_net,
            noise_net=noise_net,
            use_diagonal_cov=use_diagonal_cov
        )
        
        # Emission model
        self.emission = EmissionModel(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim
        )
    
    def encode(self, x, t, mask=None):
        """Encode observations to get initial state distribution"""
        return self.encoder(x, t, mask)
    
    def sample_latent(self, mean, logvar, init_cov_diag):
        """Sample initial latent state using reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z0 = mean + eps * std
        
        # Create initial covariance matrix (diagonal)
        if self.use_diagonal_cov:
            initial_cov = torch.diag_embed(init_cov_diag)
        else:
            # For full covariance (not implemented yet)
            raise NotImplementedError("Full covariance not implemented yet")
        
        return z0, initial_cov
    
    def decode(self, z):
        """Decode latent state to observations"""
        return self.emission(z)
    
    def forward(self, x, t_obs, t_pred=None, mask=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            t_obs: Observation time points [batch_size, seq_len]
            t_pred: Prediction time points [batch_size, pred_len] (optional)
            mask: Mask tensor for valid observations [batch_size, seq_len]
        """
        # Get batch size
        batch_size = x.shape[0]
        
        # Encode observed sequence
        mean_z0, logvar_z0, init_cov_diag = self.encode(x, t_obs, mask)
        
        # Sample initial latent state
        z0, initial_cov = self.sample_latent(mean_z0, logvar_z0, init_cov_diag)
        
        # If no prediction times are provided, use observation times
        if t_pred is None:
            t_pred = t_obs
        
        # Format integration time points
        # We need to ensure t_span has shape [time_points] for integration
        t_span = t_pred[0]
        
        # Propagate through UPN dynamics
        z_mean, z_cov = self.dynamics.predict(z0, initial_cov, t_span)
        
        # Reshape to [time_points, batch_size, latent_dim]
        # and then to [batch_size, time_points, latent_dim]
        z_mean = z_mean.permute(1, 0, 2)
        z_cov = z_cov.permute(1, 0, 2, 3)
        
        # Generate observations through emission model
        x_mean, x_logvar = self.decode(z_mean)
        
        # Return all relevant outputs
        outputs = {
            'x_mean': x_mean,
            'x_logvar': x_logvar,
            'z_mean': z_mean,
            'z_cov': z_cov,
            'z0_mean': mean_z0,
            'z0_logvar': logvar_z0
        }
        
        return outputs
    
    def compute_loss(self, x, t, mask=None, beta=1.0):
        """
        Compute ELBO loss for time series.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            t: Time points [batch_size, seq_len]
            mask: Mask tensor for valid observations [batch_size, seq_len]
            beta: Weight for KL divergence term (for beta-VAE)
        """
        # Forward pass
        outputs = self.forward(x, t, mask=mask)
        
        # Extract outputs
        x_mean = outputs['x_mean']
        x_logvar = outputs['x_logvar']
        z0_mean = outputs['z0_mean']
        z0_logvar = outputs['z0_logvar']
        
        # Compute reconstruction loss (negative log likelihood)
        # Only consider unmasked elements
        if mask is None:
            mask = torch.ones_like(x[:, :, 0])
        
        # Expand mask to match x dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        
        # Compute MSE for reconstruction (using masked values)
        x_var = torch.exp(x_logvar)
        recon_loss = 0.5 * (((x - x_mean) ** 2) / x_var + x_logvar + np.log(2 * np.pi))
        recon_loss = (recon_loss * mask_expanded).sum() / mask_expanded.sum()
        
        # Compute KL divergence for initial latent state
        # KL(q(z0|x) || p(z0)) where p(z0) is standard normal
        kl_loss = -0.5 * torch.sum(1 + z0_logvar - z0_mean.pow(2) - z0_logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Total loss (ELBO)
        total_loss = recon_loss + beta * kl_loss
        
        # Return individual loss components for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
        
        return loss_dict
    
    def predict(self, x, t_obs, t_pred, mask=None, n_samples=10):
        """
        Make predictions with uncertainty at specified time points.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            t_obs: Observation time points [batch_size, seq_len]
            t_pred: Prediction time points [batch_size, pred_len]
            mask: Mask tensor for valid observations [batch_size, seq_len]
            n_samples: Number of samples for prediction
        """
        # Encode observed sequence
        mean_z0, logvar_z0, init_cov_diag = self.encode(x, t_obs, mask)
        
        # Format integration time points (use first batch element's times)
        t_span = t_pred[0]
        
        # Storage for multiple samples
        all_x_mean = []
        all_x_var = []
        
        # Generate multiple samples
        for _ in range(n_samples):
            # Sample initial latent state
            z0, initial_cov = self.sample_latent(mean_z0, logvar_z0, init_cov_diag)
            
            # Propagate through UPN dynamics
            z_mean, z_cov = self.dynamics.predict(z0, initial_cov, t_span)
            
            # Reshape
            z_mean = z_mean.permute(1, 0, 2)
            
            # Generate observations
            x_mean, x_logvar = self.decode(z_mean)
            x_var = torch.exp(x_logvar)
            
            # Store results
            all_x_mean.append(x_mean)
            all_x_var.append(x_var)
        
        # Stack results [n_samples, batch_size, seq_len, input_dim]
        all_x_mean = torch.stack(all_x_mean)
        all_x_var = torch.stack(all_x_var)
        
        # Compute mean and variance across samples
        pred_mean = all_x_mean.mean(dim=0)
        
        # Total variance = average variance + variance of means
        pred_var = all_x_var.mean(dim=0) + ((all_x_mean - pred_mean.unsqueeze(0)) ** 2).mean(dim=0)
        
        return pred_mean, pred_var


class SyntheticTimeSeriesDataset(Dataset):
    """Dataset for synthetic time series data"""
    def __init__(self, values, times, masks, outcomes):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.times = torch.tensor(times, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)
        
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        return self.values[idx], self.times[idx], self.masks[idx], self.outcomes[idx]


def create_synthetic_time_series(num_patients=100, num_features=10, seq_length=48, missing_ratio=0.3):
    """Create synthetic time series data for demonstration"""
    # Initialize arrays
    values = np.zeros((num_patients, seq_length, num_features), dtype=np.float32)
    times = np.zeros((num_patients, seq_length), dtype=np.float32)
    masks = np.zeros((num_patients, seq_length), dtype=np.float32)
    outcomes = np.zeros(num_patients, dtype=np.float32)
    
    # Generate data for each patient
    for i in range(num_patients):
        # Generate time points (irregular)
        patient_times = np.sort(np.random.uniform(0, 48, seq_length))
        times[i] = patient_times
        
        # Generate baseline features (different for each patient)
        baseline = np.random.normal(0, 1, num_features)
        
        # Generate time series with trends and noise
        for j in range(num_features):
            # Add trend and seasonality
            trend = 0.1 * patient_times
            seasonality = 0.5 * np.sin(patient_times / 8) + 0.3 * np.cos(patient_times / 4)
            noise = 0.2 * np.random.normal(0, 1, seq_length)
            
            values[i, :, j] = baseline[j] + trend + seasonality + noise
        
        # Create missing values
        valid_mask = np.random.uniform(0, 1, (seq_length, num_features)) > missing_ratio
        values[i] = values[i] * valid_mask
        
        # Create sequence mask (at least one feature observed at each time)
        seq_mask = (valid_mask.sum(axis=1) > 0).astype(np.float32)
        masks[i] = seq_mask
        
        # Generate outcome based on the average value of features
        avg_value = np.mean(values[i])
        outcomes[i] = 1 if avg_value > 0 else 0
    
    return {
        'values': values,
        'times': times,
        'masks': masks,
        'outcomes': outcomes
    }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    
    # Unpack batch
    values, timestamps, masks, outcomes = zip(*batch)
    
    # Get batch size and max sequence length
    batch_size = len(values)
    max_len = max(v.shape[0] for v in values)
    feat_dim = values[0].shape[1]
    
    # Initialize tensors
    padded_values = torch.zeros(batch_size, max_len, feat_dim)
    padded_times = torch.zeros(batch_size, max_len)
    padded_masks = torch.zeros(batch_size, max_len)
    
    # Fill tensors with data
    for i, (v, t, m) in enumerate(zip(values, timestamps, masks)):
        seq_len = v.shape[0]
        padded_values[i, :seq_len] = v
        padded_times[i, :seq_len] = t
        
        # Create sequence mask (1 for valid positions, 0 for padding)
        padded_masks[i, :seq_len] = m
    
    # Stack outcomes
    outcomes = torch.stack(outcomes)
    
    return padded_values, padded_times, padded_masks, outcomes


def evaluate_time_series_upn(model, test_loader, device='cuda'):
    """Evaluate TimeSeriesUPN model on test data"""
    model.eval()
    
    # Initialize metrics
    test_loss = 0.0
    test_recon_loss = 0.0
    test_kl_loss = 0.0
    num_batches = 0
    
    # Storage for visualization
    all_inputs = []
    all_times = []
    all_masks = []
    all_predictions = []
    all_pred_vars = []
    
    with torch.no_grad():
        for values, times, masks, _ in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            values = values.to(device)
            times = times.to(device)
            masks = masks.to(device)
            
            # Compute loss
            loss_dict = model.compute_loss(values, times, masks)
            
            # Accumulate losses
            test_loss += loss_dict['total_loss'].item()
            test_recon_loss += loss_dict['recon_loss'].item()
            test_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
            
            # Make predictions
            pred_mean, pred_var = model.predict(values, times, times, masks, n_samples=10)
            
            # Store for visualization (first batch only)
            if len(all_inputs) < 1:
                all_inputs.append(values.cpu().numpy())
                all_times.append(times.cpu().numpy())
                all_masks.append(masks.cpu().numpy())
                all_predictions.append(pred_mean.cpu().numpy())
                all_pred_vars.append(pred_var.cpu().numpy())
    
    # Average losses
    test_loss /= num_batches
    test_recon_loss /= num_batches
    test_kl_loss /= num_batches
    
    # Print results
    print(f"Test Loss: {test_loss:.6f} (Recon: {test_recon_loss:.6f}, KL: {test_kl_loss:.6f})")
    
    # Prepare evaluation results
    eval_results = {
        'test_loss': test_loss,
        'recon_loss': test_recon_loss,
        'kl_loss': test_kl_loss,
        'inputs': all_inputs,
        'times': all_times,
        'masks': all_masks,
        'predictions': all_predictions,
        'pred_vars': all_pred_vars
    }
    
    return eval_results


def visualize_time_series_upn(eval_results, feature_names, num_patients=3, num_features=5, save_path=None):
    """Visualize TimeSeriesUPN predictions"""
    # Extract data
    inputs = eval_results['inputs'][0]  # [batch_size, seq_len, input_dim]
    times = eval_results['times'][0]    # [batch_size, seq_len]
    masks = eval_results['masks'][0]    # [batch_size, seq_len]
    predictions = eval_results['predictions'][0]  # [batch_size, seq_len, input_dim]
    pred_vars = eval_results['pred_vars'][0]  # [batch_size, seq_len, input_dim]
    
    # Limit to first num_patients patients
    inputs = inputs[:num_patients]
    times = times[:num_patients]
    masks = masks[:num_patients]
    predictions = predictions[:num_patients]
    pred_vars = pred_vars[:num_patients]
    
    # Select features with most observations
    feature_counts = np.sum(masks[:, :, np.newaxis] * np.ones_like(inputs), axis=(0, 1))
    top_features = np.argsort(feature_counts)[-num_features:]
    
    # Create figure
    num_rows = num_patients
    num_cols = min(num_features, len(feature_names))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_patients))
    
    # Handle single patient or feature case
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each patient and feature
    for i in range(num_rows):
        for j, feat_idx in enumerate(top_features):
            if j >= num_cols:
                continue
                
            ax = axes[i, j]
            
            # Get data for this patient and feature
            patient_times = times[i]
            patient_values = inputs[i, :, feat_idx]
            patient_mask = masks[i]
            patient_pred = predictions[i, :, feat_idx]
            patient_var = pred_vars[i, :, feat_idx]
            
            # Only use valid time points (non-padding)
            valid_idx = patient_mask > 0
            
            if valid_idx.sum() == 0:
                ax.text(0.5, 0.5, "No valid data", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes)
                continue
                
            # Get valid times and values
            t = patient_times[valid_idx]
            v = patient_values[valid_idx]
            v_mask = (inputs[i, valid_idx, feat_idx] != 0)  # Actual observations (not masked)
            
            # Plot ground truth (observed points)
            ax.scatter(t[v_mask], v[v_mask], c='blue', s=30, label='Observed')
            
            # Plot predictions
            pred = patient_pred[valid_idx]
            var = patient_var[valid_idx]
            std = np.sqrt(var)
            
            # Plot prediction line
            ax.plot(t, pred, 'r-', label='Prediction')
            
            # Plot uncertainty bounds (95% confidence interval)
            ax.fill_between(t, pred - 1.96 * std, pred + 1.96 * std, color='red', alpha=0.2, label='95% CI')
            
            # Set labels and title
            feat_name = feature_names[feat_idx] if j < len(feature_names) else f"Feature {feat_idx}"
            ax.set_title(f"Patient {i+1}: {feat_name}")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            
            # Add legend to first plot only
            if i == 0 and j == 0:
                ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def evaluate_forecasting(model, test_loader, forecast_horizon=12, device='cuda'):
    """Evaluate model on forecasting task with missing data"""
    model.eval()
    
    # Initialize metrics
    mse_total = 0.0
    nll_total = 0.0
    coverage_total = 0.0
    num_forecasts = 0
    
    with torch.no_grad():
        for values, times, masks, _ in tqdm(test_loader, desc="Evaluating forecasting"):
            batch_size = values.shape[0]
            seq_len = values.shape[1]
            
            # Skip short sequences
            if seq_len <= forecast_horizon + 5:
                continue
            
            # Define observation window and forecast window
            obs_end = seq_len - forecast_horizon
            
            # Split data into observation and forecast windows
            obs_values = values[:, :obs_end, :].to(device)
            obs_times = times[:, :obs_end].to(device)
            obs_masks = masks[:, :obs_end].to(device)
            
            # Forecast window
            forecast_values = values[:, obs_end:, :].to(device)
            forecast_times = times[:, obs_end:].to(device)
            forecast_masks = masks[:, obs_end:].to(device)
            
            # Make predictions on forecast window
            full_times = torch.cat([obs_times, forecast_times], dim=1)
            
            # Get model predictions
            pred_mean, pred_var = model.predict(
                obs_values, obs_times, full_times, obs_masks, n_samples=10
            )
            
            # Extract predictions for forecast window
            pred_mean_forecast = pred_mean[:, obs_end:, :]
            pred_var_forecast = pred_var[:, obs_end:, :]
            
            # Compute metrics only on valid forecast points
            valid_points = forecast_masks.sum().item()
            if valid_points == 0:
                continue
                
            # Compute MSE
            squared_error = ((pred_mean_forecast - forecast_values) ** 2) * forecast_masks.unsqueeze(-1)
            mse = squared_error.sum().item() / valid_points
            mse_total += mse * batch_size
            
            # Compute negative log likelihood
            std_forecast = torch.sqrt(pred_var_forecast)
            z_score = (forecast_values - pred_mean_forecast) / (std_forecast + 1e-8)
            nll = (0.5 * z_score ** 2 + torch.log(std_forecast + 1e-8) + 0.5 * np.log(2 * np.pi))
            nll = (nll * forecast_masks.unsqueeze(-1)).sum().item() / valid_points
            nll_total += nll * batch_size
            
            # Compute 95% coverage
            lower_bound = pred_mean_forecast - 1.96 * std_forecast
            upper_bound = pred_mean_forecast + 1.96 * std_forecast
            in_interval = ((forecast_values >= lower_bound) & (forecast_values <= upper_bound))
            in_interval = (in_interval * forecast_masks.unsqueeze(-1)).sum().item()
            coverage = in_interval / valid_points
            coverage_total += coverage * batch_size
            
            num_forecasts += batch_size
    
    # Compute average metrics
    mse_avg = mse_total / num_forecasts
    nll_avg = nll_total / num_forecasts
    coverage_avg = coverage_total / num_forecasts
    
    # Print results
    print(f"Forecasting Evaluation (horizon = {forecast_horizon} hours):")
    print(f"  MSE: {mse_avg:.6f}")
    print(f"  NLL: {nll_avg:.6f}")
    print(f"  95% Coverage: {coverage_avg:.4f} (ideal: 0.95)")
    
    return {
        'mse': mse_avg,
        'nll': nll_avg,
        'coverage': coverage_avg
    }


def visualize_forecasting(model, test_loader, forecast_horizon=12, num_patients=3, num_features=3, 
                          feature_names=None, save_path=None, device='cuda'):
    """Visualize forecasting results"""
    model.eval()
    
    # Storage for visualization
    all_obs_values = []
    all_obs_times = []
    all_forecast_values = []
    all_forecast_times = []
    all_pred_means = []
    all_pred_stds = []
    
    with torch.no_grad():
        for values, times, masks, _ in test_loader:
            batch_size = values.shape[0]
            seq_len = values.shape[1]
            
            # Skip short sequences
            if seq_len <= forecast_horizon + 5:
                continue
            
            # Define observation window and forecast window
            obs_end = seq_len - forecast_horizon
            
            # Split data into observation and forecast windows
            obs_values = values[:, :obs_end, :].to(device)
            obs_times = times[:, :obs_end].to(device)
            obs_masks = masks[:, :obs_end].to(device)
            
            # Forecast window
            forecast_values = values[:, obs_end:, :].to(device)
            forecast_times = times[:, obs_end:].to(device)
            
            # Make predictions on forecast window
            full_times = torch.cat([obs_times, forecast_times], dim=1)
            
            # Get model predictions
            pred_mean, pred_var = model.predict(
                obs_values, obs_times, full_times, obs_masks, n_samples=10
            )
            
            # Extract predictions for forecast window
            pred_mean_forecast = pred_mean[:, obs_end:, :]
            pred_std_forecast = torch.sqrt(pred_var[:, obs_end:, :])
            
            # Store for visualization
            all_obs_values.append(obs_values.cpu().numpy())
            all_obs_times.append(obs_times.cpu().numpy())
            all_forecast_values.append(forecast_values.cpu().numpy())
            all_forecast_times.append(forecast_times.cpu().numpy())
            all_pred_means.append(pred_mean_forecast.cpu().numpy())
            all_pred_stds.append(pred_std_forecast.cpu().numpy())
            
            # Only need a few examples
            if len(all_obs_values) >= num_patients:
                break
    
    # If no data was collected, return
    if len(all_obs_values) == 0:
        print("No suitable data found for visualization")
        return
    
    # Create figure
    num_rows = min(num_patients, len(all_obs_values))
    num_cols = num_features
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    
    # Handle single row/column case
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # For each patient
    for i in range(num_rows):
        # Get data for this patient
        obs_values = all_obs_values[i][0]  # First batch element
        obs_times = all_obs_times[i][0]
        forecast_values = all_forecast_values[i][0]
        forecast_times = all_forecast_times[i][0]
        pred_means = all_pred_means[i][0]
        pred_stds = all_pred_stds[i][0]
        
        # Get full time range
        full_times = np.concatenate([obs_times, forecast_times])
        
        # Find features with most observations
        feature_mask = (obs_values != 0).sum(axis=0) > 0
        if feature_mask.sum() > 0:
            feature_indices = np.where(feature_mask)[0][:num_features]
        else:
            # If no features have observations, use the first few
            feature_indices = np.arange(min(num_features, obs_values.shape[1]))
        
        # Plot each feature
        for j, feat_idx in enumerate(feature_indices):
            if j >= num_cols:
                continue
                
            ax = axes[i, j]
            
            # Get data for this feature
            obs_feat = obs_values[:, feat_idx]
            forecast_feat = forecast_values[:, feat_idx]
            pred_mean_feat = pred_means[:, feat_idx]
            pred_std_feat = pred_stds[:, feat_idx]
            
            # Mask for observed points
            obs_mask = obs_feat != 0
            forecast_mask = forecast_feat != 0
            
            # Plot observation window
            ax.scatter(obs_times[obs_mask], obs_feat[obs_mask], 
                       c='blue', s=30, label='Observed')
            
            # Plot ground truth in forecast window
            ax.scatter(forecast_times[forecast_mask], forecast_feat[forecast_mask], 
                       c='green', s=30, label='Future (Ground Truth)')
            
            # Plot predictions
            ax.plot(forecast_times, pred_mean_feat, 'r-', label='Prediction')
            
            # Plot uncertainty bounds
            ax.fill_between(forecast_times, 
                           pred_mean_feat - 1.96 * pred_std_feat,
                           pred_mean_feat + 1.96 * pred_std_feat,
                           color='red', alpha=0.2, label='95% CI')
            
            # Add vertical line to separate observation and forecast windows
            ax.axvline(x=obs_times[-1], color='black', linestyle='--', alpha=0.5,
                      label='Forecast Start')
            
            # Set labels and title
            if feature_names is not None and feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
            else:
                feat_name = f"Feature {feat_idx}"
                
            ax.set_title(f"Patient {i+1}: {feat_name}")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            
            # Add legend to first plot only
            if i == 0 and j == 0:
                ax.legend()
    
    plt.suptitle(f"Forecasting Results (Horizon: {forecast_horizon} hours)", fontsize=16)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def evaluate_missing_data(model, test_loader, missing_ratio=0.5, device='cuda'):
    """Evaluate model on missing data imputation task"""
    model.eval()
    
    # Initialize metrics
    mse_total = 0.0
    nll_total = 0.0
    coverage_total = 0.0
    num_imputations = 0
    
    with torch.no_grad():
        for values, times, masks, _ in tqdm(test_loader, desc="Evaluating imputation"):
            batch_size = values.shape[0]
            
            # Create artificial missingness
            # We'll keep the original data as ground truth and artificially mask some values
            original_values = values.clone()
            original_masks = masks.clone()
            
            # Only create artificial missingness for observed values
            observed = (original_masks.unsqueeze(-1) > 0) & (values != 0)
            
            # Create random mask for missingness
            random_mask = torch.rand(values.shape) > missing_ratio
            random_mask = random_mask.to(device)
            
            # Apply missingness to values and masks
            artificial_masks = original_masks.clone().unsqueeze(-1) * random_mask * observed
            artificial_values = original_values.clone() * artificial_masks
            
            # Flatten masks back to sequence mask
            artificial_seq_masks = (artificial_masks.sum(dim=-1) > 0).float()
            
            # Move to device
            artificial_values = artificial_values.to(device)
            artificial_seq_masks = artificial_seq_masks.to(device)
            original_values = original_values.to(device)
            times = times.to(device)
            
            # Make predictions with artificially masked data
            pred_mean, pred_var = model.predict(
                artificial_values, times, times, artificial_seq_masks, n_samples=10
            )
            
            # Compute metrics only on artificially masked points
            impute_mask = (original_masks.unsqueeze(-1) > 0) & (artificial_masks == 0) & observed
            impute_mask = impute_mask.to(device)
            
            valid_points = impute_mask.sum().item()
            if valid_points == 0:
                continue
                
            # Compute MSE
            squared_error = ((pred_mean - original_values) ** 2) * impute_mask
            mse = squared_error.sum().item() / valid_points
            mse_total += mse * batch_size
            
            # Compute negative log likelihood
            pred_std = torch.sqrt(pred_var)
            z_score = (original_values - pred_mean) / (pred_std + 1e-8)
            nll = (0.5 * z_score ** 2 + torch.log(pred_std + 1e-8) + 0.5 * np.log(2 * np.pi))
            nll = (nll * impute_mask).sum().item() / valid_points
            nll_total += nll * batch_size
            
            # Compute 95% coverage
            lower_bound = pred_mean - 1.96 * pred_std
            upper_bound = pred_mean + 1.96 * pred_std
            in_interval = ((original_values >= lower_bound) & (original_values <= upper_bound))
            in_interval = (in_interval * impute_mask).sum().item()
            coverage = in_interval / valid_points
            coverage_total += coverage * batch_size
            
            num_imputations += batch_size
    
    # Compute average metrics
    mse_avg = mse_total / num_imputations
    nll_avg = nll_total / num_imputations
    coverage_avg = coverage_total / num_imputations
    
    # Print results
    print(f"Missing Data Imputation Evaluation (missing ratio = {missing_ratio:.2f}):")
    print(f"  MSE: {mse_avg:.6f}")
    print(f"  NLL: {nll_avg:.6f}")
    print(f"  95% Coverage: {coverage_avg:.4f} (ideal: 0.95)")
    
    return {
        'mse': mse_avg,
        'nll': nll_avg,
        'coverage': coverage_avg
    }


def visualize_missing_data(model, test_loader, missing_ratio=0.5, num_patients=3, num_features=3,
                          feature_names=None, save_path=None, device='cuda'):
    """Visualize missing data imputation results"""
    model.eval()
    
    # Storage for visualization
    all_original_values = []
    all_masked_values = []
    all_times = []
    all_pred_means = []
    all_pred_stds = []
    all_impute_masks = []
    
    with torch.no_grad():
        for values, times, masks, _ in test_loader:
            batch_size = values.shape[0]
            
            # Create artificial missingness
            # We'll keep the original data as ground truth and artificially mask some values
            original_values = values.clone()
            original_masks = masks.clone()
            
            # Only create artificial missingness for observed values
            observed = (original_masks.unsqueeze(-1) > 0) & (values != 0)
            
            # Create random mask for missingness
            random_mask = torch.rand(values.shape) > missing_ratio
            
            # Apply missingness to values and masks
            artificial_masks = original_masks.clone().unsqueeze(-1) * random_mask * observed
            artificial_values = original_values.clone() * artificial_masks
            
            # Flatten masks back to sequence mask
            artificial_seq_masks = (artificial_masks.sum(dim=-1) > 0).float()
            
            # Move to device
            artificial_values = artificial_values.to(device)
            artificial_seq_masks = artificial_seq_masks.to(device)
            original_values = original_values.to(device)
            times = times.to(device)
            
            # Make predictions with artificially masked data
            pred_mean, pred_var = model.predict(
                artificial_values, times, times, artificial_seq_masks, n_samples=10
            )
            
            # Compute imputation mask (points that were artificially masked)
            impute_mask = (original_masks.unsqueeze(-1) > 0) & (artificial_masks == 0) & observed
            
            # Store for visualization
            all_original_values.append(original_values.cpu().numpy())
            all_masked_values.append(artificial_values.cpu().numpy())
            all_times.append(times.cpu().numpy())
            all_pred_means.append(pred_mean.cpu().numpy())
            all_pred_stds.append(torch.sqrt(pred_var).cpu().numpy())
            all_impute_masks.append(impute_mask.cpu().numpy())
            
            # Only need a few examples
            if len(all_original_values) >= num_patients:
                break
    
    # If no data was collected, return
    if len(all_original_values) == 0:
        print("No suitable data found for visualization")
        return
    
    # Create figure
    num_rows = min(num_patients, len(all_original_values))
    num_cols = num_features
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    
    # Handle single row/column case
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # For each patient
    for i in range(num_rows):
        # Get data for this patient
        original_values = all_original_values[i][0]  # First batch element
        masked_values = all_masked_values[i][0]
        times = all_times[i][0]
        pred_means = all_pred_means[i][0]
        pred_stds = all_pred_stds[i][0]
        impute_mask = all_impute_masks[i][0]
        
        # Find features with most observations
        feature_mask = (original_values != 0).sum(axis=0) > 0
        if feature_mask.sum() > 0:
            feature_indices = np.where(feature_mask)[0][:num_features]
        else:
            # If no features have observations, use the first few
            feature_indices = np.arange(min(num_features, original_values.shape[1]))
        
        # Plot each feature
        for j, feat_idx in enumerate(feature_indices):
            if j >= num_cols:
                continue
                
            ax = axes[i, j]
            
            # Get data for this feature
            original_feat = original_values[:, feat_idx]
            masked_feat = masked_values[:, feat_idx]
            pred_mean_feat = pred_means[:, feat_idx]
            pred_std_feat = pred_stds[:, feat_idx]
            impute_mask_feat = impute_mask[:, feat_idx]
            
            # Masks for different point types
            observed_mask = (masked_feat != 0)  # Points that remained after masking
            imputed_mask = impute_mask_feat  # Points that were artificially masked
            
            # Plot observations (points that weren't masked)
            ax.scatter(times[observed_mask], original_feat[observed_mask], 
                       c='blue', s=30, label='Observed')
            
            # Plot ground truth for imputed points
            ax.scatter(times[imputed_mask], original_feat[imputed_mask], 
                       c='green', s=30, label='Masked (Ground Truth)')
            
            # Plot imputations
            ax.scatter(times[imputed_mask], pred_mean_feat[imputed_mask], 
                       c='red', s=30, marker='x', label='Imputed')
            
            # Plot uncertainty bounds for imputed points
            for t_idx in np.where(imputed_mask)[0]:
                t = times[t_idx]
                mean = pred_mean_feat[t_idx]
                std = pred_std_feat[t_idx]
                
                # Plot error bars
                ax.errorbar(t, mean, yerr=1.96 * std, color='red', capsize=5, alpha=0.7)
            
            # Set labels and title
            if feature_names is not None and feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
            else:
                feat_name = f"Feature {feat_idx}"
                
            ax.set_title(f"Patient {i+1}: {feat_name}")
            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            
            # Add legend to first plot only
            if i == 0 and j == 0:
                ax.legend()
    
    plt.suptitle(f"Missing Data Imputation (Missing Ratio: {missing_ratio:.2f})", fontsize=16)
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def calculate_calibration_metrics(y_true, y_mean, y_std, num_bins=10):
    """
    Calculate calibration metrics for uncertainty estimates.
    
    Args:
        y_true: Ground truth values [batch_size, seq_len, feature_dim]
        y_mean: Predicted means [batch_size, seq_len, feature_dim]
        y_std: Predicted standard deviations [batch_size, seq_len, feature_dim]
        num_bins: Number of bins for calibration curve
        
    Returns:
        dict with calibration metrics
    """
    # Compute z-scores (normalized prediction errors)
    z = (y_true - y_mean) / (y_std + 1e-8)
    
    # Flatten arrays for easier processing
    z_flat = z.reshape(-1)
    
    # Remove NaN or infinite values
    valid_mask = np.isfinite(z_flat)
    z_valid = z_flat[valid_mask]
    
    if len(z_valid) == 0:
        return {
            'ece': float('nan'),
            'mce': float('nan'),
            'rmsce': float('nan'),
            'coverage_error': float('nan')
        }
    
    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(-3, 3, num_bins + 1)
    bin_indices = np.digitize(z_valid, bin_boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    expected_proportions = np.zeros(num_bins)
    observed_proportions = np.zeros(num_bins)
    
    for i in range(num_bins):
        # Calculate theoretical proportion for this bin (assuming normal distribution)
        lower_bound = bin_boundaries[i]
        upper_bound = bin_boundaries[i + 1]
        expected_proportions[i] = (norm.cdf(upper_bound) - norm.cdf(lower_bound))
        
        # Calculate observed proportion
        bin_mask = (bin_indices == i)
        if np.sum(bin_mask) > 0:
            observed_proportions[i] = np.sum(bin_mask) / len(z_valid)
    
    # ECE (Expected Calibration Error)
    ece = np.mean(np.abs(observed_proportions - expected_proportions))
    
    # MCE (Maximum Calibration Error)
    mce = np.max(np.abs(observed_proportions - expected_proportions))
    
    # RMSCE (Root Mean Squared Calibration Error)
    rmsce = np.sqrt(np.mean((observed_proportions - expected_proportions) ** 2))
    
    # Coverage error at 95% interval
    z_abs = np.abs(z_valid)
    coverage_95 = np.mean(z_abs <= 1.96)
    coverage_error = np.abs(coverage_95 - 0.95)
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'rmsce': float(rmsce),
        'coverage_error': float(coverage_error),
        'expected_proportions': expected_proportions,
        'observed_proportions': observed_proportions,
        'bin_boundaries': bin_boundaries
    }


def visualize_calibration_curve(cal_metrics, save_path=None):
    """
    Visualize calibration curve from calibration metrics.
    
    Args:
        cal_metrics: Dictionary with calibration metrics
        save_path: Path to save the figure
    """
    expected = cal_metrics['expected_proportions']
    observed = cal_metrics['observed_proportions']
    bin_boundaries = cal_metrics['bin_boundaries']
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    plt.figure(figsize=(10, 8))
    
    # Plot calibration curve
    plt.subplot(2, 1, 1)
    plt.plot(bin_centers, observed, 'bo-', linewidth=2, label='Observed')
    plt.plot(bin_centers, expected, 'r--', linewidth=2, label='Expected (Normal)')
    plt.plot([-3, 3], [-3, 3], 'k:', label='Perfect Calibration')
    plt.xlabel('Expected Proportion')
    plt.ylabel('Observed Proportion')
    plt.title('Calibration Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot error
    plt.subplot(2, 1, 2)
    plt.bar(bin_centers, np.abs(observed - expected), width=0.5, alpha=0.6)
    plt.xlabel('Predicted Probability Bin')
    plt.ylabel('Absolute Calibration Error')
    plt.title(f'Calibration Error (ECE: {cal_metrics["ece"]:.4f}, MCE: {cal_metrics["mce"]:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()