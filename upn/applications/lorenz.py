import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.integrate import solve_ivp
from scipy.stats import norm
from tqdm import tqdm

from ..applications.dynamical import (
    DynamicalSystemTrainer, 
    DynamicalSystemDataset,
    UPNDynamicalSystem,
    DeterministicODENet,
    EnsembleODENet
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0, noise_scale=0.0):
    """Lorenz attractor system"""
    x, y, z = state
    
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    
    if noise_scale > 0:
        dx_dt += noise_scale * np.random.randn()
        dy_dt += noise_scale * np.random.randn()
        dz_dt += noise_scale * np.random.randn()
        
    return [dx_dt, dy_dt, dz_dt]

def generate_lorenz_data(num_trajectories=100, t_max=15.0, dt=0.01, noise_scale=0.1):
    """Generate data from the Lorenz system"""
    t_eval = np.arange(0, t_max, dt)
    num_steps = len(t_eval)
    
    # Initialize storage for trajectories
    all_trajectories = np.zeros((num_trajectories, num_steps, 3))
    
    for i in range(num_trajectories):
        # Random initial state near the attractor
        initial_state = np.random.uniform(-15, 15, 3)
        
        # Solve the ODE
        sol = solve_ivp(
            lambda t, y: lorenz_system(t, y, noise_scale=0),
            [0, t_max],
            initial_state,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,  # Tighter tolerances for chaotic system
            atol=1e-9
        )
        
        # Store the trajectory
        trajectory = sol.y.T  # [num_steps, 3]
        
        # Add observation noise
        trajectory = trajectory + np.random.randn(*trajectory.shape) * noise_scale
        
        all_trajectories[i] = trajectory
        
    return all_trajectories, t_eval

def prepare_lorenz_dataloaders(trajectories, time_points, batch_size=16, 
                              history_length=20, future_length=50,
                              train_split=0.7, val_split=0.15):
    """Prepare DataLoaders for Lorenz system data"""
    # Split into train, validation, and test sets
    train_size = int(train_split * len(trajectories))
    val_size = int(val_split * len(trajectories))
    
    train_data = trajectories[:train_size]
    val_data = trajectories[train_size:train_size+val_size]
    test_data = trajectories[train_size+val_size:]
    
    print(f"Train set: {len(train_data)} trajectories")
    print(f"Validation set: {len(val_data)} trajectories")
    print(f"Test set: {len(test_data)} trajectories")
    
    # Create datasets
    train_dataset = DynamicalSystemDataset(train_data, time_points, history_length, future_length)
    val_dataset = DynamicalSystemDataset(val_data, time_points, history_length, future_length)
    test_dataset = DynamicalSystemDataset(test_data, time_points, history_length, future_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

#######################
# Evaluation Functions
#######################

def calculate_nll(y_true, y_mean, y_cov):
    """Calculate negative log likelihood for multivariate normal."""
    nll = 0.0
    batch_size, time_steps, dim = y_true.shape
    
    for b in range(batch_size):
        for t in range(time_steps):
            try:
                dist = MultivariateNormal(y_mean[b, t], y_cov[b, t])
                nll -= dist.log_prob(y_true[b, t]).item()
            except:
                # Fallback to squared error if covariance is not positive definite
                nll += 0.5 * ((y_true[b, t] - y_mean[b, t]) ** 2).sum().item()
                nll += dim * np.log(2 * np.pi) / 2  # Add constant term
    
    return nll / (batch_size * time_steps)

def calculate_crps(y_true, y_mean, y_std):
    """Calculate the Continuous Ranked Probability Score (CRPS) for univariate normal."""
    # For multivariate, we average the CRPS over dimensions
    crps = 0.0
    batch_size, time_steps, dim = y_true.shape
    
    for b in range(batch_size):
        for t in range(time_steps):
            for d in range(dim):
                # Standard normal CDF and PDF
                z = (y_true[b, t, d] - y_mean[b, t, d]) / y_std[b, t, d]
                cdf_z = norm.cdf(z)
                pdf_z = norm.pdf(z)
                
                # CRPS formula for Gaussian distribution
                crps += y_std[b, t, d] * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1/np.sqrt(np.pi))
    
    return crps / (batch_size * time_steps * dim)

def calculate_interval_coverage(y_true, y_mean, y_std, confidence=0.95):
    """Calculate the coverage of prediction intervals."""
    z_score = norm.ppf((1 + confidence) / 2)
    
    batch_size, time_steps, dim = y_true.shape
    total_points = batch_size * time_steps * dim
    in_interval = 0
    
    for b in range(batch_size):
        for t in range(time_steps):
            for d in range(dim):
                lower = y_mean[b, t, d] - z_score * y_std[b, t, d]
                upper = y_mean[b, t, d] + z_score * y_std[b, t, d]
                
                if lower <= y_true[b, t, d] <= upper:
                    in_interval += 1
    
    return in_interval / total_points

def evaluate_models(upn_model, det_model, ensemble_model, test_loader, system_name="Lorenz Attractor"):
    """Evaluate models on test data and compute metrics"""
    print(f"Starting evaluation for {system_name}...")
    
    # Initialize metrics
    metrics = {
        'upn_mse': 0.0,
        'det_mse': 0.0,
        'ensemble_mse': 0.0,
        'upn_nll': 0.0,
        'ensemble_nll': 0.0,
        'upn_crps': 0.0,
        'ensemble_crps': 0.0,
        'upn_coverage': 0.0,
        'ensemble_coverage': 0.0
    }
    
    # Storage for visualization
    all_history = []
    all_future = []
    all_upn_mean = []
    all_upn_std = []
    all_det_pred = []
    all_ensemble_mean = []
    all_ensemble_std = []
    all_time_points = []
    
    # Process test data
    batch_count = 0
    total_samples = 0
    
    with torch.no_grad():
        for history, history_time, future, future_time in tqdm(test_loader, desc="Evaluating"):
            batch_count += 1
            
            history = history.to(device)
            history_time = history_time.to(device)
            future = future.to(device)
            future_time = future_time.to(device)
            
            batch_size = history.shape[0]
            total_samples += batch_size
            
            # Get initial state for models
            initial_state = history[:, -1]
            
            # Initial covariance for UPN
            state_dim = initial_state.shape[1]
            initial_cov = torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1) * 1e-4
            
            # Get the last time point from history
            last_time = history_time[:, -1].unsqueeze(1)
            
            # Create time span for ODE integration
            t_span = torch.cat([last_time[0], future_time[0]], dim=0)
            
            # UPN predictions
            upn_mean, upn_cov = upn_model.predict(initial_state, initial_cov, t_span)
            upn_mean = upn_mean[1:]  # Remove first point
            upn_cov = upn_cov[1:]
            
            # Transpose to match future shape
            upn_mean = upn_mean.permute(1, 0, 2)
            upn_cov = upn_cov.permute(1, 0, 2, 3)
            
            # Get standard deviation from covariance
            upn_std = torch.sqrt(torch.diagonal(upn_cov, dim1=2, dim2=3))
            
            # Deterministic predictions
            det_pred = det_model.predict(initial_state, t_span)
            det_pred = det_pred[1:]  # Remove first point
            det_pred = det_pred.permute(1, 0, 2)
            
            # Ensemble predictions
            ensemble_mean, ensemble_cov = ensemble_model.predict(initial_state, t_span)
            ensemble_mean = ensemble_mean[1:]  # Remove first point
            ensemble_cov = ensemble_cov[1:]
            ensemble_mean = ensemble_mean.permute(1, 0, 2)
            ensemble_cov = ensemble_cov.permute(1, 0, 2, 3)
            
            # Get standard deviation from covariance
            ensemble_std = torch.sqrt(torch.diagonal(ensemble_cov, dim1=2, dim2=3))
            
            # Calculate MSE
            upn_batch_mse = ((upn_mean - future) ** 2).mean().item()
            det_batch_mse = ((det_pred - future) ** 2).mean().item()
            ensemble_batch_mse = ((ensemble_mean - future) ** 2).mean().item()
            
            metrics['upn_mse'] += upn_batch_mse * batch_size
            metrics['det_mse'] += det_batch_mse * batch_size
            metrics['ensemble_mse'] += ensemble_batch_mse * batch_size
            
            # Calculate negative log likelihood
            try:
                metrics['upn_nll'] += calculate_nll(future, upn_mean, upn_cov) * batch_size
            except Exception as e:
                print(f"Error in UPN NLL calculation: {e}")
                metrics['upn_nll'] += upn_batch_mse * batch_size
                
            try:
                metrics['ensemble_nll'] += calculate_nll(future, ensemble_mean, ensemble_cov) * batch_size
            except Exception as e:
                print(f"Error in Ensemble NLL calculation: {e}")
                metrics['ensemble_nll'] += ensemble_batch_mse * batch_size
            
            # Move tensors to CPU for numpy calculations
            future_np = future.cpu().numpy()
            upn_mean_np = upn_mean.cpu().numpy()
            upn_std_np = upn_std.cpu().numpy()
            ensemble_mean_np = ensemble_mean.cpu().numpy()
            ensemble_std_np = ensemble_std.cpu().numpy()
            
            # Calculate CRPS
            try:
                metrics['upn_crps'] += calculate_crps(future_np, upn_mean_np, upn_std_np) * batch_size
            except Exception as e:
                print(f"Error in UPN CRPS calculation: {e}")
                
            try:
                metrics['ensemble_crps'] += calculate_crps(future_np, ensemble_mean_np, ensemble_std_np) * batch_size
            except Exception as e:
                print(f"Error in Ensemble CRPS calculation: {e}")
            
            # Calculate interval coverage
            try:
                metrics['upn_coverage'] += calculate_interval_coverage(future_np, upn_mean_np, upn_std_np) * batch_size
            except Exception as e:
                print(f"Error in UPN coverage calculation: {e}")
                
            try:
                metrics['ensemble_coverage'] += calculate_interval_coverage(future_np, ensemble_mean_np, ensemble_std_np) * batch_size
            except Exception as e:
                print(f"Error in Ensemble coverage calculation: {e}")
            
            # Store data for visualization (only first batch)
            if len(all_history) < 1:
                # Store for visualization
                history_np = history.cpu().numpy()
                future_np = future.cpu().numpy()
                det_pred_np = det_pred.cpu().numpy()
                time_points_np = torch.cat([history_time[0], future_time[0]], dim=0).cpu().numpy()
                
                # Store only first example for visualization
                all_history.append(history_np[0])
                all_future.append(future_np[0])
                all_upn_mean.append(upn_mean_np[0])
                all_upn_std.append(upn_std_np[0])
                all_det_pred.append(det_pred_np[0])
                all_ensemble_mean.append(ensemble_mean_np[0])
                all_ensemble_std.append(ensemble_std_np[0])
                all_time_points.append(time_points_np)
    
    # Calculate average metrics
    for key in metrics:
        metrics[key] /= total_samples
    
    # Print evaluation results
    print(f"\nEvaluation Results for {system_name}:")
    print("-" * 40)
    print(f"Mean Squared Error (MSE):")
    print(f"  UPN: {metrics['upn_mse']:.6f}")
    print(f"  Deterministic: {metrics['det_mse']:.6f}")
    print(f"  Ensemble: {metrics['ensemble_mse']:.6f}")
    
    print("\nNegative Log Likelihood (NLL):")
    print(f"  UPN: {metrics['upn_nll']:.6f}")
    print(f"  Ensemble: {metrics['ensemble_nll']:.6f}")
    
    print("\nContinuous Ranked Probability Score (CRPS):")
    print(f"  UPN: {metrics['upn_crps']:.6f}")
    print(f"  Ensemble: {metrics['ensemble_crps']:.6f}")
    
    print("\n95% Interval Coverage:")
    print(f"  UPN: {metrics['upn_coverage']:.4f} (ideal: 0.95)")
    print(f"  Ensemble: {metrics['ensemble_coverage']:.4f} (ideal: 0.95)")
    
    # Prepare visualization data
    visualization_data = {
        'history': all_history,
        'future': all_future,
        'upn_mean': all_upn_mean,
        'upn_std': all_upn_std,
        'det_pred': all_det_pred,
        'ensemble_mean': all_ensemble_mean,
        'ensemble_std': all_ensemble_std,
        'time_points': all_time_points
    }
    
    return metrics, visualization_data

#######################
# Visualization Functions
#######################

def visualize_lorenz_3d(viz_data, save_path=None):
    """Create 3D visualization of Lorenz predictions"""
    # Extract data
    history = viz_data['history'][0]
    future = viz_data['future'][0]
    upn_mean = viz_data['upn_mean'][0]
    upn_std = viz_data['upn_std'][0]
    ensemble_mean = viz_data['ensemble_mean'][0]
    
    # Combine history and future for ground truth
    ground_truth = np.concatenate([history, future], axis=0)
    
    # Get time points
    time_points = viz_data['time_points'][0]
    history_len = history.shape[0]
    future_len = future.shape[0]
    
    # Create figure for 3D plot
    fig = plt.figure(figsize=(15, 12))
    
    # Ground truth plot (full trajectory)
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
             'k-', linewidth=2, label='Ground Truth')
    
    # Highlight history part
    ax1.plot(history[:, 0], history[:, 1], history[:, 2], 
             'b-', linewidth=3, label='History')
    
    # Add future part
    ax1.plot(future[:, 0], future[:, 1], future[:, 2], 
             'r-', linewidth=2, label='Future')
    
    ax1.set_title('Ground Truth Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # UPN prediction plot
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(history[:, 0], history[:, 1], history[:, 2], 
             'b-', linewidth=3, label='History')
    
    # UPN mean prediction
    ax2.plot(upn_mean[:, 0], upn_mean[:, 1], upn_mean[:, 2], 
             'g-', linewidth=2, label='UPN Mean')
    
    # Future ground truth for reference
    ax2.plot(future[:, 0], future[:, 1], future[:, 2], 
             'r--', alpha=0.5, linewidth=2, label='Future (Truth)')
    
    # Add uncertainty cloud at select points (every 5 steps)
    for i in range(0, future_len, 5):
        # Create small sphere at this point to represent uncertainty
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        
        # Use the standard deviation to scale the sphere
        std_scale = 2.0  # 2 std for 95% confidence
        
        # Create sphere for x, y, z with appropriate scaling
        x_sphere = upn_mean[i, 0] + upn_std[i, 0] * std_scale * np.outer(np.cos(u), np.sin(v))
        y_sphere = upn_mean[i, 1] + upn_std[i, 1] * std_scale * np.outer(np.sin(u), np.sin(v))
        z_sphere = upn_mean[i, 2] + upn_std[i, 2] * std_scale * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot the sphere with transparency
        ax2.plot_surface(x_sphere, y_sphere, z_sphere, color='g', alpha=0.1)
    
    ax2.set_title('UPN Prediction with Uncertainty')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Ensemble prediction plot
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(history[:, 0], history[:, 1], history[:, 2], 
             'b-', linewidth=3, label='History')
    
    # Ensemble mean prediction
    ax3.plot(ensemble_mean[:, 0], ensemble_mean[:, 1], ensemble_mean[:, 2], 
             'm-', linewidth=2, label='Ensemble Mean')
    
    # Future ground truth for reference
    ax3.plot(future[:, 0], future[:, 1], future[:, 2], 
             'r--', alpha=0.5, linewidth=2, label='Future (Truth)')
    
    ax3.set_title('Ensemble Prediction')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # Uncertainty comparison plot
    ax4 = fig.add_subplot(224)
    # Plot how uncertainty (std) grows over time for each dimension
    future_time_points = time_points[history_len:history_len+future_len]
    
    # UPN uncertainty growth
    ax4.plot(future_time_points, upn_std[:, 0], 'g-', label='UPN X Std')
    ax4.plot(future_time_points, upn_std[:, 1], 'g--', label='UPN Y Std')
    ax4.plot(future_time_points, upn_std[:, 2], 'g-.', label='UPN Z Std')
    
    # Ensemble uncertainty growth
    ax4.plot(future_time_points, viz_data['ensemble_std'][0][:, 0], 'm-', label='Ensemble X Std')
    ax4.plot(future_time_points, viz_data['ensemble_std'][0][:, 1], 'm--', label='Ensemble Y Std')
    ax4.plot(future_time_points, viz_data['ensemble_std'][0][:, 2], 'm-.', label='Ensemble Z Std')
    
    ax4.set_title('Uncertainty Growth Over Time')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Standard Deviation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def visualize_lorenz_2d(viz_data, save_path=None):
    """Create 2D visualizations of Lorenz predictions for each dimension"""
    # Extract data
    history = viz_data['history'][0]
    future = viz_data['future'][0]
    upn_mean = viz_data['upn_mean'][0]
    upn_std = viz_data['upn_std'][0]
    ensemble_mean = viz_data['ensemble_mean'][0]
    ensemble_std = viz_data['ensemble_std'][0]
    
    # Combine history and future for ground truth
    ground_truth = np.concatenate([history, future], axis=0)
    
    # Get time points
    time_points = viz_data['time_points'][0]
    history_len = history.shape[0]
    future_time_points = time_points[history_len:history_len+len(future)]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    dim_names = ['X', 'Y', 'Z']
    
    for i, name in enumerate(dim_names):
        ax = axes[i]
        
        # Plot ground truth
        ax.plot(time_points, ground_truth[:, i], 'k-', label='Ground Truth', linewidth=2)
        
        # Indicate history region
        ax.axvspan(time_points[0], time_points[history_len-1], alpha=0.2, color='blue', label='History Region')
        
        # Plot UPN prediction with uncertainty
        ax.plot(future_time_points, upn_mean[:, i], 'g-', label='UPN Mean', linewidth=2)
        ax.fill_between(
            future_time_points,
            upn_mean[:, i] - 2 * upn_std[:, i],
            upn_mean[:, i] + 2 * upn_std[:, i],
            color='g', alpha=0.2, label='UPN 95% CI'
        )
        
        # Plot ensemble prediction with uncertainty
        ax.plot(future_time_points, ensemble_mean[:, i], 'm-', label='Ensemble Mean', linewidth=2)
        ax.fill_between(
            future_time_points,
            ensemble_mean[:, i] - 2 * ensemble_std[:, i],
            ensemble_mean[:, i] + 2 * ensemble_std[:, i],
            color='m', alpha=0.2, label='Ensemble 95% CI'
        )
        
        ax.set_title(f'Lorenz System - {name} Dimension')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{name}')
        ax.grid(True, alpha=0.3)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_training_curves(upn_losses, ensemble_losses, det_losses, save_path=None):
    """Plot training curves for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # UPN training curves
    axes[0].plot(upn_losses[0], label='Train')
    axes[0].plot(upn_losses[1], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('UPN Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Ensemble training curves
    axes[1].plot(ensemble_losses[0], label='Train')
    axes[1].plot(ensemble_losses[1], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Ensemble Training Progress')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Deterministic training curves
    axes[2].plot(det_losses[0], label='Train')
    axes[2].plot(det_losses[1], label='Validation')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Deterministic Training Progress')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

#######################
# Main Evaluation Function
#######################

def evaluate_lorenz_system():
    """Evaluate UPN on the Lorenz attractor (chaotic system) with extended analysis"""
    print("Evaluating UPN on Lorenz Attractor (Chaotic System)")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data
    print("Generating Lorenz data...")
    data, time_points = generate_lorenz_data(
        num_trajectories=100, 
        t_max=15.0, 
        dt=0.01, 
        noise_scale=0.1
    )
    print(f"Generated {len(data)} trajectories with {len(time_points)} time points each")
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_lorenz_dataloaders(
        data, time_points, 
        batch_size=16, 
        history_length=20, 
        future_length=50
    )
    
    # Create trainer
    trainer = DynamicalSystemTrainer(
        state_dim=3,
        hidden_dim=128,
        use_diagonal_cov=True,
        adjoint=False,
        learning_rate=5e-4
    ).to(device)
    
    # Train UPN model
    print("\nTraining UPN model...")
    upn_model, upn_train_losses, upn_val_losses = trainer.train_upn(
        train_loader, val_loader, num_epochs=25, lr=5e-4
    )
    
    # Train ensemble model
    print("\nTraining ensemble model...")
    ensemble_model, ensemble_train_losses, ensemble_val_losses = trainer.train_ensemble(
        train_loader, val_loader, num_models=8, num_epochs=25, lr=5e-4
    )
    
    # Train deterministic model
    print("\nTraining deterministic model...")
    det_model, det_train_losses, det_val_losses = trainer.train_deterministic(
        train_loader, val_loader, num_epochs=25, lr=5e-4
    )
    
    # Plot training curves
    plot_training_curves(
        (upn_train_losses, upn_val_losses),
        (ensemble_train_losses, ensemble_val_losses),
        (det_train_losses, det_val_losses),
        save_path='lorenz_training_curves.png'
    )
    
    # Evaluate models
    print("\nEvaluating models on test data...")
    metrics, viz_data = evaluate_models(
        upn_model, det_model, ensemble_model, test_loader, "Lorenz Attractor"
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_lorenz_3d(viz_data, save_path='lorenz_3d_predictions.png')
    visualize_lorenz_2d(viz_data, save_path='lorenz_2d_predictions.png')
    
    # Save models
    print("\nSaving models...")
    torch.save({
        'model_state_dict': upn_model.state_dict(),
    }, 'upn_lorenz_model.pt')
    
    torch.save({
        'model_state_dict': ensemble_model.state_dict(),
    }, 'ensemble_lorenz_model.pt')
    
    torch.save({
        'model_state_dict': det_model.state_dict(),
    }, 'det_lorenz_model.pt')
   
    print("Evaluation completed!")
    print("Results and visualizations saved to current directory")
   
    return metrics, viz_data

def evaluate_across_time_horizons(upn_model, det_model, ensemble_model, test_loader, system_name="Lorenz Attractor"):
    """Evaluate models at multiple prediction horizons"""
    print(f"Evaluating models across time horizons for {system_name}")
   
    # Define different horizon lengths to evaluate
    horizons = [10, 20, 40, 60, 80, 100]
   
    # Initialize metrics storage for each horizon
    horizon_metrics = {h: {'upn_mse': 0.0, 'det_mse': 0.0, 'ensemble_mse': 0.0, 
                            'upn_nll': 0.0, 'ensemble_nll': 0.0,
                            'upn_coverage': 0.0, 'ensemble_coverage': 0.0} 
                        for h in horizons}
   
    # Process test data
    batch_count = 0
    total_samples = 0
   
    with torch.no_grad():
        for history, history_time, future, future_time in tqdm(test_loader, desc="Evaluating horizons"):
            batch_count += 1
            
            history = history.to(device)
            history_time = history_time.to(device)
            future = future.to(device)
            future_time = future_time.to(device)
            
            batch_size = history.shape[0]
            total_samples += batch_size
            
            # Get initial state for models
            initial_state = history[:, -1]
            
            # Initial covariance for UPN
            state_dim = initial_state.shape[1]
            initial_cov = torch.eye(state_dim, device=device).unsqueeze(0).expand(batch_size, -1, -1) * 1e-4
            
            # Get the last time point from history
            last_time = history_time[:, -1].unsqueeze(1)
            
            # Create time span for full trajectory ODE integration
            t_span = torch.cat([last_time[0], future_time[0]], dim=0)
            
            # Make full predictions once
            upn_mean, upn_cov = upn_model.predict(initial_state, initial_cov, t_span)
            upn_mean = upn_mean[1:]  # Remove first point
            upn_cov = upn_cov[1:]
            
            det_pred = det_model.predict(initial_state, t_span)
            det_pred = det_pred[1:]  # Remove first point
            
            ensemble_mean, ensemble_cov = ensemble_model.predict(initial_state, t_span)
            ensemble_mean = ensemble_mean[1:]  # Remove first point
            ensemble_cov = ensemble_cov[1:]
            
            # Transpose to match future shape
            upn_mean = upn_mean.permute(1, 0, 2)
            upn_cov = upn_cov.permute(1, 0, 2, 3)
            det_pred = det_pred.permute(1, 0, 2)
            ensemble_mean = ensemble_mean.permute(1, 0, 2)
            ensemble_cov = ensemble_cov.permute(1, 0, 2, 3)
            
            # Get standard deviation from covariance
            upn_std = torch.sqrt(torch.diagonal(upn_cov, dim1=2, dim2=3))
            ensemble_std = torch.sqrt(torch.diagonal(ensemble_cov, dim1=2, dim2=3))
            
            # For each horizon length, compute metrics
            for h in horizons:
                if h > future.shape[1]:
                    continue  # Skip if horizon exceeds available future data
                
                # Extract predictions and ground truth up to horizon h
                future_h = future[:, :h]
                upn_mean_h = upn_mean[:, :h]
                upn_cov_h = upn_cov[:, :h]
                upn_std_h = upn_std[:, :h]
                det_pred_h = det_pred[:, :h]
                ensemble_mean_h = ensemble_mean[:, :h]
                ensemble_cov_h = ensemble_cov[:, :h]
                ensemble_std_h = ensemble_std[:, :h]
                
                # Calculate MSE for each horizon
                upn_batch_mse = ((upn_mean_h - future_h) ** 2).mean().item()
                det_batch_mse = ((det_pred_h - future_h) ** 2).mean().item()
                ensemble_batch_mse = ((ensemble_mean_h - future_h) ** 2).mean().item()
                
                horizon_metrics[h]['upn_mse'] += upn_batch_mse * batch_size
                horizon_metrics[h]['det_mse'] += det_batch_mse * batch_size
                horizon_metrics[h]['ensemble_mse'] += ensemble_batch_mse * batch_size
                
                # Calculate NLL
                try:
                    horizon_metrics[h]['upn_nll'] += calculate_nll(future_h, upn_mean_h, upn_cov_h) * batch_size
                except Exception as e:
                    horizon_metrics[h]['upn_nll'] += upn_batch_mse * batch_size
                    
                try:
                    horizon_metrics[h]['ensemble_nll'] += calculate_nll(future_h, ensemble_mean_h, ensemble_cov_h) * batch_size
                except Exception as e:
                    horizon_metrics[h]['ensemble_nll'] += ensemble_batch_mse * batch_size
                
                # Move tensors to CPU for numpy calculations
                future_np = future_h.cpu().numpy()
                upn_mean_np = upn_mean_h.cpu().numpy()
                upn_std_np = upn_std_h.cpu().numpy()
                ensemble_mean_np = ensemble_mean_h.cpu().numpy()
                ensemble_std_np = ensemble_std_h.cpu().numpy()
                
                # Calculate interval coverage
                try:
                    horizon_metrics[h]['upn_coverage'] += calculate_interval_coverage(future_np, upn_mean_np, upn_std_np) * batch_size
                except Exception as e:
                    print(f"Error in UPN coverage calculation for horizon {h}: {e}")
                    
                try:
                    horizon_metrics[h]['ensemble_coverage'] += calculate_interval_coverage(future_np, ensemble_mean_np, ensemble_std_np) * batch_size
                except Exception as e:
                    print(f"Error in Ensemble coverage calculation for horizon {h}: {e}")
   
    # Calculate average metrics for each horizon
    for h in horizons:
        if total_samples > 0:
            for key in horizon_metrics[h]:
                horizon_metrics[h][key] /= total_samples
   
    # Visualize results
    visualize_horizon_results(horizon_metrics, horizons, system_name)
   
    return horizon_metrics

def visualize_horizon_results(horizon_metrics, horizons, system_name):
    """Visualize metrics across different prediction horizons"""
    # Filter out horizons with no data
    valid_horizons = [h for h in horizons if 'upn_mse' in horizon_metrics[h] and horizon_metrics[h]['upn_mse'] > 0]
    
    if not valid_horizons:
        print("No valid horizon metrics found.")
        return
        
    # Extract metrics for each valid horizon
    upn_mse = [horizon_metrics[h]['upn_mse'] for h in valid_horizons]
    det_mse = [horizon_metrics[h]['det_mse'] for h in valid_horizons]
    ens_mse = [horizon_metrics[h]['ensemble_mse'] for h in valid_horizons]
    
    upn_nll = [horizon_metrics[h]['upn_nll'] for h in valid_horizons]
    ens_nll = [horizon_metrics[h]['ensemble_nll'] for h in valid_horizons]
    
    upn_coverage = [horizon_metrics[h]['upn_coverage'] for h in valid_horizons]
    ens_coverage = [horizon_metrics[h]['ensemble_coverage'] for h in valid_horizons]
    
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # MSE vs Horizon plot
    ax = axes[0]
    ax.plot(valid_horizons, upn_mse, 'g-o', label='UPN', linewidth=2, markersize=8)
    ax.plot(valid_horizons, det_mse, 'r-^', label='Deterministic', linewidth=2, markersize=8)
    ax.plot(valid_horizons, ens_mse, 'm-s', label='Ensemble', linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error vs. Horizon')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # NLL vs Horizon plot
    ax = axes[1]
    ax.plot(valid_horizons, upn_nll, 'g-o', label='UPN', linewidth=2, markersize=8)
    ax.plot(valid_horizons, ens_nll, 'm-s', label='Ensemble', linewidth=2, markersize=8)
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('NLL')
    ax.set_title('Negative Log Likelihood vs. Horizon')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Coverage vs Horizon plot
    ax = axes[2]
    ax.plot(valid_horizons, upn_coverage, 'g-o', label='UPN', linewidth=2, markersize=8)
    ax.plot(valid_horizons, ens_coverage, 'm-s', label='Ensemble', linewidth=2, markersize=8)
    ax.axhline(y=0.95, color='black', linestyle='--', label='Ideal Coverage')
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('Coverage Probability')
    ax.set_title('95% Interval Coverage vs. Horizon')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f"{system_name}: Performance Metrics Across Different Prediction Horizons", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{system_name.lower().replace(" ", "_")}_horizon_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_lorenz_3d_with_time_analysis(viz_data, save_path=None):
    """Enhanced 3D visualization of Lorenz predictions with time analysis"""
    # Create figure with more subplots
    fig = plt.figure(figsize=(18, 15))
    
    # Extract data
    history = viz_data['history'][0]
    future = viz_data['future'][0]
    upn_mean = viz_data['upn_mean'][0]
    upn_std = viz_data['upn_std'][0]
    ensemble_mean = viz_data['ensemble_mean'][0]
    ensemble_std = viz_data['ensemble_std'][0]
    
    # Get time points
    time_points = viz_data['time_points'][0]
    history_len = history.shape[0]
    future_len = future.shape[0]
    future_time_points = time_points[history_len:history_len+future_len]
    
    # 3D visualization of full trajectory
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(history[:, 0], history[:, 1], history[:, 2], 'b-', linewidth=3, label='History')
    ax1.plot(future[:, 0], future[:, 1], future[:, 2], 'r-', linewidth=2, label='Future (Truth)')
    ax1.set_title('Ground Truth Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # UPN mean prediction with uncertainty at different points
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(history[:, 0], history[:, 1], history[:, 2], 'b-', linewidth=3, label='History')
    ax2.plot(upn_mean[:, 0], upn_mean[:, 1], upn_mean[:, 2], 'g-', linewidth=2, label='UPN Mean')
    ax2.plot(future[:, 0], future[:, 1], future[:, 2], 'r--', alpha=0.5, linewidth=1, label='Future (Truth)')
    
    # Sample points along trajectory to visualize uncertainty growth
    uncertainty_indices = [0, int(future_len*0.25), int(future_len*0.5), int(future_len*0.75), future_len-1]
    colors = ['green', 'yellowgreen', 'yellow', 'orange', 'red']
    
    # Add small spheres at selected points
    for i, idx in enumerate(uncertainty_indices):
        if idx >= len(upn_mean):
            continue
            
        # Create small sphere at this point to represent uncertainty
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        
        # Use the standard deviation to scale the sphere
        std_scale = 2.0  # 2 std for 95% confidence
        
        # Create sphere for x, y, z with appropriate scaling
        x_sphere = upn_mean[idx, 0] + upn_std[idx, 0] * std_scale * np.outer(np.cos(u), np.sin(v))
        y_sphere = upn_mean[idx, 1] + upn_std[idx, 1] * std_scale * np.outer(np.sin(u), np.sin(v))
        z_sphere = upn_mean[idx, 2] + upn_std[idx, 2] * std_scale * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot the sphere with transparency and different colors
        ax2.plot_surface(x_sphere, y_sphere, z_sphere, color=colors[i], alpha=0.15)
        
        # Add a point to mark the center
        ax2.scatter([upn_mean[idx, 0]], [upn_mean[idx, 1]], [upn_mean[idx, 2]], 
                    color=colors[i], s=50, label=f't={future_time_points[idx]:.2f}')
    
    ax2.set_title('UPN Prediction with Uncertainty Growth')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # MSE and standard deviation over time
    ax3 = fig.add_subplot(223)
    
    # Calculate MSE at each time step
    upn_mse_over_time = ((upn_mean - future) ** 2).mean(axis=1)
    ensemble_mse_over_time = ((ensemble_mean - future) ** 2).mean(axis=1)
    
    # Plot MSE
    ax3.plot(future_time_points, upn_mse_over_time, 'g-', label='UPN MSE')
    ax3.plot(future_time_points, ensemble_mse_over_time, 'm-', label='Ensemble MSE')
    
    ax3.set_title('MSE Over Time')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('MSE')
    ax3.set_yscale('log')  # Use log scale to visualize exponential growth
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Uncertainty growth over time
    ax4 = fig.add_subplot(224)
    
    # Calculate average standard deviation across dimensions
    upn_avg_std = upn_std.mean(axis=1)
    ensemble_avg_std = ensemble_std.mean(axis=1)
    
    # Plot average standard deviation
    ax4.plot(future_time_points, upn_avg_std, 'g-', label='UPN Avg Std')
    ax4.plot(future_time_points, ensemble_avg_std, 'm-', label='Ensemble Avg Std')
    
    # Plot individual dimension standard deviations
    for d in range(3):
        ax4.plot(future_time_points, upn_std[:, d], 'g--', alpha=0.5, 
                label=f'UPN Dim {d+1} Std' if d == 0 else None)
        ax4.plot(future_time_points, ensemble_std[:, d], 'm--', alpha=0.5,
                label=f'Ensemble Dim {d+1} Std' if d == 0 else None)
    
    ax4.set_title('Uncertainty Growth Over Time')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_yscale('log')  # Use log scale to better visualize growth patterns
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

if __name__ == "__main__":
    # Run the evaluation
    metrics, viz_data = evaluate_lorenz_system()
    
    # Optionally run horizon analysis
    # print("\nRunning horizon analysis...")
    # horizon_metrics = evaluate_across_time_horizons(upn_model, det_model, ensemble_model, test_loader)