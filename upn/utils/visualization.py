import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(ax, x, y, cov, n_std=2.0, facecolor='none', **kwargs):
    """
    Plot a confidence ellipse representing a covariance matrix.
    
    Args:
        ax: Matplotlib axis
        x: X-coordinate of ellipse center
        y: Y-coordinate of ellipse center
        cov: 2x2 covariance matrix
        n_std: Number of standard deviations to include
        facecolor: Color for filling the ellipse
        **kwargs: Additional keyword arguments for Ellipse
    """
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues in decreasing order
    sort_indices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sort_indices]
    eigenvecs = eigenvecs[:, sort_indices]
    
    # Compute angle
    theta = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Width and height are "full" widths, not radius
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    # Create ellipse
    ellipse = Ellipse(
        xy=(x, y),
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        **kwargs
    )
    
    return ax.add_patch(ellipse)

def plot_trajectory_with_uncertainty(time_points, true_trajectory, pred_mean, pred_cov, 
                                     dims=[0, 1], interval=10, title=None, figsize=(12, 6),
                                     dim_labels=None, save_path=None):
    """
    Plot a trajectory with uncertainty.
    
    Args:
        time_points: Time points array [time_steps]
        true_trajectory: True trajectory array [time_steps, state_dim]
        pred_mean: Predicted mean array [time_steps, state_dim]
        pred_cov: Predicted covariance array [time_steps, state_dim, state_dim]
        dims: List of dimensions to plot
        interval: Interval for plotting uncertainty ellipses
        title: Plot title
        figsize: Figure size
        dim_labels: Labels for dimensions
        save_path: Path to save the figure
    """
    # Make sure everything is numpy arrays
    time_points = np.array(time_points)
    true_trajectory = np.array(true_trajectory)
    pred_mean = np.array(pred_mean)
    pred_cov = np.array(pred_cov)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Default dimension labels
    if dim_labels is None:
        dim_labels = [f"Dimension {d}" for d in dims]
    
    # Loop over dimensions
    for i, (dim, label) in enumerate(zip(dims, dim_labels)):
        ax = axes[i]
        
        # Plot true trajectory
        ax.plot(time_points, true_trajectory[:, dim], 'k-', label='True')
        
        # Plot predicted trajectory
        ax.plot(time_points, pred_mean[:, dim], 'r-', label='UPN Mean')
        
        # Plot uncertainty bounds (95% confidence interval)
        std_dev = np.sqrt(pred_cov[:, dim, dim])
        ax.fill_between(
            time_points,
            pred_mean[:, dim] - 1.96 * std_dev,
            pred_mean[:, dim] + 1.96 * std_dev,
            color='r', alpha=0.2, label='95% CI'
        )
        
        ax.set_title(f'{label} vs Time')
        ax.set_xlabel('Time')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def plot_2d_phase_with_uncertainty(true_trajectory, pred_mean, pred_cov, 
                                  dims=[0, 1], interval=10, title=None, figsize=(8, 8),
                                  dim_labels=None, save_path=None):
    """
    Plot a 2D phase plot with uncertainty ellipses.
    
    Args:
        true_trajectory: True trajectory array [time_steps, state_dim]
        pred_mean: Predicted mean array [time_steps, state_dim]
        pred_cov: Predicted covariance array [time_steps, state_dim, state_dim]
        dims: List of 2 dimensions to plot
        interval: Interval for plotting uncertainty ellipses
        title: Plot title
        figsize: Figure size
        dim_labels: Labels for dimensions
        save_path: Path to save the figure
    """
    # Make sure everything is numpy arrays
    true_trajectory = np.array(true_trajectory)
    pred_mean = np.array(pred_mean)
    pred_cov = np.array(pred_cov)
    
    # Check that only 2 dimensions are specified
    assert len(dims) == 2, "Must specify exactly 2 dimensions for 2D phase plot"
    
    # Default dimension labels
    if dim_labels is None:
        dim_labels = [f"Dimension {d}" for d in dims]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot true trajectory
    ax.plot(true_trajectory[:, dims[0]], true_trajectory[:, dims[1]], 'k-', label='True')
    
    # Plot predicted trajectory
    ax.plot(pred_mean[:, dims[0]], pred_mean[:, dims[1]], 'r-', label='UPN Mean')
    
    # Plot uncertainty ellipses at regular intervals
    time_steps = len(pred_mean)
    for i in range(0, time_steps, interval):
        # Extract 2D covariance for the selected dimensions
        cov_2d = pred_cov[i, :, :][np.ix_(dims, dims)]
        
        # Plot confidence ellipse
        confidence_ellipse(
            ax, 
            pred_mean[i, dims[0]], 
            pred_mean[i, dims[1]], 
            cov_2d, 
            n_std=2.0, 
            edgecolor='r', 
            alpha=0.2
        )
    
    ax.set_title('Phase Plot' if title is None else title)
    ax.set_xlabel(dim_labels[0])
    ax.set_ylabel(dim_labels[1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_training_curves(train_losses, val_losses=None, title=None, figsize=(10, 6), 
                         log_scale=False, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        title: Plot title
        figsize: Figure size
        log_scale: Whether to use log scale for y-axis
        save_path: Path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training loss
    ax.plot(train_losses, 'b-', label='Training Loss')
    
    # Plot validation loss if provided
    if val_losses is not None:
        ax.plot(val_losses, 'r-', label='Validation Loss')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Training Curves')
    
    # Set labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax