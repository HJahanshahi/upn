import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DynamicalSystemDataset(Dataset):
    """Dataset for dynamical system trajectories"""
    def __init__(self, trajectories, time_points):
        """
        Initialize a dynamical system dataset.
        
        Args:
            trajectories: Trajectory tensor [num_trajectories, time_steps, state_dim]
            time_points: Time points tensor [time_steps]
        """
        self.trajectories = torch.tensor(trajectories, dtype=torch.float32)
        self.time_points = torch.tensor(time_points, dtype=torch.float32)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.time_points

def generate_oscillator_data(num_trajectories=100, t_max=10.0, dt=0.1, noise_scale=0.05):
    """
    Generate data from a simple harmonic oscillator.
    
    Args:
        num_trajectories: Number of trajectories to generate
        t_max: Maximum time
        dt: Time step
        noise_scale: Scale of observation noise
        
    Returns:
        time_points: Time points array [time_steps]
        trajectories: Trajectory tensor [num_trajectories, time_steps, 2]
    """
    # Time points
    time_points = np.arange(0, t_max, dt)
    num_steps = len(time_points)
    
    # Initialize storage for trajectories
    trajectories = np.zeros((num_trajectories, num_steps, 2))
    
    # System matrix for harmonic oscillator
    A = np.array([
        [0.0, 1.0],
        [-1.0, -0.1]  # Spring constant = 1, damping = 0.1
    ])
    
    # Generate trajectories
    for i in range(num_trajectories):
        # Random initial state
        x0 = np.random.uniform(-1, 1, 2)
        
        # Current state
        x = x0.copy()
        trajectories[i, 0] = x
        
        # Simulate trajectory
        for j in range(1, num_steps):
            # Euler integration
            dx = A @ x * dt
            x = x + dx
            trajectories[i, j] = x
        
        # Add observation noise
        trajectories[i] += noise_scale * np.random.randn(num_steps, 2)
    
    return time_points, trajectories


def prepare_dataloaders(time_points, trajectories, batch_size=16, train_split=0.7, val_split=0.15):
    """
    Prepare DataLoaders for training, validation, and testing.
    
    Args:
        time_points: Time points array [time_steps]
        trajectories: Trajectory tensor [num_trajectories, time_steps, state_dim]
        batch_size: Batch size for DataLoaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    # Get number of trajectories
    num_trajectories = len(trajectories)
    
    # Shuffle indices
    indices = np.random.permutation(num_trajectories)
    
    # Split indices
    train_size = int(train_split * num_trajectories)
    val_size = int(val_split * num_trajectories)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Split data
    train_trajectories = trajectories[train_indices]
    val_trajectories = trajectories[val_indices]
    test_trajectories = trajectories[test_indices]
    
    # Create datasets
    train_dataset = DynamicalSystemDataset(train_trajectories, time_points)
    val_dataset = DynamicalSystemDataset(val_trajectories, time_points)
    test_dataset = DynamicalSystemDataset(test_trajectories, time_points)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader