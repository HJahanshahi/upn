import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from upn import (
    DynamicalSystemUPN,
    generate_oscillator_data,
    prepare_dataloaders,
    plot_trajectory_with_uncertainty,
    plot_2d_phase_with_uncertainty,
    plot_training_curves
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # Generate oscillator data
    time_points, trajectories = generate_oscillator_data(
        num_trajectories=100,
        t_max=20.0,
        dt=0.1,
        noise_scale=0.05
    )
    print(f"Generated {len(trajectories)} trajectories with {len(time_points)} time points each")
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        time_points,
        trajectories,
        batch_size=16,
        train_split=0.7,
        val_split=0.15
    )
    
    # Create UPN model
    upn_model = DynamicalSystemUPN(
        state_dim=2,
        hidden_dim=64,
        use_diagonal_cov=True,
        learning_rate=1e-3
    ).to(device)
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(
        upn_model.optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        # verbose=True
    )
    
    # Train model
    print("Training UPN model...")
    train_losses, val_losses = upn_model.train(
        train_loader,
        val_loader,
        num_epochs=100,
        scheduler=scheduler,
        early_stopping=20,
        verbose=True
    )
    
    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        title='UPN Training on Oscillator Data',
        save_path='oscillator_training_curves.png'
    )
    
    # Evaluate on test set
    test_loss = upn_model.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Make predictions on a test trajectory
    test_trajectory = trajectories[-1]  # Use last trajectory for visualization
    initial_state = torch.tensor(test_trajectory[0:1], dtype=torch.float32)
    initial_cov = torch.eye(2).unsqueeze(0) * 1e-4
    
    # Get predictions
    pred_mean, pred_cov = upn_model.predict(initial_state, initial_cov, torch.tensor(time_points, dtype=torch.float32))
    
    # Convert to numpy for plotting
    pred_mean_np = pred_mean.cpu().numpy()
    pred_cov_np = pred_cov.cpu().numpy()
    
    # Squeeze batch dimension
    pred_mean_np = pred_mean_np[:, 0, :]
    pred_cov_np = pred_cov_np[:, 0, :, :]
    
    # Plot trajectory with uncertainty
    plot_trajectory_with_uncertainty(
        time_points,
        test_trajectory,
        pred_mean_np,
        pred_cov_np,
        dims=[0, 1],
        interval=20,
        title='Oscillator Trajectory Prediction',
        dim_labels=['Position', 'Velocity'],
        save_path='oscillator_trajectory_prediction.png'
    )
    
    # Plot phase plot with uncertainty
    plot_2d_phase_with_uncertainty(
        test_trajectory,
        pred_mean_np,
        pred_cov_np,
        dims=[0, 1],
        interval=20,
        title='Oscillator Phase Space Prediction',
        dim_labels=['Position', 'Velocity'],
        save_path='oscillator_phase_prediction.png'
    )
    
    print("Done! Check the output images for results.")

if __name__ == "__main__":
    main()