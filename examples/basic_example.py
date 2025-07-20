import torch
import matplotlib.pyplot as plt
import numpy as np
from upn.core import UPN

def generate_oscillator_data(num_samples=100, t_max=10.0, dt=0.1, noise_scale=0.05):
    """Generate data from a simple harmonic oscillator"""
    t = torch.arange(0, t_max, dt)
    x0 = torch.tensor([[1.0, 0.0]])  # Initial state [position, velocity]
    

    A = torch.tensor([
        [0.0, 1.0],
        [-1.0, -0.1]  # Spring constant = 1, damping = 0.1
    ])
    
    # Solve for true trajectory
    x = x0.clone()
    trajectory = [x0.clone()]
    
    for i in range(1, len(t)):
        dx = torch.matmul(x, A.T) * dt
        x = x + dx
        trajectory.append(x.clone())
    
    trajectory = torch.cat(trajectory, dim=0).unsqueeze(0)  # [1, time_steps, 2]
    
    # Add noise
    noisy_trajectory = trajectory + noise_scale * torch.randn_like(trajectory)
    
    return t, trajectory, noisy_trajectory

# Test the UPN model
def main():
    # Generate sample data
    t, true_traj, noisy_traj = generate_oscillator_data(t_max=20.0)
    
    # Create UPN model
    state_dim = 2
    upn_model = UPN(state_dim=state_dim, hidden_dim=32, use_diagonal_cov=True)
    
    # Set initial state and covariance
    initial_state = noisy_traj[0, 0].unsqueeze(0)  # [1, 2]
    initial_cov = 0.1 * torch.eye(state_dim).unsqueeze(0)  # [1, 2, 2]
    
    # Make predictions
    with torch.no_grad():
        pred_mean, pred_cov = upn_model.predict(initial_state, initial_cov, t)
    
    # Convert to numpy for plotting
    t_np = t.numpy()
    true_traj_np = true_traj[0].numpy()
    noisy_traj_np = noisy_traj[0].numpy()
    pred_mean_np = pred_mean[:, 0].numpy()
    
    # Extract standard deviations for uncertainty visualization
    pred_std_np = np.sqrt(np.diagonal(pred_cov[:, 0].numpy(), axis1=1, axis2=2))
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Position plot
    axes[0].plot(t_np, true_traj_np[:, 0], 'k-', label='True')
    axes[0].plot(t_np, noisy_traj_np[:, 0], 'ko', alpha=0.3, label='Noisy')
    axes[0].plot(t_np, pred_mean_np[:, 0], 'r-', label='UPN Mean')
    
    # Plot uncertainty bounds (95% confidence interval)
    axes[0].fill_between(
        t_np,
        pred_mean_np[:, 0] - 1.96 * pred_std_np[:, 0],
        pred_mean_np[:, 0] + 1.96 * pred_std_np[:, 0],
        color='r', alpha=0.2, label='95% CI'
    )
    
    axes[0].set_title('Position vs Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    
    # Velocity plot
    axes[1].plot(t_np, true_traj_np[:, 1], 'k-', label='True')
    axes[1].plot(t_np, noisy_traj_np[:, 1], 'ko', alpha=0.3, label='Noisy')
    axes[1].plot(t_np, pred_mean_np[:, 1], 'r-', label='UPN Mean')
    
    # Plot uncertainty bounds (95% confidence interval)
    axes[1].fill_between(
        t_np,
        pred_mean_np[:, 1] - 1.96 * pred_std_np[:, 1],
        pred_mean_np[:, 1] + 1.96 * pred_std_np[:, 1],
        color='r', alpha=0.2, label='95% CI'
    )
    
    axes[1].set_title('Velocity vs Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('upn_oscillator_example.png')
    plt.show()

if __name__ == "__main__":
    main()