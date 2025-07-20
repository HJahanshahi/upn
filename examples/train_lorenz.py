import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from upn.applications.lorenz import (
    lorenz_system,
    generate_lorenz_data,
    prepare_lorenz_dataloaders,
    visualize_lorenz_2d,
    visualize_lorenz_3d,
    plot_training_curves,
    evaluate_lorenz_system,
    evaluate_models,
    DynamicalSystemTrainer
)


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def quick_demo():
    """Quick demonstration with fewer epochs for testing"""
    print("Quick UPN Demo on Lorenz System")
    print("=" * 40)
    
    # Generate smaller dataset for quick demo
    print("Generating Lorenz system data...")
    trajectories, time_points = generate_lorenz_data(
        num_trajectories=50,  # Smaller dataset
        t_max=10.0,
        dt=0.01,
        noise_scale=0.1
    )
    print(f"Generated {len(trajectories)} trajectories with {len(time_points)} time points each")
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_lorenz_dataloaders(
        trajectories,
        time_points,
        batch_size=16,
        history_length=20,
        future_length=30,
        train_split=0.7,
        val_split=0.15
    )
    
    # Create trainer
    trainer = DynamicalSystemTrainer(
        state_dim=3,
        hidden_dim=128,
        use_diagonal_cov=True,
        adjoint=False,
        learning_rate=5e-4
    ).to(device)
    
    # Train UPN model (just a few epochs for demo)
    print("\nTraining UPN model...")
    upn_model, upn_train_losses, upn_val_losses = trainer.train_upn(
        train_loader, val_loader, num_epochs=5, lr=5e-4  # Few epochs for demo
    )
    
    # Train ensemble model for comparison
    print("\nTraining ensemble model...")
    ensemble_model, ensemble_train_losses, ensemble_val_losses = trainer.train_ensemble(
        train_loader, val_loader, num_models=3, num_epochs=5, lr=5e-4  # Smaller ensemble
    )
    
    # Train deterministic model
    print("\nTraining deterministic model...")
    det_model, det_train_losses, det_val_losses = trainer.train_deterministic(
        train_loader, val_loader, num_epochs=5, lr=5e-4
    )
    
    # Plot training curves
    plot_training_curves(
        (upn_train_losses, upn_val_losses),
        (ensemble_train_losses, ensemble_val_losses),
        (det_train_losses, det_val_losses),
        save_path='quick_demo_training_curves.png'
    )
    
    # Quick evaluation
    print("\nEvaluating models...")
    metrics, viz_data = evaluate_models(
        upn_model, det_model, ensemble_model, test_loader, "Lorenz Demo"
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_lorenz_3d(viz_data, save_path='quick_demo_3d.png')
    visualize_lorenz_2d(viz_data, save_path='quick_demo_2d.png')
    
    print("Quick demo completed! Check the output files.")
    return metrics, viz_data

def full_evaluation():
    """Full evaluation as described in the paper"""
    print("Full UPN Evaluation on Lorenz System")
    print("=" * 50)
    
    metrics, viz_data = evaluate_lorenz_system()
    
    return metrics, viz_data

def custom_visualization_demo():
    """Demonstrate custom visualization capabilities"""
    print("Custom Visualization Demo")
    print("=" * 30)
    
    # Generate data
    trajectories, time_points = generate_lorenz_data(
        num_trajectories=20,
        t_max=8.0,
        dt=0.01,
        noise_scale=0.05
    )
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_lorenz_dataloaders(
        trajectories, time_points, 
        batch_size=8, 
        history_length=15, 
        future_length=25
    )
    
    # Create and train a simple UPN model
    trainer = DynamicalSystemTrainer(
        state_dim=3,
        hidden_dim=64,  # Smaller network for faster training
        use_diagonal_cov=True,
        learning_rate=1e-3
    ).to(device)
    
    # Quick training
    upn_model, _, _ = trainer.train_upn(
        train_loader, val_loader, num_epochs=3, lr=1e-3
    )
    
    # Get a sample for visualization
    history, history_time, future, future_time = next(iter(test_loader))
    
    # Take first example
    history = history[0:1].to(device)
    history_time = history_time[0:1].to(device) 
    future = future[0:1].to(device)
    future_time = future_time[0:1].to(device)
    
    # Make prediction
    initial_state = history[:, -1]
    initial_cov = torch.eye(3, device=device).unsqueeze(0) * 1e-4
    
    # Create time span
    last_time = history_time[0, -1].unsqueeze(0)
    t_span = torch.cat([last_time, future_time[0]], dim=0)
    
    # Predict
    pred_mean, pred_cov = upn_model.predict(initial_state, initial_cov, t_span)
    pred_mean = pred_mean[1:]  # Remove first point
    pred_cov = pred_cov[1:]
    pred_std = torch.sqrt(torch.diagonal(pred_cov, dim1=2, dim2=3))
    
    # Create custom visualization
    create_custom_lorenz_visualization(
        history, history_time, future, future_time,
        pred_mean, pred_std, save_path='custom_lorenz_viz.png'
    )
    
    print("Custom visualization demo completed!")

def create_custom_lorenz_visualization(history, history_time, future, future_time, 
                                     pred_mean, pred_std, save_path=None):
    """Create a comprehensive custom visualization"""
    
    # Convert to numpy
    history_np = history[0].cpu().numpy()
    history_time_np = history_time[0].cpu().numpy()
    future_np = future[0].cpu().numpy()
    future_time_np = future_time[0].cpu().numpy()
    pred_mean_np = pred_mean.permute(1, 0, 2)[0].cpu().numpy()
    pred_std_np = pred_std.permute(1, 0, 2)[0].cpu().numpy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D phase space plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot trajectories
    ax1.plot(history_np[:, 0], history_np[:, 1], history_np[:, 2], 
             'b-', linewidth=3, label='History', alpha=0.8)
    ax1.plot(future_np[:, 0], future_np[:, 1], future_np[:, 2], 
             'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(pred_mean_np[:, 0], pred_mean_np[:, 1], pred_mean_np[:, 2], 
             'r-', linewidth=2, label='UPN Prediction', alpha=0.8)
    
    # Add uncertainty spheres at select points
    for i in range(0, len(pred_mean_np), 8):
        u = np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        
        x = pred_mean_np[i, 0] + pred_std_np[i, 0] * np.outer(np.cos(u), np.sin(v))
        y = pred_mean_np[i, 1] + pred_std_np[i, 1] * np.outer(np.sin(u), np.sin(v))
        z = pred_mean_np[i, 2] + pred_std_np[i, 2] * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax1.plot_surface(x, y, z, color='red', alpha=0.1)
    
    ax1.set_title('3D Phase Space with Uncertainty')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Time series plots for each dimension
    dimensions = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i, (dim, color) in enumerate(zip(dimensions, colors)):
        ax = fig.add_subplot(2, 2, i+2)
        
        # Plot time series
        ax.plot(history_time_np, history_np[:, i], 'b-', linewidth=2, label='History')
        ax.plot(future_time_np, future_np[:, i], 'k-', linewidth=2, label='Ground Truth')
        ax.plot(future_time_np, pred_mean_np[:, i], color=color, linewidth=2, label='UPN Prediction')
        
        # Add uncertainty bands
        ax.fill_between(
            future_time_np,
            pred_mean_np[:, i] - 2*pred_std_np[:, i],
            pred_mean_np[:, i] + 2*pred_std_np[:, i],
            color=color, alpha=0.2, label='95% CI'
        )
        
        ax.set_title(f'{dim} Dimension')
        ax.set_xlabel('Time')
        ax.set_ylabel(dim)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved custom visualization to {save_path}")
    
    plt.show()
    return fig

def compare_prediction_horizons():
    """Compare UPN performance across different prediction horizons"""
    print("Comparing Prediction Horizons")
    print("=" * 35)
    
    # Generate data
    trajectories, time_points = generate_lorenz_data(
        num_trajectories=30,
        t_max=12.0,
        dt=0.01,
        noise_scale=0.08
    )
    
    horizons = [10, 20, 40]  # Different prediction horizons
    results = {}
    
    for horizon in horizons:
        print(f"\nTesting horizon: {horizon} steps")
        
        # Prepare data with specific horizon
        train_loader, val_loader, test_loader = prepare_lorenz_dataloaders(
            trajectories, time_points,
            batch_size=8,
            history_length=15,
            future_length=horizon
        )
        
        # Train model
        trainer = DynamicalSystemTrainer(
            state_dim=3,
            hidden_dim=64,
            use_diagonal_cov=True,
            learning_rate=1e-3
        ).to(device)
        
        upn_model, train_losses, val_losses = trainer.train_upn(
            train_loader, val_loader, num_epochs=3, lr=1e-3
        )
        
        # Store results
        results[horizon] = {
            'model': upn_model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1]
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot final validation losses
    plt.subplot(2, 2, 1)
    horizons_list = list(results.keys())
    final_losses = [results[h]['final_val_loss'] for h in horizons_list]
    plt.bar(horizons_list, final_losses)
    plt.title('Final Validation Loss vs Horizon')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Validation Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot training curves for each horizon
    for i, horizon in enumerate(horizons_list):
        plt.subplot(2, 2, i+2)
        plt.plot(results[horizon]['train_losses'], label='Train')
        plt.plot(results[horizon]['val_losses'], label='Validation')
        plt.title(f'Training Curves - Horizon {horizon}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('horizon_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Horizon comparison completed!")
    return results

def main():
    """Main function with different evaluation options"""
    print("UPN Lorenz System Evaluation")
    print("=" * 40)
    print("Choose evaluation type:")
    print("1. Quick Demo (fast, few epochs)")
    print("2. Full Evaluation (as in paper)")
    print("3. Custom Visualization Demo")
    print("4. Prediction Horizon Comparison")
    print("5. All of the above")
    
    try:
        choice = input("Enter choice (1-5) or press Enter for quick demo: ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"
    
    if choice == "1":
        metrics, viz_data = quick_demo()
    elif choice == "2":
        metrics, viz_data = full_evaluation()
    elif choice == "3":
        custom_visualization_demo()
    elif choice == "4":
        results = compare_prediction_horizons()
    elif choice == "5":
        print("\nRunning all evaluations...")
        print("\n1. Quick Demo:")
        quick_demo()
        print("\n2. Custom Visualization:")
        custom_visualization_demo()
        print("\n3. Horizon Comparison:")
        compare_prediction_horizons()
        print("\n4. Full Evaluation:")
        full_evaluation()
    else:
        print("Invalid choice, running quick demo...")
        quick_demo()
    
    print("\nAll evaluations completed!")
    print("Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()