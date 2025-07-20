import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from upn import (
    plot_training_curves,
    UPNFlow,
    generate_flow_data,
    train_upn_flow,
    visualize_flow_density,
    visualize_flow_samples,
    visualize_flow_transformation
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    # Choose dataset type
    dataset_type = "moons"  # Options: "moons", "circles", "blobs", "swiss_roll"
    n_samples = 2000
    noise = 0.1
    
    # Generate dataset
    print(f"Generating {dataset_type} dataset...")
    flow_data = generate_flow_data(dataset_type, n_samples=n_samples, noise=noise)
    flow_data = flow_data.to(device)
    print(f"Generated {flow_data.shape[0]} samples with dimension {flow_data.shape[1]}")
    
    # Plot dataset
    plt.figure(figsize=(8, 8))
    plt.scatter(flow_data[:, 0].cpu().numpy(), flow_data[:, 1].cpu().numpy(), 
               s=10, alpha=0.5)
    plt.title(f"{dataset_type.capitalize()} Dataset")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(alpha=0.3)
    plt.savefig(f'{dataset_type}_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Train UPN Flow model
    print(f"\nTraining UPN Flow model on {dataset_type} dataset...")
    hidden_dim = 64
    flow_model, train_losses, val_losses = train_upn_flow(
        flow_data, 
        val_split=0.2,
        batch_size=128, 
        hidden_dim=hidden_dim, 
        num_epochs=300,  # Adjust based on dataset complexity
        lr=1e-3,
        device=device
    )
    
    # Plot training curves
    plot_training_curves(
        train_losses, 
        val_losses, 
        title=f'UPN Flow Training on {dataset_type.capitalize()} Dataset',
        save_path=f'{dataset_type}_flow_training.png'
    )
    
    # Visualize learned density
    visualize_flow_density(
        flow_model, 
        flow_data, 
        title=f"UPN Flow Density on {dataset_type.capitalize()} Dataset",
        save_path=f'{dataset_type}_flow_density.png'
    )
    
    # Visualize samples
    visualize_flow_samples(
        flow_model, 
        flow_data, 
        title=f"UPN Flow Samples with Uncertainty on {dataset_type.capitalize()} Dataset",
        save_path=f'{dataset_type}_flow_samples.png'
    )
    
    # Visualize transformation
    visualize_flow_transformation(
        flow_model,
        title=f"UPN Flow Transformation on {dataset_type.capitalize()} Dataset",
        save_path=f'{dataset_type}_flow_transformation.png'
    )
    
    # Save model
    torch.save({
        'model_state_dict': flow_model.state_dict(),
    }, f'upn_flow_{dataset_type}_model.pt')
    
    print(f"\nExperiment completed. Results saved to {dataset_type}_*.png")

if __name__ == "__main__":
    main()