import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from upn import (
    DynamicalSystemUPN,
    plot_training_curves
)

from upn.applications.medical import (
    TimeSeriesUPN,
    create_synthetic_time_series,
    SyntheticTimeSeriesDataset,
    collate_fn,
    visualize_time_series_upn,
    evaluate_forecasting,
    visualize_forecasting
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    print("Creating synthetic medical time series data...")
    # Generate synthetic medical data
    synthetic_data = create_synthetic_time_series(
        num_patients=200, 
        num_features=10, 
        seq_length=48,
        missing_ratio=0.3
    )
    
    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(10)]
    
    # Split into train/val/test
    train_size = int(0.7 * len(synthetic_data['values']))
    val_size = int(0.15 * len(synthetic_data['values']))
    
    train_dataset = SyntheticTimeSeriesDataset(
        synthetic_data['values'][:train_size],
        synthetic_data['times'][:train_size],
        synthetic_data['masks'][:train_size],
        synthetic_data['outcomes'][:train_size]
    )
    
    val_dataset = SyntheticTimeSeriesDataset(
        synthetic_data['values'][train_size:train_size+val_size],
        synthetic_data['times'][train_size:train_size+val_size],
        synthetic_data['masks'][train_size:train_size+val_size],
        synthetic_data['outcomes'][train_size:train_size+val_size]
    )
    
    test_dataset = SyntheticTimeSeriesDataset(
        synthetic_data['values'][train_size+val_size:],
        synthetic_data['times'][train_size+val_size:],
        synthetic_data['masks'][train_size+val_size:],
        synthetic_data['outcomes'][train_size+val_size:]
    )
    
    print(f"Train: {len(train_dataset)} patients")
    print(f"Validation: {len(val_dataset)} patients")
    print(f"Test: {len(test_dataset)} patients")
    
    # Create data loaders with custom collate function for variable length sequences
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Define model parameters
    input_dim = 10  # Number of features
    latent_dim = 8  # Dimension of latent space
    hidden_dim = 64  # Dimension of hidden layers
    
    # Create and train time series UPN model
    print("Training TimeSeriesUPN model...")
    model = TimeSeriesUPN(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        use_diagonal_cov=True
    ).to(device)
    
    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    
    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        for values, times, masks, _ in train_loader:
            # Move to device
            values = values.to(device)
            times = times.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Compute loss
            loss_dict = model.compute_loss(values, times, masks)
            
            # Update model
            loss = loss_dict['total_loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict['recon_loss'].item()
            epoch_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
        
        # Average losses
        epoch_loss /= num_batches
        epoch_recon_loss /= num_batches
        epoch_kl_loss /= num_batches
        
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for values, times, masks, _ in val_loader:
                # Move to device
                values = values.to(device)
                times = times.to(device)
                masks = masks.to(device)
                
                # Compute loss
                loss_dict = model.compute_loss(values, times, masks)
                
                # Accumulate losses
                val_loss += loss_dict['total_loss'].item()
                val_recon_loss += loss_dict['recon_loss'].item()
                val_kl_loss += loss_dict['kl_loss'].item()
                num_val_batches += 1
        
        # Average losses
        val_loss /= num_val_batches
        val_recon_loss /= num_val_batches
        val_kl_loss /= num_val_batches
        
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {epoch_loss:.6f} (Recon: {epoch_recon_loss:.6f}, KL: {epoch_kl_loss:.6f})")
        print(f"  Val Loss: {val_loss:.6f} (Recon: {val_recon_loss:.6f}, KL: {val_kl_loss:.6f})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  New best model with validation loss: {best_val_loss:.6f}")
            
        # Early stopping
        if epoch >= 20 and all(val_losses[-i-1] >= val_losses[-i-2] for i in range(10)):
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Plot training curves
    plot_training_curves(
        train_losses,
        val_losses,
        title='TimeSeriesUPN Training on Medical Data',
        save_path='medical_training_curves.png'
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
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
        for values, times, masks, _ in test_loader:
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
    
    # Visualize reconstruction results
    visualize_time_series_upn(
        eval_results, feature_names, num_patients=3, num_features=5,
        save_path='medical_reconstruction_results.png'
    )
    
    # Evaluate forecasting
    forecast_horizon = 12  # Forecast 12 time steps ahead
    print(f"\nEvaluating forecasting performance (horizon={forecast_horizon})...")
    
    forecast_metrics = evaluate_forecasting(
        model, test_loader, forecast_horizon=forecast_horizon, device=device
    )
    
    # Visualize forecasting
    visualize_forecasting(
        model, test_loader, forecast_horizon=forecast_horizon, 
        num_patients=3, num_features=3, feature_names=feature_names,
        save_path='medical_forecasting_results.png', device=device
    )
    
    print("Done! Check the output images for results.")

if __name__ == "__main__":
    main()