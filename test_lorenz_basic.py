import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import numpy as np

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

print("Testing UPN Lorenz System...")
print("=" * 40)

# Test step by step with error handling
def test_imports():
    """Test all necessary imports"""
    try:
        print("1. Testing basic upn import...")
        import upn
        print("   ✓ upn module imported successfully")
        
        print("2. Testing Lorenz function imports...")
        from upn.applications.lorenz import generate_lorenz_data, prepare_lorenz_dataloaders
        print("   ✓ Lorenz functions imported successfully")
        
        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return False

def test_data_generation():
    """Test Lorenz data generation"""
    try:
        print("3. Testing data generation...")
        from upn.applications.lorenz import generate_lorenz_data
        
        data, time_points = generate_lorenz_data(
            num_trajectories=3,
            t_max=1.0,
            dt=0.1,
            noise_scale=0.05
        )
        
        print(f"   ✓ Data shape: {data.shape}")
        print(f"   ✓ Time points shape: {time_points.shape}")
        return True, data, time_points
    except Exception as e:
        print(f"   ✗ Data generation failed: {e}")
        return False, None, None

def test_data_loaders(data, time_points):
    """Test data loader creation"""
    try:
        print("4. Testing data loaders...")
        from upn.applications.lorenz import prepare_lorenz_dataloaders
        
        train_loader, val_loader, test_loader = prepare_lorenz_dataloaders(
            data, time_points,
            batch_size=2,
            history_length=3,
            future_length=5
        )
        
        print(f"   ✓ Created loaders with {len(train_loader)} train batches")
        
        # Test loading one batch
        history, history_time, future, future_time = next(iter(train_loader))
        print(f"   ✓ Batch loaded - History: {history.shape}, Future: {future.shape}")
        
        return True
    except Exception as e:
        print(f"   ✗ Data loader test failed: {e}")
        return False

def test_trainer():
    """Test trainer creation"""
    try:
        print("5. Testing trainer creation...")
        from upn.applications.dynamical import DynamicalSystemTrainer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        trainer = DynamicalSystemTrainer(
            state_dim=3,
            hidden_dim=16,  # Small for testing
            use_diagonal_cov=True,
            learning_rate=1e-3
        ).to(device)
        
        print("   ✓ Trainer created successfully")
        return True
    except Exception as e:
        print(f"   ✗ Trainer creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting UPN Lorenz tests...\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Check your __init__.py files.")
        return False
    
    # Test data generation
    success, data, time_points = test_data_generation()
    if not success:
        print("\n❌ Data generation test failed.")
        return False
    
    # Test data loaders
    if not test_data_loaders(data, time_points):
        print("\n❌ Data loader test failed.")
        return False
    
    # Test trainer
    if not test_trainer():
        print("\n❌ Trainer test failed.")
        return False
    
    print("\n🎉 All tests passed!")
    print("\nYou can now run the full Lorenz evaluation:")
    print("python examples/train_lorenz.py")
    
    return True

if __name__ == "__main__":
    main()