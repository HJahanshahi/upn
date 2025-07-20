import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from ..core.upn_base import UPNBase
from ..core.dynamics import DynamicsNetwork
from ..core.noise import ProcessNoiseNetwork

class DynamicalSystemDataset(Dataset):
    """Dataset for dynamical system trajectories"""
    def __init__(self, trajectories, time_points, history_length=10, future_length=20):
        """
        Initialize dataset for trajectory prediction.
        
        Args:
            trajectories: Array of trajectories [num_traj, time_steps, state_dim]
            time_points: Time points [time_steps]
            history_length: Length of history window
            future_length: Length of future prediction window
        """
        self.trajectories = torch.tensor(trajectories, dtype=torch.float32)
        self.time_points = torch.tensor(time_points, dtype=torch.float32)
        self.history_length = history_length
        self.future_length = future_length
        
    def __len__(self):
        return len(self.trajectories) * (len(self.time_points) - self.history_length - self.future_length + 1)
    
    def __getitem__(self, idx):
        # Convert idx to trajectory idx and start idx
        traj_idx = idx // (len(self.time_points) - self.history_length - self.future_length + 1)
        start_idx = idx % (len(self.time_points) - self.history_length - self.future_length + 1)
        
        # Get history and future indices
        history_indices = slice(start_idx, start_idx + self.history_length)
        future_indices = slice(start_idx + self.history_length, 
                              start_idx + self.history_length + self.future_length)
        
        # Get data
        history = self.trajectories[traj_idx, history_indices]
        future = self.trajectories[traj_idx, future_indices]
        
        # Get time points
        history_time = self.time_points[history_indices]
        future_time = self.time_points[future_indices]
        
        return history, history_time, future, future_time

class UPNDynamicalSystem(nn.Module):
    """
    UPN implementation specifically for dynamical systems modeling.
    Inherits from UPNBase and adds dynamical system specific functionality.
    """
    def __init__(self, state_dim, hidden_dim=64, use_diagonal_cov=True, adjoint=False):
        super().__init__()
        
        # Create the base UPN model
        self.upn = UPNBase(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            use_diagonal_cov=use_diagonal_cov,
            adjoint=adjoint
        )
        
        self.state_dim = state_dim
        self.use_diagonal_cov = use_diagonal_cov
    
    def forward(self, t, states):
        """Forward pass through UPN dynamics"""
        return self.upn(t, states)
    
    def predict(self, initial_state, initial_cov, t_span):
        """Make predictions with uncertainty"""
        return self.upn.predict(initial_state, initial_cov, t_span)
    
    def compute_loss(self, true_traj, pred_mean, pred_cov):
        """Compute loss for training"""
        return self.upn.compute_loss(true_traj, pred_mean, pred_cov)

class DeterministicODENet(nn.Module):
    """Deterministic ODE network for comparison"""
    def __init__(self, state_dim, hidden_dim=64, adjoint=False):
        super().__init__()
        self.state_dim = state_dim
        self.adjoint = adjoint
        
        # Neural network for the dynamics
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, t, x):
        # Handle scalar time input
        batch_size = x.shape[0]
        
        # Convert scalar time to tensor if needed
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=x.device)
            
        # Ensure time has proper shape [batch_size, 1]
        if t.dim() == 0:
            t = t.expand(batch_size).unsqueeze(-1)
        elif t.dim() == 1 and t.numel() == 1:
            t = t.expand(batch_size).unsqueeze(-1)
        elif t.dim() == 1 and t.numel() == batch_size:
            t = t.unsqueeze(-1)
            
        # Concatenate time and state
        tx = torch.cat([t, x], dim=1)
        return self.net(tx)
    
    def integrate(self, z0, t):
        """Integrate the ODE dynamics from initial state z0 over time points t"""
        from torchdiffeq import odeint, odeint_adjoint
        
        if self.adjoint:
            return odeint_adjoint(self, z0, t, method='dopri5')
        else:
            return odeint(self, z0, t, method='dopri5')
    
    def predict(self, initial_state, t_span):
        """Make deterministic predictions"""
        # Integrate forward
        z_all = self.integrate(initial_state, t_span)
        return z_all

class EnsembleODENet(nn.Module):
    """Ensemble of ODENets for uncertainty estimation"""
    def __init__(self, state_dim, hidden_dim=64, num_models=5, adjoint=False):
        super().__init__()
        self.state_dim = state_dim
        self.num_models = num_models
        self.adjoint = adjoint
        
        # Create an ensemble of models
        self.models = nn.ModuleList([
            DeterministicODENet(state_dim, hidden_dim, adjoint) for _ in range(num_models)
        ])
        
    def predict(self, initial_state, t_span):
        """Make ensemble predictions with uncertainty estimation"""
        batch_size = initial_state.shape[0]
        time_steps = len(t_span)
        
        # Predictions from each model
        preds = []
        
        for model in self.models:
            pred = model.predict(initial_state, t_span)
            preds.append(pred)
            
        # Stack predictions
        preds = torch.stack(preds, dim=0)  # [num_models, time_steps, batch_size, state_dim]
        
        # Compute mean and covariance across the ensemble
        mean_pred = preds.mean(dim=0)  # [time_steps, batch_size, state_dim]
        
        # Compute covariance
        cov_pred = torch.zeros(time_steps, batch_size, self.state_dim, self.state_dim, 
                              device=initial_state.device)
        
        for t in range(time_steps):
            for b in range(batch_size):
                # Get predictions for this time step and batch element
                ensemble_preds = preds[:, t, b, :]  # [num_models, state_dim]
                
                # Compute empirical covariance
                centered = ensemble_preds - mean_pred[t, b]  # [num_models, state_dim]
                cov = torch.mm(centered.T, centered) / (self.num_models - 1)  # [state_dim, state_dim]
                cov_pred[t, b] = cov + 1e-6 * torch.eye(self.state_dim, device=initial_state.device)
                
        return mean_pred, cov_pred

class DynamicalSystemTrainer:
    """
    Trainer class for dynamical systems with UPN and baseline models.
    """
    def __init__(self, state_dim, hidden_dim=64, use_diagonal_cov=True, 
                 adjoint=False, learning_rate=1e-3):
        """
        Initialize trainer for dynamical systems.
        
        Args:
            state_dim: Dimension of the state vector
            hidden_dim: Dimension of hidden layers in networks
            use_diagonal_cov: Whether to use diagonal approximation for covariance
            adjoint: Whether to use adjoint method for backpropagation
            learning_rate: Learning rate for optimizer
        """
        self.state_dim = state_dim
        self.use_diagonal_cov = use_diagonal_cov
        self.learning_rate = learning_rate
        
        # Device (set later)
        self.device = torch.device('cpu')
        
        # Models will be created when needed
        self.upn_model = None
        self.det_model = None
        self.ensemble_model = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def to(self, device):
        """Move to specified device"""
        self.device = device
        return self
    
    def create_upn_model(self):
        """Create UPN model"""
        self.upn_model = UPNDynamicalSystem(
            state_dim=self.state_dim,
            hidden_dim=64,
            use_diagonal_cov=self.use_diagonal_cov,
            adjoint=False
        ).to(self.device)
        return self.upn_model
    
    def create_deterministic_model(self):
        """Create deterministic baseline model"""
        self.det_model = DeterministicODENet(
            state_dim=self.state_dim,
            hidden_dim=64,
            adjoint=False
        ).to(self.device)
        return self.det_model
    
    def create_ensemble_model(self, num_models=5):
        """Create ensemble baseline model"""
        self.ensemble_model = EnsembleODENet(
            state_dim=self.state_dim,
            hidden_dim=64,
            num_models=num_models,
            adjoint=False
        ).to(self.device)
        return self.ensemble_model
    
    def train_upn(self, train_loader, val_loader, num_epochs=25, lr=1e-3):
        """Train UPN model"""
        if self.upn_model is None:
            self.create_upn_model()
        
        # Optimizer
        optimizer = optim.Adam(self.upn_model.parameters(), lr=lr)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            self.upn_model.train()
            epoch_loss = 0.0
            
            for history, history_time, future, future_time in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                history = history.to(self.device)
                history_time = history_time.to(self.device)
                future = future.to(self.device)
                future_time = future_time.to(self.device)
                
                optimizer.zero_grad()
                
                # Get initial state and covariance from last point in history
                initial_state = history[:, -1]
                
                # Initial covariance (assumed to be small at the start)
                initial_cov = torch.eye(self.state_dim, device=self.device).unsqueeze(0).expand(history.shape[0], -1, -1) * 1e-4
                
                # Get the last time point from history for each batch element
                last_time = history_time[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]
                
                # Create the full time span for ODE integration
                t_span = torch.cat([last_time[0], future_time[0]], dim=0)
                
                # Make predictions
                mean_pred, cov_pred = self.upn_model.predict(initial_state, initial_cov, t_span)
                
                # Remove the first prediction (corresponds to the last history point)
                mean_pred = mean_pred[1:]
                cov_pred = cov_pred[1:]
                
                # Transpose mean_pred to match future shape [batch_size, time_steps, state_dim]
                mean_pred = mean_pred.permute(1, 0, 2)
                cov_pred = cov_pred.permute(1, 0, 2, 3)
                
                # Compute loss
                loss = self.upn_model.compute_loss(future, mean_pred, cov_pred)
                
                # Update model
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.upn_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item() * history.shape[0]
            
            train_loss = epoch_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._evaluate_upn(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.upn_model.state_dict().copy()
                print(f"New best model with validation loss: {best_val_loss:.6f}")
        
        # Load best model
        self.upn_model.load_state_dict(best_model_state)
        
        return self.upn_model, train_losses, val_losses
    
    def _evaluate_upn(self, val_loader):
        """Evaluate UPN model on validation set"""
        self.upn_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for history, history_time, future, future_time in val_loader:
                history = history.to(self.device)
                history_time = history_time.to(self.device)
                future = future.to(self.device)
                future_time = future_time.to(self.device)
                
                # Get initial state and covariance from last point in history
                initial_state = history[:, -1]
                initial_cov = torch.eye(self.state_dim, device=self.device).unsqueeze(0).expand(history.shape[0], -1, -1) * 1e-4
                
                # Get the last time point from history
                last_time = history_time[:, -1].unsqueeze(1)
                t_span = torch.cat([last_time[0], future_time[0]], dim=0)
                
                # Make predictions
                mean_pred, cov_pred = self.upn_model.predict(initial_state, initial_cov, t_span)
                
                # Remove the first prediction
                mean_pred = mean_pred[1:]
                cov_pred = cov_pred[1:]
                
                # Transpose to match future shape
                mean_pred = mean_pred.permute(1, 0, 2)
                cov_pred = cov_pred.permute(1, 0, 2, 3)
                
                # Compute loss
                loss = self.upn_model.compute_loss(future, mean_pred, cov_pred)
                val_loss += loss.item() * history.shape[0]
        
        return val_loss / len(val_loader.dataset)
    
    def train_ensemble(self, train_loader, val_loader, num_models=5, num_epochs=25, lr=1e-3):
        """Train ensemble model"""
        if self.ensemble_model is None:
            self.create_ensemble_model(num_models)
        
        # Optimizer
        optimizer = optim.Adam(self.ensemble_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        # MSE loss for deterministic training
        mse_loss = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # Training
            self.ensemble_model.train()
            epoch_loss = 0.0
            
            for history, history_time, future, future_time in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                history = history.to(self.device)
                history_time = history_time.to(self.device)
                future = future.to(self.device)
                future_time = future_time.to(self.device)
                
                optimizer.zero_grad()
                
                # Total loss across all models in the ensemble
                total_loss = 0.0
                
                # Train each model in the ensemble independently
                for i, submodel in enumerate(self.ensemble_model.models):
                    # Get initial state from last point in history
                    initial_state = history[:, -1]
                    
                    # Get the last time point from history
                    last_time = history_time[:, -1].unsqueeze(1)
                    t_span = torch.cat([last_time[0], future_time[0]], dim=0)
                    
                    # Make predictions
                    predictions = submodel.predict(initial_state, t_span)
                    
                    # Remove the first prediction
                    predictions = predictions[1:]
                    
                    # Transpose predictions to match shape of future
                    predictions = predictions.permute(1, 0, 2)
                    
                    # Compute loss
                    loss = mse_loss(predictions, future)
                    total_loss += loss
                    
                # Average loss across ensemble
                avg_loss = total_loss / len(self.ensemble_model.models)
                
                # Update model
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ensemble_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += avg_loss.item() * history.shape[0]
            
            train_loss = epoch_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._evaluate_ensemble(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {name: param.clone() for name, param in self.ensemble_model.state_dict().items()}
                print(f"New best model with validation loss: {best_val_loss:.6f}")
        
        # Load best model
        self.ensemble_model.load_state_dict(best_model_state)
        
        return self.ensemble_model, train_losses, val_losses
    
    def _evaluate_ensemble(self, val_loader):
        """Evaluate ensemble model on validation set"""
        self.ensemble_model.eval()
        val_loss = 0.0
        mse_loss = nn.MSELoss()
        
        with torch.no_grad():
            for history, history_time, future, future_time in val_loader:
                history = history.to(self.device)
                history_time = history_time.to(self.device)
                future = future.to(self.device)
                future_time = future_time.to(self.device)
                
                # Get initial state from last point in history
                initial_state = history[:, -1]
                
                # Get the last time point from history
                last_time = history_time[:, -1].unsqueeze(1)
                t_span = torch.cat([last_time[0], future_time[0]], dim=0)
                
                # Make ensemble predictions
                mean_pred, _ = self.ensemble_model.predict(initial_state, t_span)
                
                # Remove the first prediction
                mean_pred = mean_pred[1:]
                
                # Transpose mean_pred to match future shape
                mean_pred = mean_pred.permute(1, 0, 2)
                
                # Compute MSE loss using the mean prediction
                loss = mse_loss(mean_pred, future)
                val_loss += loss.item() * history.shape[0]
        
        return val_loss / len(val_loader.dataset)
    
    def train_deterministic(self, train_loader, val_loader, num_epochs=25, lr=1e-3):
        """Train deterministic model"""
        if self.det_model is None:
            self.create_deterministic_model()
        
        # Optimizer
        optimizer = optim.Adam(self.det_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
        
        # MSE loss
        mse_loss = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            self.det_model.train()
            epoch_loss = 0.0
            
            for history, history_time, future, future_time in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                history = history.to(self.device)
                history_time = history_time.to(self.device)
                future = future.to(self.device)
                future_time = future_time.to(self.device)
                
                optimizer.zero_grad()
                
                # Get initial state from last point in history
                initial_state = history[:, -1]
                
                # Get the last time point from history
                last_time = history_time[:, -1].unsqueeze(1)
                t_span = torch.cat([last_time[0], future_time[0]], dim=0)
                
                # Make predictions
                predictions = self.det_model.predict(initial_state, t_span)
                
                # Remove the first prediction
                predictions = predictions[1:]
                
                # Transpose predictions to match the shape of future
                predictions = predictions.permute(1, 0, 2)
                
                # Compute loss
                loss = mse_loss(predictions, future)
                
                # Update model
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.det_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item() * history.shape[0]
            
            train_loss = epoch_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._evaluate_deterministic(val_loader)
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.det_model.state_dict().copy()
                print(f"New best model with validation loss: {best_val_loss:.6f}")
        
        # Load best model
        self.det_model.load_state_dict(best_model_state)
        
        return self.det_model, train_losses, val_losses
    
    def _evaluate_deterministic(self, val_loader):
        """Evaluate deterministic model on validation set"""
        self.det_model.eval()
        val_loss = 0.0
        mse_loss = nn.MSELoss()
        
        with torch.no_grad():
            for history, history_time, future, future_time in val_loader:
                history = history.to(self.device)
                history_time = history_time.to(self.device)
                future = future.to(self.device)
                future_time = future_time.to(self.device)
                
                # Get initial state from last point in history
                initial_state = history[:, -1]
                
                # Get the last time point from history
                last_time = history_time[:, -1].unsqueeze(1)
                t_span = torch.cat([last_time[0], future_time[0]], dim=0)
                
                # Make predictions
                predictions = self.det_model.predict(initial_state, t_span)
                
                # Remove the first prediction
                predictions = predictions[1:]
                
                # Transpose predictions to match the shape of future
                predictions = predictions.permute(1, 0, 2)
                
                # Compute loss
                loss = mse_loss(predictions, future)
                val_loss += loss.item() * history.shape[0]
        
        return val_loss / len(val_loader.dataset)