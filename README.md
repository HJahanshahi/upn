# UPN: Uncertainty Propagation Networks

A PyTorch library for modeling uncertainty in continuous-time dynamical systems using coupled mean and covariance ODEs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

---

## 📘 Overview

Uncertainty Propagation Networks (UPNs) are neural architectures that **simultaneously model both state evolution and uncertainty propagation** in continuous-time dynamical systems. Unlike standard Neural ODEs that provide only point estimates, UPNs solve coupled differential equations for mean and covariance dynamics, enabling principled uncertainty quantification.

### Key Innovation
- **Coupled ODEs**: Tracks both μ(t) and Σ(t) evolution  
- **State-dependent noise**: Learns Q(μ,t) as a neural network  
- **Continuous-time**: Handles irregular sampling naturally  
- **Theoretical foundation**: Grounded in stochastic differential equations

---

## ✨ Key Features

- **🎯 Principled uncertainty quantification** — Well-calibrated confidence intervals  
- **⏰ Continuous-time formulation** — No discretization artifacts  
- **🔧 State-dependent process noise** — Adaptive uncertainty based on system state  
- **🏗️ Modular architecture** — Easy to extend and customize  
- **📊 Comprehensive evaluation** — MSE, NLL, CRPS, interval coverage metrics  
- **🎨 Rich visualizations** — 2D/3D plots with uncertainty bands

---

## 📦 Installation

### Quick Install
```bash
git clone https://github.com/HJahanshahi/upn.git
cd upn
pip install -e .
```

### Dependencies  
Automatically installed with the package:
- `torch` (≥1.9) — Deep learning framework  
- `torchdiffeq` — Differentiable ODE solvers  
- `numpy` — Numerical computing  
- `scipy` — Scientific computing  
- `matplotlib` — Plotting and visualization  
- `tqdm` — Progress bars

### Verify Installation
```python
import upn
from upn.applications.lorenz import evaluate_lorenz_system
print("✅ UPN library installed successfully!")
```

---

## ⚡ Quick Start

### Basic UPN Usage
```python
import torch
from upn import UPNBase

# Create UPN model
model = UPNBase(
    state_dim=3,           # System dimension
    hidden_dim=64,         # Network size
    use_diagonal_cov=True  # Diagonal covariance approximation
)

# Initial conditions
batch_size = 16
initial_state = torch.randn(batch_size, 3)
initial_cov = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1) * 0.1
t_span = torch.linspace(0, 5.0, 50)

# Predict with uncertainty
mean_pred, cov_pred = model.predict(initial_state, initial_cov, t_span)
print(f"Predictions: {mean_pred.shape}, Uncertainty: {cov_pred.shape}")
```

### Lorenz Attractor (Reproducing Paper Results)
```python
from upn.applications.lorenz import evaluate_lorenz_system

# Complete evaluation as in the paper
metrics, viz_data = evaluate_lorenz_system()
# Creates: training curves, 3D phase plots, 2D time series, saved models
```

---

## 🧪 Examples & Experiments

### Run these examples to reproduce paper results:

#### 1. Lorenz Attractor (Section 5.1.2 in paper)
```bash
python examples/train_lorenz.py
```
**Demonstrates**: Chaotic system modeling, uncertainty in phase space, ensemble comparison  
**Outputs**:
- `lorenz_training_curves.png` — Training progress  
- `lorenz_3d_predictions.png` — 3D phase space with uncertainty  
- `lorenz_2d_predictions.png` — Time series predictions  
- Model checkpoints and evaluation metrics  

#### 2. Interactive Demo
```bash
python examples/train_lorenz.py
# Choose: 1=Quick demo, 2=Full evaluation, 3=Custom viz, 4=Horizon comparison
```

#### 3. Custom Systems
```python
from upn.applications.dynamical import DynamicalSystemTrainer

# Train on your own dynamical system
trainer = DynamicalSystemTrainer(state_dim=2, hidden_dim=64)
upn_model, _, _ = trainer.train_upn(your_train_loader, your_val_loader)
```

---

## 🏗️ Library Architecture

```
upn/
├── core/                   # Core UPN components
│   ├── upn_base.py         # Main UPN class
│   ├── dynamics.py         # Neural dynamics f(μ,t)
│   ├── noise.py            # Process noise Q(μ,t)
│   └── ode_solver.py       # Integration utilities
├── applications/           # Domain-specific implementations
│   ├── dynamical.py        # Dynamical systems framework
│   └── lorenz.py           # Lorenz attractor implementation
└── examples/               # Complete examples
    └── train_lorenz.py     # Main Lorenz experiment
```

### Core Classes
- `UPNBase`: Main UPN implementation with coupled ODEs  
- `DynamicsNetwork`: Neural network for drift function f(μ,t)  
- `ProcessNoiseNetwork`: Neural network for noise Q(μ,t)  
- `DynamicalSystemTrainer`: High-level trainer with baselines

---

## 📊 Evaluation Metrics

UPN provides comprehensive uncertainty quantification metrics:
```python
from upn.applications.lorenz import evaluate_models

metrics, viz_data = evaluate_models(upn_model, det_model, ensemble_model, test_loader)
```

**Available metrics**:
- MSE: Mean squared error  
- NLL: Negative log-likelihood  
- CRPS: Continuous ranked probability score  
- Coverage: 95% confidence interval coverage  

**Expected Results (Lorenz System)**:
- UPN Coverage: ~0.95 (near-perfect calibration)  
- Ensemble Coverage: ~0.1–0.2 (severe underestimation)  
- UPN NLL: Significantly lower than baselines

---

## 🎨 Visualizations

### Built-in Plotting Functions
```python
from upn.applications.lorenz import (
    visualize_lorenz_3d,      # 3D phase space with uncertainty
    visualize_lorenz_2d,      # Time series predictions  
    plot_training_curves      # Training progress
)

# Create publication-quality plots
visualize_lorenz_3d(viz_data, save_path='phase_space.png')
visualize_lorenz_2d(viz_data, save_path='time_series.png')
```

### Custom Visualizations
```python
import matplotlib.pyplot as plt

# Extract predictions
history = viz_data['history'][0]
future = viz_data['future'][0] 
upn_mean = viz_data['upn_mean'][0]
upn_std = viz_data['upn_std'][0]

# Plot with uncertainty bands
plt.plot(time_points, upn_mean, 'r-', label='UPN Mean')
plt.fill_between(time_points, 
                 upn_mean - 2*upn_std, 
                 upn_mean + 2*upn_std,
                 alpha=0.2, label='95% CI')
plt.legend()
```

---

## 🔬 Extending the Library

### Adding New Dynamical Systems
```python
def my_system(t, state):
    # Your ODE: dx/dt = f(x,t)
    return dydt
```

```python
from scipy.integrate import solve_ivp

def generate_my_data(num_trajectories=50):
    # Integrate your system and add noise
    return trajectories, time_points
```

```python
from upn.applications.dynamical import DynamicalSystemTrainer

trainer = DynamicalSystemTrainer(state_dim=your_dim)
model = trainer.train_upn(train_loader, val_loader)
```

### Custom Loss Functions
```python
class CustomUPN(UPNBase):
    def compute_loss(self, true_traj, pred_mean, pred_cov):
        nll = super().compute_loss(true_traj, pred_mean, pred_cov)
        # Add your custom terms
        return nll + custom_regularization
```

---

## ⚙️ Performance & Troubleshooting

### Memory Optimization
- Use `use_diagonal_cov=True` for large systems  
- Set `adjoint=True` for memory-efficient gradients  
- Reduce `batch_size` if encountering OOM

### GPU Acceleration
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'upn'`  
> 💡 Solution: Install in development mode  
```bash
pip install -e .
```

**Training Instability**: NaN losses or poor convergence  
> 💡 Solutions:
- Reduce learning rate (try 1e-4 instead of 1e-3)  
- Use gradient clipping  
- Check initial covariance values (`1e-4 * torch.eye()`)

**Poor Calibration**: Coverage far from 0.95  
> 💡 Solutions:
- Train longer (more epochs)  
- Tune process noise network architecture  
- Validate hyperparameters on held-out data

---

## 📈 Performance Benchmarks

**Lorenz System** (100 trajectories, 25 epochs):

- Training time: ~5 min (GPU), ~15 min (CPU)  
- Memory usage: ~2GB (full covariance), ~500MB (diagonal)  
- 95% Coverage: UPN ≈ 0.95, Ensemble ≈ 0.15

---

## 🤝 Contributing

We welcome contributions! Please:
- Fork the repository  
- Create a feature branch: `git checkout -b feature-name`  
- Follow PEP8 style guidelines  
- Add tests for new functionality  
- Submit a pull request

### Development Setup
```bash
git clone https://github.com/HJahanshahi/upn.git
cd upn
pip install -e ".[dev]"  # Includes development dependencies
```

---

## 📝 Citation

If you use this library in your research, please cite:
```bibtex
@article{jahanshahi2025uncertainty,
  title={Uncertainty Propagation Networks for Neural Ordinary Differential Equations},
  author={Jahanshahi, Hadi and Zhu, Zheng H},
  journal={arXiv preprint arXiv:2508.16815},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

**Authors**: Hadi Jahanshahi, Zheng H. Zhu (York University)  
**Dependencies**: PyTorch team, `torchdiffeq` contributors  
**Inspiration**: Neural ODE and stochastic process literature

---

## 🔗 Related Work

- **Neural ODEs** — Foundational continuous-time neural networks  
- **PyTorch** — Deep learning framework  
- **SciPy** — Scientific computing ecosystem

---

⭐ Star this repository if you find it useful!  
For questions, issues, or contributions, please visit our [GitHub Issues](https://github.com/HJahanshahi/upn/issues) page.
