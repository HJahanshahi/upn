# UPN: Uncertainty Propagation Networks for Neural Ordinary Differential Equations

A PyTorch library for continuous-time modeling with principled uncertainty quantification via coupled mean–covariance ODEs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

---

## Overview

Uncertainty Propagation Networks (UPNs) **simultaneously model both state evolution and uncertainty propagation** in continuous-time dynamical systems. Unlike standard Neural ODEs that provide only point estimates, UPNs solve coupled differential equations for mean and covariance dynamics, providing calibrated confidence intervals in a single forward pass.

**Key features:**
- Coupled ODEs for mean μ(t) and covariance Σ(t) evolution (Eqs. 2–3)
- State-dependent process noise Q(μ,t) via Cholesky parameterisation (Eq. 5)
- Efficient half-vectorization (vech) for symmetric covariance matrices (Eq. 6)
- Integrated Kalman measurement updates for sparse observations (Eqs. 11–13)
- Two modes: pure prediction and adaptive filtering
- 6–14× faster inference than sampling-based alternatives (Neural SDEs, ensembles)

**Reference:**
> Jahanshahi, H. & Zhu, Z.H. (2026). Uncertainty Propagation Networks for Neural Ordinary Differential Equations. *Neurocomputing*, 677, 133134. https://doi.org/10.1016/j.neucom.2026.133134

---

## Installation

```bash
git clone https://github.com/HJahanshahi/upn.git
cd upn
pip install -e .
```

### Verify

```python
from upn import UPN
import torch

model = UPN(state_dim=6, obs_dim=3, hidden_dim=32)
mu0 = torch.randn(4, 6)
S0 = torch.eye(6).unsqueeze(0).expand(4, -1, -1) * 0.01
t = torch.linspace(0, 1, 10)

mean_traj, cov_traj = model.integrate(mu0, S0, t)
print(f"Mean: {mean_traj.shape}, Covariance: {cov_traj.shape}")
# Mean: torch.Size([10, 4, 6]), Covariance: torch.Size([10, 4, 6, 6])
```

---

## Library Structure

```
upn/
├── core/
│   ├── dynamics.py      # DynamicsNetwork f_θ(μ, t)         — Eq. 2
│   ├── noise.py         # ProcessNoiseNetwork Q_φ(μ, t)     — Eq. 5
│   ├── vech.py          # Half-vectorization operators       — Eq. 6
│   └── upn.py           # UPN model (coupled ODEs + Kalman)  — Eqs. 2,3,8,11-13
├── baselines/
│   └── __init__.py      # LatentODE, EnsembleNeuralODE, NeuralSDE
data/
└── cubesat/             # CubeSat experimental trajectory data
examples/
├── cubesat/
│   ├── train_cubesat.py      # Train all models (Section 5.3)
│   └── evaluate_cubesat.py   # Evaluate & generate figures (Table 5, Figs. 27-31)
├── dynamical_systems/
│   ├── train_dynamical.py    # 4 non-chaotic systems (Section 5.1, Table 3)
│   └── plot_results.py       # Visualization for non-chaotic results
└── lorenz/
    └── train_lorenz.py       # Chaotic Lorenz attractor (Section 5.2, Table 4)
```

---

## Quick Start

### Basic UPN Usage

```python
from upn import UPN
import torch

# Create model
model = UPN(
    state_dim=6,      # Full Markovian state dimension
    obs_dim=3,        # Observation dimension  
    hidden_dim=32,    # Network hidden units
)

# Initial conditions
batch_size = 16
mu0 = torch.randn(batch_size, 6)
S0 = torch.eye(6).unsqueeze(0).expand(batch_size, -1, -1) * 0.01
t = torch.linspace(0, 5.0, 50)

# Pure prediction (Scenario 1)
mean_pred, cov_pred = model.integrate(mu0, S0, t)

# With Kalman updates (Scenario 2)
observations = torch.randn(batch_size, 49, 3)  # sparse observations
mean_pred, cov_pred = model.integrate(
    mu0, S0, t,
    observations=observations,
    update_frequency=5,  # update every 5 steps
)

# Compute loss
loss = model.compute_nll_loss(observations, mean_pred[1:], cov_pred[1:])
```

---

## Reproducing Paper Results

### CubeSat Trajectory Prediction (Section 5.3)

**Train all models:**
```bash
python examples/cubesat/train_cubesat.py
```

**Evaluate and generate figures:**
```bash
python examples/cubesat/evaluate_cubesat.py
```

### Non-Chaotic Dynamical Systems (Section 5.1)

Four canonical systems: Damped Harmonic Oscillator, Van der Pol, Linear 2D, Damped Pendulum.

```bash
python examples/dynamical_systems/train_dynamical.py
```

**Expected results (Table 3):** UPN achieves ~96.7% average coverage vs Ensemble ~26%.

### Chaotic Lorenz Attractor (Section 5.2)

30-model ensemble comparison with Lyapunov analysis.

```bash
python examples/lorenz/train_lorenz.py
```

**Expected results (Table 4):** UPN achieves ~94.5% coverage vs Ensemble ~66.8%.

**Expected results (Table 5):**

| Method | MSE | 95% Coverage |
|---|---|---|
| UPN (without updates) | ~0.06 | ~81–96% |
| UPN (with updates) | ~0.004–0.006 | ~97–98% |
| Latent ODE | ~0.11–0.20 | ~90–92% |
| Ensemble Neural ODE | ~0.03 | ~58–68% |
| Neural SDE | ~0.03 | ~44–53% |
| Baseline | ~1.39 | N/A |

MSE reduction from Kalman updates: **~90%+**

---

## Baselines

The library includes three baseline implementations for fair comparison:

```python
from upn.baselines import LatentODE, EnsembleNeuralODE, NeuralSDE

# Latent ODE (RNN encoder + ODE dynamics + Gaussian decoder)
latent = LatentODE(obs_dim=3, latent_dim=6, hidden_dim=32)

# Ensemble Neural ODE (5 independently trained models)
ensemble = EnsembleNeuralODE(state_dim=6, hidden_dim=32, n_ensemble=5)

# Neural SDE (Euler-Maruyama integration)
sde = NeuralSDE(state_dim=6, hidden_dim=32)
```

---

## Citation

```bibtex
@article{jahanshahi2026uncertainty,
  title={Uncertainty Propagation Networks for Neural Ordinary Differential Equations},
  author={Jahanshahi, Hadi and Zhu, Zheng H.},
  journal={Neurocomputing},
  volume={677},
  pages={133134},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.neucom.2026.133134}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work was supported by NSERC Discovery Grant (RGPIN2024-06290) and CREATE Program Grant (555425-2021).

**Authors:** Hadi Jahanshahi & Zheng H. Zhu, York University, Canada.
