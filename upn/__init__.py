"""
UPN — Uncertainty Propagation Networks for Neural Ordinary Differential Equations.

A PyTorch library for continuous-time modeling with principled uncertainty
quantification via coupled mean–covariance ODEs.

Reference:
    Jahanshahi, H. & Zhu, Z.H. (2026). Uncertainty Propagation Networks
    for Neural Ordinary Differential Equations. Neurocomputing, 677, 133134.
"""

from .core import UPN, DynamicsNetwork, ProcessNoiseNetwork, vech, unvech

__version__ = "1.0.0"

__all__ = [
    "UPN",
    "DynamicsNetwork",
    "ProcessNoiseNetwork",
    "vech",
    "unvech",
]
