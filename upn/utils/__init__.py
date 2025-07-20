# Core components
from .core.upn_base import UPNBase
from .core.dynamics import DynamicsNetwork
from .core.noise import ProcessNoiseNetwork
from .core.ode_solver import (
    vech, unvech, compute_jacobian_batched, 
    solve_coupled_ode, ensure_positive_definite
)

# Dynamical systems applications
from .applications.dynamical import (
    DynamicalSystemDataset,
    UPNDynamicalSystem,
    DeterministicODENet,
    EnsembleODENet,
    DynamicalSystemTrainer
)

# Lorenz system specific
from .applications.lorenz import (
    lorenz_system,
    generate_lorenz_data,
    prepare_lorenz_dataloaders,
    evaluate_lorenz_system,
    evaluate_models,
    calculate_nll,
    calculate_crps,
    calculate_interval_coverage,
    visualize_lorenz_3d,
    visualize_lorenz_2d,
    visualize_lorenz_3d_with_time_analysis,
    visualize_horizon_results,
    plot_training_curves,
    evaluate_across_time_horizons
)

# Legacy imports (if you have these files)
try:
    from .data import (
        generate_oscillator_data,
        prepare_dataloaders
    )
except ImportError:
    pass

try:
    from .visualization import (
        plot_trajectory_with_uncertainty,
        plot_2d_phase_with_uncertainty
    )
except ImportError:
    pass

__all__ = [
    # Core components
    'UPNBase',
    'DynamicsNetwork', 
    'ProcessNoiseNetwork',
    'vech',
    'unvech',
    'compute_jacobian_batched',
    'solve_coupled_ode',
    'ensure_positive_definite',
    
    # Dynamical systems
    'DynamicalSystemDataset',
    'UPNDynamicalSystem',
    'DeterministicODENet', 
    'EnsembleODENet',
    'DynamicalSystemTrainer',
    
    # Lorenz system
    'lorenz_system',
    'generate_lorenz_data',
    'prepare_lorenz_dataloaders',
    'evaluate_lorenz_system',
    'evaluate_models',
    'calculate_nll',
    'calculate_crps', 
    'calculate_interval_coverage',
    'visualize_lorenz_3d',
    'visualize_lorenz_2d',
    'visualize_lorenz_3d_with_time_analysis',
    'visualize_horizon_results',
    'plot_training_curves',
    'evaluate_across_time_horizons',
    
    # Legacy (if available)
    'generate_oscillator_data',
    'prepare_dataloaders',
    'plot_trajectory_with_uncertainty',
    'plot_2d_phase_with_uncertainty'
]