# Core components
try:
    from .core.upn_base import UPNBase
    from .core.dynamics import DynamicsNetwork
    from .core.noise import ProcessNoiseNetwork
    from .core.ode_solver import (
        vech, unvech, compute_jacobian_batched, 
        solve_coupled_ode, ensure_positive_definite
    )
except ImportError:
    # Fallback for old structure
    try:
        from .core import DynamicsNetwork, ProcessNoiseNetwork, UPN as UPNBase
    except ImportError:
        print("Warning: Could not import core components")

# Dynamical systems applications
try:
    from .applications.dynamical import (
        DynamicalSystemDataset,
        UPNDynamicalSystem,
        DeterministicODENet,
        EnsembleODENet,
        DynamicalSystemTrainer
    )
except ImportError:
    print("Warning: Could not import dynamical systems components")

# Lorenz system specific
try:
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
except ImportError:
    print("Warning: Could not import Lorenz system components")

# Legacy applications (if they exist)
try:
    from .applications import (
        DynamicalSystemUPN,
        TimeSeriesUPN,
        SyntheticTimeSeriesDataset, 
        create_synthetic_time_series,
        collate_fn,
        evaluate_time_series_upn,
        visualize_time_series_upn,
        evaluate_forecasting,
        visualize_forecasting,
        evaluate_missing_data,
        visualize_missing_data,
        UPNFlow,
        generate_flow_data,
        train_upn_flow,
        visualize_flow_density,
        visualize_flow_samples,
        visualize_flow_transformation,
        LorenzDataset,
        LorenzSystemUPN
    )
except ImportError:
    print("Warning: Could not import legacy application components")

# Utilities (if they exist)
try:
    from .utils.data import (
        generate_oscillator_data,
        prepare_dataloaders
    )
except ImportError:
    try:
        from .data import (
            generate_oscillator_data,
            prepare_dataloaders
        )
    except ImportError:
        print("Warning: Could not import data utilities")

try:
    from .utils.visualization import (
        plot_trajectory_with_uncertainty,
        plot_2d_phase_with_uncertainty
    )
except ImportError:
    try:
        from .visualization import (
            plot_trajectory_with_uncertainty,
            plot_2d_phase_with_uncertainty
        )
    except ImportError:
        print("Warning: Could not import visualization utilities")

version = '0.1.0'

# Define what's available for import
__all__ = [
    # Core classes (new structure)
    'UPNBase', 'DynamicsNetwork', 'ProcessNoiseNetwork',
    'vech', 'unvech', 'compute_jacobian_batched', 'solve_coupled_ode', 'ensure_positive_definite',
    
    # Dynamical systems (new structure)
    'DynamicalSystemDataset', 'UPNDynamicalSystem', 'DeterministicODENet', 
    'EnsembleODENet', 'DynamicalSystemTrainer',
    
    # Lorenz system (new structure)
    'lorenz_system', 'generate_lorenz_data', 'prepare_lorenz_dataloaders',
    'evaluate_lorenz_system', 'evaluate_models', 'calculate_nll', 'calculate_crps',
    'calculate_interval_coverage', 'visualize_lorenz_3d', 'visualize_lorenz_2d',
    'visualize_lorenz_3d_with_time_analysis', 'visualize_horizon_results',
    'plot_training_curves', 'evaluate_across_time_horizons',
    
    # Legacy applications (if available)
    'DynamicalSystemUPN', 'TimeSeriesUPN', 'UPNFlow', 'LorenzSystemUPN', 'LorenzDataset',
    'SyntheticTimeSeriesDataset', 'create_synthetic_time_series', 'collate_fn',
    'evaluate_time_series_upn', 'visualize_time_series_upn', 
    'evaluate_forecasting', 'visualize_forecasting',
    'evaluate_missing_data', 'visualize_missing_data',
    'generate_flow_data', 'train_upn_flow',
    'visualize_flow_density', 'visualize_flow_samples', 'visualize_flow_transformation',
    
    # Utilities (if available)
    'generate_oscillator_data', 'prepare_dataloaders',
    'plot_trajectory_with_uncertainty', 'plot_2d_phase_with_uncertainty'
]

# Clean up __all__ to only include actually imported items
_available_items = []
for item in __all__:
    if item in globals():
        _available_items.append(item)

__all__ = _available_items