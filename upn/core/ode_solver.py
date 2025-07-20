import torch
from torch.autograd.functional import jacobian

def vech(matrix):
    """
    Half-vectorization operator that stacks the lower triangular part of a symmetric matrix.
    
    Args:
        matrix: Input symmetric matrix [..., n, n]
        
    Returns:
        Vector with the lower triangular elements [..., n*(n+1)/2]
    """
    n = matrix.shape[-1]
    tril_indices = torch.tril_indices(n, n)
    return matrix[..., tril_indices[0], tril_indices[1]]

def unvech(vector, n):
    """
    Convert a half-vectorized representation back to a symmetric matrix.
    
    Args:
        vector: Half-vectorized matrix [..., n*(n+1)/2]
        n: Size of the matrix
        
    Returns:
        Symmetric matrix [..., n, n]
    """
    batch_shape = vector.shape[:-1]
    matrix = torch.zeros((*batch_shape, n, n), device=vector.device)
    
    # Get indices for the lower triangular part
    tril_indices = torch.tril_indices(n, n)
    
    # Fill the lower triangular part
    matrix[..., tril_indices[0], tril_indices[1]] = vector
    
    # Fill the upper triangular part (for symmetry)
    triu_indices = torch.triu_indices(n, n, 1)  # Strict upper triangular (excluding diagonal)
    matrix[..., triu_indices[1], triu_indices[0]] = matrix[..., triu_indices[0], triu_indices[1]]
    
    return matrix

def compute_jacobian_batched(func, x, t):
    """
    Compute Jacobian of func with respect to x for each element in batch.
    
    Args:
        func: Function that computes the dynamics (neural network)
        x: Input tensor [batch_size, state_dim]
        t: Time value (scalar or tensor)
        
    Returns:
        Jacobian matrix [batch_size, state_dim, state_dim]
    """
    batch_size, state_dim = x.shape
    J = torch.zeros(batch_size, state_dim, state_dim, device=x.device, dtype=x.dtype)
    
    # More efficient approach using autograd.grad
    for i in range(state_dim):
        # Enable gradients for input
        x_grad = x.clone().detach().requires_grad_(True)
        
        # Compute output
        output = func(t, x_grad)
        
        # Compute gradient of i-th output component w.r.t. all input components
        grad_outputs = torch.zeros_like(output)
        grad_outputs[:, i] = 1.0
        
        grad_inputs = torch.autograd.grad(
            outputs=output,
            inputs=x_grad,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        
        J[:, i, :] = grad_inputs
    
    return J

def solve_coupled_ode(dynamics_func, initial_state, t_span, method='dopri5', **kwargs):
    """
    Solve coupled ODE system for UPN dynamics.
    
    Args:
        dynamics_func: Function that computes the dynamics (should be a UPN instance)
        initial_state: Initial combined state [batch_size, state_dim + cov_dim]
        t_span: Time points to solve at [time_steps]
        method: ODE solver method
        **kwargs: Additional arguments for the ODE solver
        
    Returns:
        Solution trajectory [time_steps, batch_size, state_dim + cov_dim]
    """
    from torchdiffeq import odeint
    
    # Ensure t_span is 1D
    if t_span.ndim > 1:
        t_span = t_span.flatten()
    
    # Sort time points to ensure monotonic order
    t_span, _ = torch.sort(t_span)
    
    return odeint(dynamics_func, initial_state, t_span, method=method, **kwargs)

def ensure_positive_definite(cov_matrix, eps=1e-6):
    """
    Ensure covariance matrix is positive definite by adding small diagonal term.
    
    Args:
        cov_matrix: Covariance matrix [..., n, n]
        eps: Small positive value to add to diagonal
        
    Returns:
        Positive definite covariance matrix
    """
    if cov_matrix.ndim == 2:
        n = cov_matrix.shape[-1]
        return cov_matrix + eps * torch.eye(n, device=cov_matrix.device)
    else:
        # Handle batched matrices
        *batch_dims, n, _ = cov_matrix.shape
        eye = torch.eye(n, device=cov_matrix.device).expand(*batch_dims, n, n)
        return cov_matrix + eps * eye










# import torch
# from torch.autograd.functional import jacobian

# def vech(matrix):
#     """
#     Half-vectorization operator that stacks the lower triangular part of a symmetric matrix.
    
#     Args:
#         matrix: Input symmetric matrix [..., n, n]
        
#     Returns:
#         Vector with the lower triangular elements [..., n*(n+1)/2]
#     """
#     n = matrix.shape[-1]
#     tril_indices = torch.tril_indices(n, n)
#     return matrix[..., tril_indices[0], tril_indices[1]]

# def unvech(vector, n):
#     """
#     Convert a half-vectorized representation back to a symmetric matrix.
    
#     Args:
#         vector: Half-vectorized matrix [..., n*(n+1)/2]
#         n: Size of the matrix
        
#     Returns:
#         Symmetric matrix [..., n, n]
#     """
#     batch_shape = vector.shape[:-1]
#     matrix = torch.zeros((*batch_shape, n, n), device=vector.device)
    
#     # Get indices for the lower triangular part
#     tril_indices = torch.tril_indices(n, n)
    
#     # Fill the lower triangular part
#     matrix[..., tril_indices[0], tril_indices[1]] = vector
    
#     # Fill the upper triangular part (for symmetry)
#     triu_indices = torch.triu_indices(n, n, 1)  # Strict upper triangular (excluding diagonal)
#     matrix[..., triu_indices[1], triu_indices[0]] = matrix[..., triu_indices[0], triu_indices[1]]
    
#     return matrix

# def compute_jacobian_batched(func, x, t):
#     """
#     Compute Jacobian of func with respect to x for each element in batch.
    
#     Args:
#         func: Function that computes the dynamics
#         x: Input tensor [batch_size, state_dim]
#         t: Time value or tensor
        
#     Returns:
#         Jacobian matrix [batch_size, state_dim, state_dim]
#     """
#     batch_size, state_dim = x.shape
#     J = torch.zeros(batch_size, state_dim, state_dim, device=x.device)
    
#     for i in range(batch_size):
#         # Compute Jacobian for each element in the batch
#         x_i = x[i].unsqueeze(0)  # Unsqueeze to keep batch dimension
        
#         def func_i(x_single):
#             return func(t, x_single.unsqueeze(0))[0]
        
#         J[i] = jacobian(func_i, x_i[0])
        
#     return J