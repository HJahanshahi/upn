"""
Half-vectorization (vech) and its inverse for symmetric matrices.

These operators compress the covariance matrix from n² to n(n+1)/2
elements by exploiting symmetry (Section 3.3, Eq. 6).
"""

import torch


def vech(A: torch.Tensor) -> torch.Tensor:
    """
    Half-vectorization: extract lower-triangular elements of a symmetric matrix.

    Args:
        A: Symmetric matrix [batch, n, n].

    Returns:
        Half-vectorized form [batch, n(n+1)/2].
    """
    n = A.shape[1]
    indices = torch.tril_indices(n, n, device=A.device)
    return A[:, indices[0], indices[1]]


def unvech(v: torch.Tensor, n: int) -> torch.Tensor:
    """
    Inverse half-vectorization: reconstruct symmetric matrix from lower-triangular elements.

    Args:
        v: Half-vectorized form [batch, n(n+1)/2].
        n: Matrix dimension.

    Returns:
        Symmetric matrix [batch, n, n].
    """
    batch_size = v.shape[0]
    device = v.device

    A = torch.zeros(batch_size, n, n, device=device)
    indices = torch.tril_indices(n, n, device=device)
    A[:, indices[0], indices[1]] = v
    # Mirror to upper triangle
    A = A + A.transpose(1, 2) - torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2))
    return A
