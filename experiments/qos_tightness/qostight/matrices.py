"""Synthetic matrices.

Circulant support gives UNIFORM sparsity: every row has s nonzeros and every
column s*N/D. This is the ideal-sparsity case for QOS, unlike real data whose
column sparsity is dominated by a few very common features.
"""
import numpy as np

IDEAL_SPARSITY_NOTE = ("synthetic circulant support: uniform row and column sparsity (ideal case), "
                       "unlike real data whose s is set by heavy columns")


def circulant_sparse(N, D, s, seed=42):
    """N x D matrix, row i nonzero at columns (i + o_k) mod D for s distinct offsets; ||X|| = 1."""
    if not 1 <= s <= D or N % D:
        raise ValueError("need 1 <= s <= D and D | N")
    rng = np.random.default_rng(seed)
    offsets = rng.choice(D, size=s, replace=False)
    X = np.zeros((N, D))
    rows = np.repeat(np.arange(N), s)
    cols = (rows + np.tile(offsets, N)) % D
    X[rows, cols] = rng.choice([-1.0, 1.0], size=N * s) * rng.uniform(0.5, 1.0, size=N * s)
    return X / np.linalg.norm(X, 2)
