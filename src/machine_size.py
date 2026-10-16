"""Resource accounting under the validated Bank row-only sparsity convention.

The QOS reference uses maximum row-or-column sparsity. Bank's estimate of 41
is conditional on row-only sparsity and is not a measured hardware footprint.
Counts here are mathematical nonzeros, not storage slots: explicit zeros and
duplicate sparse entries that cancel are excluded. The reference's matrix_nnz
instead counts sparse storage slots (and all entries for dense inputs).
"""

import operator

import numpy as np
from scipy import sparse


def _countable_matrix(X):
    if sparse.issparse(X):
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        # Canonicalize a private sparse copy; never densify or mutate the input.
        matrix = X.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix
    matrix = np.asarray(X)
    if matrix.ndim != 2:
        raise ValueError("X must be two-dimensional")
    return matrix


def max_row_sparsity(X):
    """Maximum mathematical nonzeros in any row; zero for empty matrices."""
    matrix = _countable_matrix(X)
    if sparse.issparse(matrix):
        counts = np.diff(matrix.indptr)
    else:
        counts = np.count_nonzero(matrix, axis=1)
    return int(counts.max()) if counts.size else 0


def max_row_or_col_sparsity(X):
    """Maximum mathematical nonzeros in any row or column (the QOS reference s).

Not the Bank convention: Bank accounting uses max_row_sparsity. Zero for
empty matrices.
"""
    matrix = _countable_matrix(X)
    if sparse.issparse(matrix):
        rows = np.diff(matrix.indptr)
        columns = np.bincount(matrix.indices, minlength=matrix.shape[1])
    else:
        rows = np.count_nonzero(matrix, axis=1)
        columns = np.count_nonzero(matrix, axis=0)
    return int(max(rows.max(initial=0), columns.max(initial=0)))


def matrix_nnz(X):
    """Count mathematical nonzeros in a dense or sparse 2-D matrix."""
    matrix = _countable_matrix(X)
    return int(matrix.nnz if sparse.issparse(matrix) else np.count_nonzero(matrix))


def qos_svm_machine_size(n_samples, n_features, sparsity):
    """Return 2*ceil(log2(N+2D)) + ceil(log2(s+1)) + 4.

N and D must be positive integers and s a nonnegative integer. For Bank, pass
max_row_sparsity(X); the reference uses maximum row-or-column sparsity instead.
The +4 is three logical qubits plus one classical scalar register, in the
reference's accounting units. Integer arithmetic avoids rounding at powers
of two. No PCA or hardware execution accounting is implied.
"""
    values = []
    for name, value, minimum in [("n_samples", n_samples, 1),
                                 ("n_features", n_features, 1),
                                 ("sparsity", sparsity, 0)]:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer, not a boolean")
        try:
            integer = operator.index(value)
        except TypeError:
            raise TypeError(f"{name} must be an integer") from None
        if integer < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        values.append(integer)
    N, D, s = values
    return 2 * (N + 2 * D - 1).bit_length() + s.bit_length() + 4
