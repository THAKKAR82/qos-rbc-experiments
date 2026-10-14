"""Validated Bank feature sketches, with seed-local deterministic randomness.

Dense bucketing retains the notebook's division and accumulation order rather
than switching to sparse multiplication: both rounding and Ridge's automatic
solver matter to the validated results. Sparse inputs are never densified.
Only sparsity=1 JL is supported. The QOS reference repairs collisions between
assignment rounds at higher sparsity; the Bank notebook does not, and that
extension has not been validated here. Functions return matrices, not the
reference's (matrix, lifting_metadata) tuples.
"""

import operator

import numpy as np
import scipy.sparse as sp


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _balanced_bucket_assignment(n_features, n_buckets, rng):
    """Consume one permutation; assign features to near-equal-sized buckets."""
    permutation = rng.permutation(n_features)
    bucket_of_feature = np.empty(n_features, dtype=int)
    bucket_of_feature[permutation] = np.arange(n_features) * n_buckets // n_features
    bucket_sizes = np.bincount(bucket_of_feature, minlength=n_buckets).astype(float)
    return bucket_of_feature, bucket_sizes


def balanced_feature_bucket(X, n_buckets, seed=42):
    """Merge features with weights 1/sqrt(bucket occupancy).

    Dense input returns a dense array, preserving Bank's automatic Ridge solver
    path. Sparse input returns CSR for compressed output; at k>=D a copy in the
    input's format is returned. This input-preserving contract needs no output
    option: explicitly convert the input if a different representation is
    desired. k>=D preserves values, dtype, and dimension without randomization.
    """
    n_buckets = _positive_integer(n_buckets, "n_buckets")
    if not sp.issparse(X):
        X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    n_features = X.shape[1]
    if n_buckets >= n_features:
        return X.copy()

    rng = np.random.default_rng(seed)
    assignment, occupancy = _balanced_bucket_assignment(n_features, n_buckets, rng)
    if sp.issparse(X):
        weights = 1.0 / np.sqrt(occupancy[assignment])
        projection = sp.csr_matrix(
            (weights, (np.arange(n_features), assignment)),
            shape=(n_features, n_buckets),
        )
        # Private input copy also protects noncanonical sparse inputs.
        result = (sp.csr_matrix(X, copy=True) @ projection).tocsr()
        result.eliminate_zeros()
        return result

    result = np.zeros((X.shape[0], n_buckets), dtype=float)
    for j in range(n_features):
        bucket = assignment[j]
        result[:, bucket] += X[:, j] / np.sqrt(occupancy[bucket])
    return result


def sparse_jl_features(X, n_buckets, seed=42, sparsity=1):
    """Balanced signed, occupancy-normalized JL with one entry per feature.

    Always returns CSR. RNG order is one permutation followed by D random
    signs, exactly as in the validated notebook and reference for sparsity=1.
    Preserve the notebook's division by sqrt(occupancy), rather than changing
    it to multiplication by the reciprocal. Compressed k must be at least 2,
    as in the notebook/reference. k>=D returns an independent CSR identity copy.
    Higher sparsity is rejected even at the identity endpoint; it is unvalidated
    and the notebook lacks the reference's multi-round collision repair.
    """
    n_buckets = _positive_integer(n_buckets, "n_buckets")
    sparsity = _positive_integer(sparsity, "sparsity")
    if sparsity != 1:
        raise ValueError("Only validated sparsity=1 is supported")
    if getattr(X, "ndim", np.ndim(X)) != 2:
        raise ValueError("X must be two-dimensional")
    X = sp.csr_matrix(X, copy=True)
    n_features = X.shape[1]
    if n_buckets >= n_features:
        return X
    if n_buckets < 2:
        raise ValueError("Compressed sparse JL requires n_buckets >= 2")

    rng = np.random.default_rng(seed)
    assignment, occupancy = _balanced_bucket_assignment(n_features, n_buckets, rng)
    signs = rng.choice([-1.0, 1.0], size=n_features)
    weights = signs / np.sqrt(occupancy[assignment])
    projection = sp.csr_matrix(
        (weights, (np.arange(n_features), assignment)),
        shape=(n_features, n_buckets),
    )
    result = (X @ projection).tocsr()
    result.eliminate_zeros()
    return result
