"""Toy-tested legacy projection adapter; outside the four-method pilot grid."""
import numpy as np
from scipy import sparse
from src.sketches import _balanced_bucket_assignment
from .state import deep_bytes


class LegacyJL:
    def __init__(self, dimension, width, seed=42):
        if not 2 <= width < dimension:
            raise ValueError('This compatibility adapter covers compressed projections')
        rng = np.random.default_rng(seed)
        assignment, occupancy = _balanced_bucket_assignment(dimension, width, rng)
        signs = rng.choice([-1., 1.], size=dimension)
        weights = signs / np.sqrt(occupancy[assignment])
        self.projection = sparse.csr_matrix((weights, (np.arange(dimension), assignment)),
                                            shape=(dimension, width))

    def row(self, row):
        result = (sparse.csr_matrix(row, copy=True) @ self.projection).tocsr()
        result.eliminate_zeros()
        return result

    def state_bytes(self):
        return deep_bytes(self)
