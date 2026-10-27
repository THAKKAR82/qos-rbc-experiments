"""Elliptic structural check: edges across time steps decide the gamma = 0 gate."""
import pandas as pd
import pytest

from experiments.sparsity_screen import elliptic_structure as es


def _toy():
    step = pd.Series({1: 1, 2: 1, 3: 2, 4: 2, 5: 3}, name="step")
    edges = pd.DataFrame({"txId1": [1, 2, 3, 4, 2], "txId2": [2, 3, 4, 5, 1]})
    return edges, step


def test_cross_step_edges_counts_endpoints_in_different_steps():
    edges, step = _toy()
    assert es.cross_step_edges(edges, step) == 2  # 2->3 (1->2) and 4->5 (2->3).


def test_cross_step_edges_rejects_unknown_nodes():
    edges, step = _toy()
    with pytest.raises(KeyError):
        es.cross_step_edges(pd.concat([edges, pd.DataFrame({"txId1": [9], "txId2": [1]})]), step)


def test_step_aligned_cold_start_is_one_without_cross_step_edges():
    step = pd.Series({1: 1, 2: 1, 3: 2, 4: 2}, name="step")
    edges = pd.DataFrame({"txId1": [1, 3], "txId2": [2, 4]})
    labels = pd.Series({1: 1, 2: -1, 3: 1, 4: -1})
    assert es.step_aligned_cold_start(edges, step, labels, last_train_step=1) == 1.0
    edges = pd.concat([edges, pd.DataFrame({"txId1": [3], "txId2": [1]})])   # One cross-step edge.
    assert es.step_aligned_cold_start(edges, step, labels, last_train_step=1) == 0.5
