"""Full-dimensional multi-pass SGD convergence controls (approved: 12 trajectories, 10-minute cap)."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state import convergence as cv
from experiments.cfpb_streaming_state.methods import Learner


def test_checkpoint_schedule_six_points_through_5N():
    assert cv.checkpoints(7265) == [909, 1817, 3633, 7265, 14530, 36325]


def test_pass_and_cursor_mapping():
    assert cv.pass_cursor(0, 10) == (0, 0)
    assert cv.pass_cursor(9, 10) == (0, 9)
    assert cv.pass_cursor(10, 10) == (1, 0)
    assert cv.pass_cursor(23, 10) == (2, 3)


def test_plan_is_exactly_twelve_declared_trajectories():
    plan = cv.plan()
    assert len(plan) == 12 and len({c["id"] for c in plan}) == 12
    assert {c["task"] for c in plan} == {"credit_vs_checking", "mortgage_vs_rest"}
    assert {c["order"] for c in plan} == {"chronological", "random"}
    assert {c["eta0"] for c in plan} == {0.1, 0.3, 1.0}
    assert all(c["representation"] == "A_frozen" and c["method"] == "identity_sgd" and c["max_passes"] == 5
               and c["seed"] == 42 for c in plan)


def test_retention_is_ratio_to_frozen_ridge():
    assert cv.retention(dict(accuracy=0.9, balanced_accuracy=0.8),
                        dict(accuracy=0.95, balanced_accuracy=0.9)) == pytest.approx(
        dict(accuracy=0.9 / 0.95, balanced_accuracy=0.8 / 0.9, joint=min(0.9 / 0.95, 0.8 / 0.9)))


def test_identity_sgd_two_passes_matches_dense_reference():
    rng = np.random.default_rng(42)
    n, d = 30, 12
    X = rng.standard_normal((n, d)) * (rng.random((n, d)) < .4)
    y = np.where(rng.random(n) < .5, 1, -1)
    eta0, lam = .3, 1 / n
    learner = Learner("identity_sgd", d, eta=eta0, lam=lam)
    w, t = np.zeros(d), 0
    for _ in range(2):  # cyclic replay of the same order, as declared
        for i in range(n):
            cols = np.flatnonzero(X[i])
            keys = np.column_stack((cols.astype(np.uint64), np.zeros(len(cols), np.uint64)))
            learner.update(keys, X[i, cols], int(y[i]))
            eta = eta0 / (1 + eta0 * lam * t)
            pred = X[i] @ w
            w = (1 - eta * lam) * w + eta * (y[i] - pred) * X[i]
            t += 1
    np.testing.assert_allclose(learner.weights * learner.scale, w, rtol=1e-10, atol=1e-12)
