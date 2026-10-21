"""Frozen-selection test evaluation: exact reproduction of saved validation results, then test scoring.

Light synthetic fixtures only (no CFPB data).
"""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state import scoped as sc
from experiments.cfpb_streaming_state.confirmation import scoped_test as st


def keys(ids):
    return np.column_stack((np.asarray(ids, np.uint64), np.zeros(len(ids), np.uint64)))


def _rows(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cols = rng.choice(30, size=5, replace=False)
        x = rng.random(5)
        out.append((keys(cols), x / np.linalg.norm(x), 1 if x[0] + (cols[0] < 15) > 1 else -1))
    return out


@pytest.fixture(scope="module")
def saved():
    train, val, test = _rows(60, 1), _rows(40, 2), _rows(50, 3)
    cell = dict(id="toy", task="toy", representation="B_causal", method="hash_sgd", budget_bytes=4096)
    result = sc.run_cell(cell, train, val, ref_predictions=np.array([r[2] for r in val]), featurizer_bytes=1)
    return train, val, test, result


def test_frozen_evaluation_reproduces_saved_validation_and_scores_test(saved):
    train, val, test, result = saved
    out = st.evaluate_frozen_cell(result, train, val, test)
    assert [e["seed"] for e in out["evaluation"]] == sc.EVALUATION_SEEDS
    for e, s in zip(out["evaluation"], result["evaluation"]):
        assert e["validation_reproduced"] is True
        assert len(e["test_predictions"]) == len(test) and set(e["test_predictions"]) <= {-1, 1}
        assert [c["passes"] for c in e["test_curve"]] == list(sc.EVAL_CHECKPOINT_PASSES)
    # Same frozen configuration and seed -> same test predictions on a second run.
    again = st.evaluate_frozen_cell(result, train, val, test)
    assert [e["test_predictions"] for e in again["evaluation"]] == [e["test_predictions"] for e in out["evaluation"]]
    assert out["selected_config"] == result["selected_config"]


def test_frozen_evaluation_refuses_when_validation_is_not_reproduced(saved):
    train, val, test, result = saved
    tampered = copy.deepcopy(result)
    p = tampered["evaluation"][1]["predictions"]
    p[0] = -p[0]
    with pytest.raises(st.ReproductionError, match="seed 123"):
        st.evaluate_frozen_cell(tampered, train, val, test)


def test_frozen_evaluation_refuses_changed_allocation(saved):
    train, val, test, result = saved
    tampered = copy.deepcopy(result)
    tampered["allocations"]["evaluation"]["42"]["width"] += 1
    with pytest.raises(st.ReproductionError, match="allocation"):
        st.evaluate_frozen_cell(tampered, train, val, test)
