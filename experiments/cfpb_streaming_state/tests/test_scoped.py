"""Scoped classical benchmark: plan, schedules, budgets, successive halving, paired bootstrap k*.

Light synthetic fixtures only (no CFPB data, no full encoder vocabulary load).
"""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state import scoped as sc
from experiments.cfpb_streaming_state.methods import Learner, budgeted


def keys(ids):
    return np.column_stack((np.asarray(ids, np.uint64), np.zeros(len(ids), np.uint64)))


# ---------------------------------------------------------------- plan
def test_plan_is_108_cells_with_mission_excluded():
    plan = sc.plan()
    assert len(plan["cells"]) == 3 * 2 * 3 * 6 == 108 and len({c["id"] for c in plan["cells"]}) == 108
    assert {c["method"] for c in plan["cells"]} == {"hash_sgd", "sparse_jl_sgd", "awm"}
    assert "mission" in plan["excluded_methods"] and plan["excluded_methods"]["mission"]
    assert {c["budget_bytes"] for c in plan["cells"]} == {k * 1024 for k in (4, 16, 32, 64, 128, 512)}
    assert {c["task"] for c in plan["cells"]} == {"credit_vs_checking", "checking_vs_mortgage", "mortgage_vs_rest"}
    assert plan["ordering"] == "chronological" and plan["tuning_seed"] == 2026
    assert plan["evaluation_seeds"] == [42, 123, 456] and len(plan["candidates"]) == 8
    assert plan["halving"] == [[8, 1], [4, 2], [2, 5]]


def test_checking_vs_mortgage_task_spec():
    t = sc.SCOPED_TASKS["checking_vs_mortgage"]
    assert (t["positive"], t["negative"], t["N"], t["validation_N"]) == (2, 3, 4987, 1107)
    assert t["audit_id"] == "checking_or_savings_account__vs__mortgage"
    assert set(sc.SCOPED_TASKS) == {"credit_vs_checking", "checking_vs_mortgage", "mortgage_vs_rest"}


# ---------------------------------------------------------------- schedules (pilot default unchanged)
def test_inverse_time_default_is_unchanged_and_inverse_sqrt_formula():
    a = Learner("identity_sgd", 4, eta=0.3, lam=0.1)
    assert a.schedule == "inverse_time"
    assert a.step_size(10) == pytest.approx(0.3 / (1 + 0.3 * 0.1 * 10))
    b = Learner("identity_sgd", 4, eta=0.3, lam=0.1, schedule="inverse_sqrt", tau=50)
    assert b.step_size(10) == pytest.approx(0.3 / math.sqrt(1 + 10 / 50))
    with pytest.raises(ValueError):
        Learner("identity_sgd", 4, schedule="constant_typo")


def test_schedule_survives_snapshot_restore_and_old_snapshots_default():
    b = Learner("identity_sgd", 4, eta=0.3, lam=0.1, schedule="inverse_sqrt", tau=50)
    b.update(keys([0, 1]), np.array([1.0, 0.5]), 1)
    meta, arrays = b.snapshot()
    r = Learner.restore(meta, arrays)
    assert (r.schedule, r.tau) == ("inverse_sqrt", 50.0)
    old = {k: v for k, v in meta.items() if k not in ("schedule", "tau")}
    assert Learner.restore(old, arrays).schedule == "inverse_time"


@pytest.mark.parametrize("method", ["hash_sgd", "sparse_jl_sgd", "awm"])
@pytest.mark.parametrize("kib", [4, 16, 512])
def test_budgeted_passes_eta_schedule_and_fits_every_budget(method, kib):
    m = budgeted(method, kib * 1024, 1000, seed=2026, eta=0.03, schedule="inverse_sqrt", tau=1000)
    assert m.eta0 == 0.03 and m.schedule == "inverse_sqrt"
    assert m.state_bytes() + 512 <= kib * 1024


# ---------------------------------------------------------------- successive halving
def test_successive_halving_keeps_top_k_and_counts_cost():
    quality = {i: q for i, q in enumerate([.5, .9, .7, .8, .6, .95, .3, .85])}
    calls = []

    def advance(cand, passes):
        calls.append((cand, passes))
        return quality[cand] + 0.001 * passes  # monotone in passes, preserves order

    out = sc.successive_halving(list(range(8)), advance, halving=[[8, 1], [4, 2], [2, 5]])
    assert out["selected"] == 5
    assert [len(r["survivors"]) for r in out["rungs"]] == [8, 4, 2]
    assert out["rungs"][1]["survivors"] == [5, 1, 7, 3]
    assert out["pass_equivalents"] == 8 * 1 + 4 * (2 - 1) + 2 * (5 - 2) == 18


def test_successive_halving_ties_go_to_lower_index():
    out = sc.successive_halving([0, 1, 2, 3], lambda c, p: 0.5, halving=[[4, 1], [2, 2], [1, 3]])
    assert out["rungs"][1]["survivors"] == [0, 1] and out["selected"] == 0


# ---------------------------------------------------------------- retention, k*, paired bootstrap
def _synthetic_predictions(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < .5, 1, -1)
    ref = y.copy(); ref[:20] *= -1                                # Ridge: 95% accurate
    preds = {}
    for budget, wrong in ((4096, 80), (16384, 40), (65536, 24), (524288, 21)):
        p = y.copy(); p[:wrong] *= -1
        preds[budget] = np.stack([p, p, p])                       # 3 identical seeds
    return y, ref, preds


def test_joint_retention_and_kstar_point_estimates():
    y, ref, preds = _synthetic_predictions()
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    def expected(P):
        acc = np.mean([accuracy_score(y, s) for s in P]) / accuracy_score(y, ref)
        ba = np.mean([balanced_accuracy_score(y, s) for s in P]) / balanced_accuracy_score(y, ref)
        return min(acc, ba)
    ret = {b: sc.joint_retention(y, P, ref) for b, P in preds.items()}
    for b, P in preds.items():
        assert ret[b] == pytest.approx(expected(P), rel=1e-12)
    want95 = next((b for b in sorted(ret) if ret[b] >= 0.95), None)
    assert sc.k_star(ret, 0.95) == want95 and want95 is not None
    assert sc.k_star({4096: .5, 16384: .6}, 0.95) is None


def test_paired_bootstrap_uses_shared_indices_and_collapses_when_deterministic():
    y, ref, preds = _synthetic_predictions()
    out = sc.bootstrap_kstar(y, ref, preds, thresholds=(0.95, 0.98), n_boot=200, seed=42)
    assert set(out) == {0.95, 0.98}
    for r in out.values():
        assert r["point"] in (None, 4096, 16384, 65536, 524288)
        assert "ci95" in r and "not_reached_fraction" in r
        assert 0 <= r["not_reached_fraction"] <= 1
    # Shared indices: reference and every budget are resampled together, so a candidate identical to
    # the reference has joint retention exactly 1 in every replicate.
    same = sc.bootstrap_kstar(y, ref, {4096: np.stack([ref, ref, ref])}, thresholds=(0.99,), n_boot=50, seed=42)
    assert same[0.99]["point"] == 4096 and same[0.99]["ci95"] == [4096, 4096]


# ---------------------------------------------------------------- one cell end to end on a tiny spool
def test_run_cell_tiny_synthetic_end_to_end():
    rng = np.random.default_rng(1)
    def row(i):
        cols = rng.choice(30, size=5, replace=False)
        x = rng.random(5)
        y = 1 if x[0] + (cols[0] < 15) > 1 else -1
        return keys(cols), x / np.linalg.norm(x), y
    train = [row(i) for i in range(60)]
    val = [row(i) for i in range(40)]
    ref = np.array([r[2] for r in val])
    cell = dict(id="toy", task="toy", representation="B_causal", method="hash_sgd", budget_bytes=4096)
    out = sc.run_cell(cell, train, val, ref_predictions=ref, featurizer_bytes=123,
                      candidates=sc.CANDIDATES, halving=[[8, 1], [4, 2], [2, 5]],
                      tuning_seed=2026, evaluation_seeds=[42, 123, 456])
    assert out["status"] == "completed" and out["pass_equivalents_tuning"] == 18
    assert len(out["evaluation"]) == 3 and all(len(e["predictions"]) == 40 for e in out["evaluation"])
    assert out["learner_state_bytes"] + 512 <= 4096 and out["featurizer_state_bytes"] == 123
    assert out["selected_candidate"] in range(8)
    assert all(0 <= e["primary"]["accuracy"] <= 1 for e in out["evaluation"])


def test_task_mask_uses_supplied_task_set():
    from experiments.cfpb_streaming_state.data import task_mask
    labels = np.array([0, 1, 2, 3, 2, 3, 1])
    np.testing.assert_array_equal(task_mask(labels, "checking_vs_mortgage", sc.SCOPED_TASKS),
                                  [False, False, True, True, True, True, False])
    np.testing.assert_array_equal(task_mask(labels, "mortgage_vs_rest"), np.ones(7, bool))


def test_analyze_writes_kstar_retention_and_separate_state_tables(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(sc, "SCOPED_TASKS", {"toy": dict(positive=1, negative=2, N=10, validation_N=40)})
    monkeypatch.setattr(sc, "TRACKS", ("B_causal",))
    monkeypatch.setattr(sc, "METHODS", ("hash_sgd",))
    monkeypatch.setattr(sc, "BUDGETS", (4096, 16384))
    monkeypatch.setattr(sc, "N_BOOT", 50)
    y = [1, -1] * 20
    (tmp_path / "preparation.json").write_text(json.dumps(
        {"references": {"toy": {"primary_labels": y, "selected_primary_predictions": y}}}))
    (tmp_path / "cells").mkdir()
    for b, wrong in ((4096, 8), (16384, 0)):
        p = list(y); p[:wrong] = [-v for v in p[:wrong]]
        cell = dict(id=f"toy_B_causal_hash_sgd_b{b}", task="toy", representation="B_causal", method="hash_sgd",
                    budget_bytes=b, selected_config=dict(eta=0.1, schedule="inverse_time"),
                    learner_state_bytes=b - 600, featurizer_state_bytes=47939,
                    evaluation=[dict(seed=s, predictions=p) for s in (42, 123, 456)])
        (tmp_path / "cells" / f"{cell['id']}.json").write_text(json.dumps(cell))
    out = sc.analyze(tmp_path)
    k95 = next(r for r in out["kstar"] if r["threshold"] == 0.95)
    assert k95["k_star_bytes"] == 16384
    st = {r["budget_bytes"]: r for r in out["state"]}
    assert st[4096]["learner_state_bytes"] == 3496 and st[4096]["featurizer_state_bytes"] == 47939
    assert st[4096]["total_persistent_bytes"] == 3496 + 47939
    for name in ("kstar.csv", "retention.csv", "state.csv"):
        assert (tmp_path / name).exists()
