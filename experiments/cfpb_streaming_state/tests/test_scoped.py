"""Scoped classical benchmark: plan, schedules, budgets, successive halving, paired bootstrap k*.

Light synthetic fixtures only (no CFPB data, no full encoder vocabulary load).
"""
import math
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state import scoped as sc
from experiments.cfpb_streaming_state.methods import (
    ALLOCATOR_VERSION, Learner, allocation_metadata, budgeted,
)


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
@pytest.mark.parametrize("kib", [4, 16, 32, 64, 128, 512])
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
    assert out["allocator_version"] == ALLOCATOR_VERSION
    assert len(out["allocations"]["tuning"]) == 8
    assert len(out["allocations"]["evaluation"]) == 3
    for phase in out["allocations"].values():
        assert all(a["state_bound_bytes"] + 512 <= 4096 for a in phase.values())
    assert all(0 <= e["primary"]["accuracy"] <= 1 for e in out["evaluation"])


def test_task_mask_uses_supplied_task_set():
    from experiments.cfpb_streaming_state.data import task_mask
    labels = np.array([0, 1, 2, 3, 2, 3, 1])
    np.testing.assert_array_equal(task_mask(labels, "checking_vs_mortgage", sc.SCOPED_TASKS),
                                  [False, False, True, True, True, True, False])
    np.testing.assert_array_equal(task_mask(labels, "mortgage_vs_rest"), np.ones(7, bool))


def test_analyze_writes_kstar_retention_and_separate_state_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SCOPED_TASKS", {"toy": dict(positive=1, negative=2, N=10, validation_N=40)})
    monkeypatch.setattr(sc, "TRACKS", ("B_causal",))
    monkeypatch.setattr(sc, "METHODS", ("hash_sgd",))
    monkeypatch.setattr(sc, "BUDGETS", (4096, 16384))
    monkeypatch.setattr(sc, "N_BOOT", 50)
    y = [1, -1] * 20
    (tmp_path / "preparation.json").write_text(json.dumps(
        {"spools": {}, "references": {"toy": {"primary_labels": y, "selected_primary_predictions": y}}}))
    (tmp_path / "cells").mkdir()
    for b, wrong in ((4096, 8), (16384, 0)):
        p = list(y); p[:wrong] = [-v for v in p[:wrong]]
        cell = dict(id=f"toy_B_causal_hash_sgd_b{b}", task="toy", representation="B_causal", method="hash_sgd",
                    budget_bytes=b, selected_config=dict(eta=0.1, schedule="inverse_time"),
                    learner_state_bytes=b - 600, featurizer_state_bytes=47939,
                    evaluation=[dict(seed=s, predictions=p) for s in (42, 123, 456)])
        _stamp_cell(cell)
        (tmp_path / "cells" / f"{cell['id']}.json").write_text(json.dumps(cell))
    _write_run_provenance(tmp_path)
    out = sc.analyze(tmp_path)
    k95 = next(r for r in out["kstar"] if r["threshold"] == 0.95)
    assert k95["k_star_bytes"] == 16384
    st = {r["budget_bytes"]: r for r in out["state"]}
    assert st[4096]["learner_state_bytes"] == 3496 and st[4096]["featurizer_state_bytes"] == 47939
    assert st[4096]["total_persistent_bytes"] == 3496 + 47939
    for name in ("kstar.csv", "retention.csv", "state.csv"):
        assert (tmp_path / name).exists()


def _stamp_cell(cell):
    a = allocation_metadata(budgeted(cell["method"], cell["budget_bytes"], 10))
    cell.update(status="completed", allocator_version=ALLOCATOR_VERSION, run_id="synthetic",
                source_sha256=sc._source_hashes(),
                allocations={"tuning": {"0": a}, "evaluation": {"42": a}})
    return cell


def _write_run_provenance(out):
    sc.atomic_json(out / "run_plan.json", sc.plan())
    sc.atomic_json(out / "manifest.json", dict(
        allocator_version=ALLOCATOR_VERSION, source_sha256=sc._source_hashes(),
        environment=sc._environment(), run_id="synthetic", status="interrupted",
        preparation_sha256=sc.sha256(out / "preparation.json"),
        completed_cells_sha256={p.stem: sc.sha256(p) for p in (out / "cells").glob("*.json")}))


def _files(out):
    return {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["legacy", "source", "environment", "cell_version", "cell_hash", "orphan"])
def test_incompatible_resume_and_analysis_fail_before_writes(tmp_path, monkeypatch, failure):
    (tmp_path / "cells").mkdir()
    sc.atomic_json(tmp_path / "preparation.json", {"spools": {}})
    cell = _stamp_cell(dict(sc.plan()["cells"][0]))
    p = tmp_path / "cells" / f"{cell['id']}.json"
    sc.atomic_json(p, cell)
    _write_run_provenance(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if failure == "legacy":
        del manifest["allocator_version"]
    elif failure == "source":
        manifest["source_sha256"] = {}
    elif failure == "environment":
        manifest["environment"] = {}
    elif failure == "cell_version":
        cell["allocator_version"] = "legacy"
        sc.atomic_json(p, cell)
        manifest["completed_cells_sha256"][cell["id"]] = sc.sha256(p)
    elif failure == "cell_hash":
        p.write_text(p.read_text() + "\n")
    elif failure == "orphan":
        manifest["completed_cells_sha256"] = {}
    sc.atomic_json(tmp_path / "manifest.json", manifest)
    before = _files(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible resume reached preparation/training/resource preflight")
    monkeypatch.setattr(sc, "prepare_stage", forbidden)
    monkeypatch.setattr(sc, "run_cell", forbidden)
    monkeypatch.setattr("src.cfpb_prepare.check_resources", forbidden)
    with pytest.raises(ValueError):
        sc.run(tmp_path)
    with pytest.raises(ValueError):
        sc.analyze(tmp_path)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("stage", ["preflight", "preparation", "encoding", "cell", "analysis"])
@pytest.mark.parametrize("exception", [MemoryError, ValueError])
def test_failure_marks_interrupted_and_preserves_completed_cells(tmp_path, monkeypatch, stage, exception):
    monkeypatch.setattr(sc, "ROOT", tmp_path)
    monkeypatch.setattr(sc, "SCOPED_TASKS", {"toy": dict(N=2, validation_N=2)})
    monkeypatch.setattr(sc, "TRACKS", ("B_causal",))
    monkeypatch.setattr(sc, "METHODS", ("hash_sgd",))
    monkeypatch.setattr(sc, "BUDGETS", (4096, 16384))
    monkeypatch.setattr(sc.subprocess, "check_output", lambda *a, **k: b"synthetic-test-commit")
    rows = [(keys([1]), np.array([1.0]), 1), (keys([2]), np.array([1.0]), -1)]
    def fail():
        raise exception("synthetic accounting limit: measured + 512 exceeds budget")
    def preflight(**kwargs):
        if stage == "preflight":
            fail()
    def prepare(out):
        if stage == "preparation":
            fail()
        spools = {}
        for role in ("train", "validation"):
            p = out / "data_cache" / f"{role}.jsonl"
            p.write_text("synthetic spool")
            p.with_suffix(".offsets.npy").write_bytes(b"synthetic offsets")
            spools[role] = dict(path=str(p.relative_to(out)), N=2, sha256=sc.sha256(p),
                                offsets_sha256=sc.sha256(p.with_suffix(".offsets.npy")))
        sc.atomic_json(out / "preparation.json", dict(spools={"toy": spools}, references={
            "toy": dict(primary_labels=[1, -1], selected_primary_predictions=[1, -1])}))
    class Encoder:
        def __init__(self, track):
            pass
        def state_inventory(self):
            return []
    def encode(*args):
        if stage == "encoding":
            fail()
        return rows
    saved = {}
    def cell(c, *args, **kwargs):
        if stage == "cell" and c["budget_bytes"] == 16384:
            run_dir = next((tmp_path / "results/classical_streaming").glob("scoped_*"))
            saved.update({p: p.read_bytes() for p in (run_dir / "cells").glob("*.json")})
            fail()
        return dict(_stamp_cell(dict(c)), seconds=0.0)
    def analyze(out):
        sc.validate_resume(out)
        if stage == "analysis":
            fail()
    monkeypatch.setattr("src.cfpb_prepare.check_resources", preflight)
    monkeypatch.setattr("experiments.cfpb_streaming_state.data.Encoder", Encoder)
    monkeypatch.setattr(sc, "prepare_stage", prepare)
    monkeypatch.setattr(sc, "_encode_rows", encode)
    monkeypatch.setattr(sc, "run_cell", cell)
    monkeypatch.setattr(sc, "analyze", analyze)
    with pytest.raises(exception, match="synthetic accounting limit"):
        sc.run()
    out = next((tmp_path / "results/classical_streaming").glob("scoped_*"))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["interruption_type"] == exception.__name__
    assert "synthetic accounting limit" in manifest["interruption"]
    assert "during" in manifest["interruption"] and manifest["wall_seconds"] >= 0
    assert {p: p.read_bytes() for p in saved} == saved
    if stage == "cell":
        assert len(saved) == len(manifest["completed_cells_sha256"]) == 1
        assert manifest["interrupted_cell"] == sc.plan()["cells"][1]["id"]
    # A matching interrupted run is eligible for resume; validation itself is read-only.
    before = _files(out)
    assert sc.validate_resume(out) == manifest
    assert _files(out) == before
    if stage == "cell":
        resumed_cells = []
        def finish(c, *args, **kwargs):
            resumed_cells.append(c["id"])
            return dict(_stamp_cell(dict(c)), seconds=0.0)
        monkeypatch.setattr(sc, "run_cell", finish)
        assert sc.run(out) == out
        after = json.loads((out / "manifest.json").read_text())
        assert after["status"] == "cells_completed" and "interruption" not in after
        assert resumed_cells == [sc.plan()["cells"][1]["id"]]
        assert {p: p.read_bytes() for p in saved} == saved
        assert len(after["completed_cells_sha256"]) == 2


def test_final_accounting_failure_reports_cell_state_margin_and_budget(monkeypatch):
    # Deliberately bypass allocation to exercise the unchanged final check.
    def oversized_dimensions(*args, **kwargs):
        return Learner("awm", 3000, depth=3, capacity=512, eta=.1, lam=.5, budget=65536)
    monkeypatch.setattr(sc, "budgeted", oversized_dimensions)
    rows = [(keys([1, 2]), np.array([.6, .8]), y) for y in (1, -1)]
    cell = dict(id="synthetic_64k_failure", method="awm", budget_bytes=65536)
    with pytest.raises(MemoryError, match=r"synthetic_64k_failure: learner state \d+ \+ margin 512 "
                                         r"exceeds budget 65536 bytes"):
        sc.run_cell(cell, rows, rows, ref_predictions=[1, -1], featurizer_bytes=0,
                    candidates=[dict(eta=.1, schedule="inverse_time")],
                    halving=[[1, 1]], evaluation_seeds=[42])
