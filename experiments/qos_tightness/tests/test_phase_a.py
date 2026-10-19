"""Phase (a): grid/seeds plan, checkpoint resume, M* interpolation, fits, bootstrap, local slopes."""
import json
import math

import numpy as np
import pytest


# ---------------------------------------------------------------- plan
def test_plan_grid_and_seeds():
    from qostight.phase_a import M_GRID, seeds_for, S_VALUES
    assert len(M_GRID) == 8 and M_GRID == tuple(10**k for k in range(4, 12))
    assert S_VALUES == (2, 4, 8, 16, 32)
    assert seeds_for(16) == (42, 123, 456, 789, 2026) and seeds_for(32) == (42, 123, 456)


def test_pending_points_skips_checkpointed(tmp_path):
    from qostight.phase_a import all_points, pending_points, append_checkpoint, read_checkpoints
    pts = all_points("primary")
    assert pts[0][1] == 2 and pts[-1][1] == 32  # s ascending
    assert len(pts) == 4 * 5 * 8 + 3 * 8
    append_checkpoint(tmp_path / "w0.jsonl", dict(mode="primary", s=2, seed=42, M=10**4, x=1))
    done = read_checkpoints(tmp_path)
    assert ("primary", 2, 42, 10**4) not in set(pending_points(pts, done))
    assert len(pending_points(pts, done)) == len(pts) - 1


# ---------------------------------------------------------------- M* interpolation
def test_crossing_power_law_exact():
    from qostight.phase_a import crossing_M
    Ms = np.array([1e4, 1e5, 1e6, 1e7])
    errs = 50.0 / Ms  # eps = 50/M -> M* = 50/eps
    r = crossing_M(Ms, errs, 1e-3)
    assert r["M_star"] == pytest.approx(5e4) and r["status"] == "bracketed"


def test_crossing_unreached_and_below_grid():
    from qostight.phase_a import crossing_M
    Ms = np.array([1e4, 1e5, 1e6])
    assert math.isnan(crossing_M(Ms, np.array([1.0, 0.5, 0.2]), 1e-3)["M_star"])
    assert crossing_M(Ms, np.array([1.0, 0.5, 0.2]), 1e-3)["status"] == "not reached"
    assert crossing_M(Ms, np.array([1e-4, 1e-5, 1e-6]), 1e-3)["status"] == "below grid"


# ---------------------------------------------------------------- fits
def _synthetic(alpha=3.0, beta=1.0, seeds=(1, 2, 3)):
    rows = []
    for s in (2, 4, 8, 16):
        for seed in seeds:
            for eps in (1e-2, 1e-3):
                rows.append(dict(s=s, seed=seed, eps=eps, M_star=7.0 * s**alpha / eps**beta))
    return rows


def test_fit_alpha_and_bootstrap_on_exact_power_law():
    from qostight.phase_a import fit_alpha
    r = fit_alpha(_synthetic(alpha=3.0), eps=1e-2, n_boot=200)
    assert r["alpha"] == pytest.approx(3.0) and r["ci95"][0] == pytest.approx(3.0) == pytest.approx(r["ci95"][1])
    assert r["n_s"] == 4


def test_beta_from_two_targets():
    from qostight.phase_a import fit_beta
    r = fit_beta(_synthetic(beta=1.0), n_boot=200)
    assert r["beta"] == pytest.approx(1.0) and r["ci95"][1] - r["ci95"][0] == pytest.approx(0, abs=1e-12)


def test_local_slopes_raw_and_polylog_divided():
    from qostight.phase_a import local_slopes, d23_polylog
    rows = [dict(r, M_star=r["M_star"] * d23_polylog(r["s"], 64, r["eps"])) for r in _synthetic(alpha=5.0)]
    raw = local_slopes(rows, eps=1e-2, n_boot=100)
    div = local_slopes(rows, eps=1e-2, n_boot=100, polylog=lambda s, eps: d23_polylog(s, 64, eps))
    assert [p["pair"] for p in raw] == [(2, 4), (4, 8), (8, 16)]
    assert all(p["slope"] > 5.0 for p in raw)  # polylog inflates raw slopes
    assert all(p["slope"] == pytest.approx(5.0) for p in div)


def test_d23_polylog_matches_theorem_form():
    from qostight.phase_a import d23_polylog
    assert d23_polylog(8, 64, 1e-3) == pytest.approx(math.log(8 * math.log2(64) / 1e-3) ** 4)


# ---------------------------------------------------------------- measurement (tiny size)
def test_measure_point_primary_small_converges_and_counts_samples():
    from qostight.phase_a import measure_point
    lo, hi = (measure_point("primary", 2, 42, M, N=8) for M in (10**4, 10**10))
    for k in ("err_element", "err_index", "err_d23"):
        assert hi[k] < lo[k]
    assert hi["err_index"] < 1e-4 and hi["err_d23"] < 1e-3
    assert lo["samples_element"] == 10**4 and lo["samples_index"] == 10**4 * lo["sign_degree"] * 3
    assert lo["samples_d23"] == lo["amplification_degree"] * lo["samples_be_query"]
    assert lo["sign_verified"] is True


def test_measure_point_a2prime_modes_small():
    from qostight.phase_a import measure_point
    i = measure_point("a2p_i", 2, 42, 10**6, N=8)
    ii = measure_point("a2p_ii", 2, 42, 10**6, N=8)
    assert i["sign_degree"] == ii["sign_degree"] == 101
    assert i["sign_method"].startswith("pyqsp") and ii["sign_method"].startswith("verified")


# ---------------------------------------------------------------- orchestration helpers
def test_shard_round_robin():
    from qostight.phase_a import shard
    pts = list(range(7))
    assert shard(pts, 2) == [[0, 2, 4, 6], [1, 3, 5]] and shard(pts, 1) == [pts]


def test_analyze_recovers_exponents_from_synthetic_checkpoints():
    from qostight.phase_a import analyze, M_GRID, S_VALUES, seeds_for
    rows = []
    for s in S_VALUES:
        for seed in seeds_for(s):
            for M in M_GRID:
                rows.append(dict(mode="primary", s=s, seed=seed, M=M,
                                 samples_element=M, err_element=3e3 * s / M,
                                 samples_index=M * s, err_index=2e3 * s**3 / (M * s),
                                 samples_d23=M * s * s, err_d23=1e2 * s**5 / (M * s * s),
                                 samples_be_query=M, err_rescale=1e3 / M))
    out = analyze(rows, n_boot=100)
    a = {(f["quantity"], f["eps"], f["polylog"]): f["alpha"] for f in out["alpha_fits"]}
    assert a[("element", 1e-2, "raw")] == pytest.approx(1.0)
    assert a[("index", 1e-3, "raw")] == pytest.approx(3.0)
    assert a[("d23", 1e-2, "raw")] == pytest.approx(5.0)
    b = {f["quantity"]: f["beta"] for f in out["beta_fits"]}
    assert b["d23"] == pytest.approx(1.0)
    assert {r["status"] for r in out["mstar"]} == {"bracketed"}


def test_fit_alpha_insufficient_s_is_nan_not_error():
    from qostight.phase_a import fit_alpha
    r = fit_alpha([dict(s=2, seed=1, eps=1e-2, M_star=float("nan"))], eps=1e-2, n_boot=10)
    assert math.isnan(r["alpha"]) and r["status"] == "insufficient s"
