"""Phase-1 runner pieces: memory gate, single-config calibration, D.23 validation, dense cross-check."""
import numpy as np
import pytest


def test_memory_gate_thresholds():
    from qostight.resources import gate_ok
    gib = 2**30
    assert gate_ok(available=5 * gib, rss=1 * gib, before_stage=True)
    assert not gate_ok(available=3 * gib, rss=1 * gib, before_stage=True)   # < 4 GiB free to start
    assert gate_ok(available=3 * gib, rss=1 * gib, before_stage=False)
    assert not gate_ok(available=1.5 * gib, rss=1 * gib, before_stage=False)  # < 2 GiB free
    assert not gate_ok(available=8 * gib, rss=4.5 * gib, before_stage=False)  # RSS > 4 GiB


def test_calibrate_one_small_config_reports_time_memory_and_size():
    from qostight.phase1 import calibrate_one
    r = calibrate_one(N=8, s=2, unit_num_samples=10**5)
    assert r["status"] == "ok" and r["N"] == r["D"] == 8 and r["s"] == 2
    assert r["seconds"] > 0 and r["peak_rss_bytes"] > 0
    assert r["index_oracle_dim"] == 8 * 2 * 8
    assert r["be_error_spec"] < 1e-1


def test_d23_validation_small_passes_stated_criteria():
    from qostight.phase1 import d23_validation
    rows = d23_validation(N=8, s=2, M_grid=(10**4, 10**6, 10**10), poly_eps=1e-4, seed=42)
    exact = [r for r in rows if r["input"] == "exact A/s"][0]
    assert exact["error_spec"] <= 1e-4 and exact["pass"]
    sk = [r for r in rows if r["input"] == "sketched" and r["path"] == "primary: Theorem D.23 amplification"]
    errs = [r["error_spec"] for r in sorted(sk, key=lambda r: r["unit_num_samples"])]
    assert errs[0] > errs[-1] and errs[-1] <= 1e-3
    assert {r["path"] for r in rows if r["input"] == "sketched"} == {
        "primary: Theorem D.23 amplification", "optimistic: reference convention"}


def test_dense_crosscheck_on_sketched_block_encoding_small():
    from qostight.phase1 import dense_crosscheck
    rows = dense_crosscheck(N=8, s=2, seed=42)
    assert {r["polynomial"] for r in rows} == {"D.23 amplification", "1/x pseudo-inverse"}
    for r in rows:
        assert r["max_abs_diff"] < 1e-6, r
