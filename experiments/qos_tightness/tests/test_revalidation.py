"""Re-validation pieces after the sign fix: a2' comparisons and calibration provenance."""
import numpy as np


def test_a2prime_comparison_rows_small():
    from qostight.phase1 import a2prime_comparison
    rows = a2prime_comparison(N=8, s=2, unit_num_samples=10**10)
    labels = [r["comparison"] for r in rows]
    assert labels == ["a2' (i): reference pyqsp path, failure recorded as found",
                      "a2' (ii): verified-bounded, fixed degree"]
    for r in rows:
        assert r["sign_degree"] == 101 and r["N"] == 8 and r["s"] == 2
        assert {"angles_verified", "angle_max_dev", "sign_achieved_error", "d23_error_spec"} <= set(r)
    assert rows[1]["angles_verified"] is True


def test_calibrate_one_records_sign_method_and_eps():
    from qostight.phase1 import calibrate_one
    r = calibrate_one(N=8, s=2, unit_num_samples=10**5)
    assert r["sign_method"] == "verified Chebyshev erf (primary)"
    assert r["sign_eps"] < 1e-4


def test_phase_a_budget_estimate_and_missing_detection():
    from qostight.run_phase1 import phase_a_budget, PHASE_A_PLAN
    calib = [dict(N=64, s=s, status="ok", seconds=10.0) for s in PHASE_A_PLAN["s_values"]]
    b = phase_a_budget(calib)
    n = PHASE_A_PLAN["M_values"] * len(PHASE_A_PLAN["seeds"])
    assert b["primary_sweep_hours"] == 10.0 * len(PHASE_A_PLAN["s_values"]) * n / 3600
    assert b["missing_s"] == [] and b["within_budget"] is True
    b2 = phase_a_budget(calib[:-1])
    assert b2["missing_s"] == [32] and b2["within_budget"] is False


def test_retime_parent_does_not_import_jax():
    import subprocess, sys
    from pathlib import Path
    code = ("import sys; from qostight import run_phase1; "
            "print('jax' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "False", out.stderr[-500:]


def test_retime_pairs_cover_phase_a_plan_and_s32_at_128():
    from qostight.run_phase1 import RETIME_PAIRS, PHASE_A_PLAN
    assert RETIME_PAIRS == [(64, s) for s in PHASE_A_PLAN["s_values"]] + [(128, 32)]
