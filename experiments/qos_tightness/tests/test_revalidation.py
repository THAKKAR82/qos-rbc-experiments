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
