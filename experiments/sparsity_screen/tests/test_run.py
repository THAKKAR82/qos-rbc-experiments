"""Screen runner: gates in protocol order, fixed reasons, immutable outputs."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import sparsity_screen as ss
from experiments.sparsity_screen import run

SMALL = dict(grid=(500, 1_000, 2_000), replicates=2, n_boot=50)


def cold_case(n, r):
    enc = ss.control_case("bounded")(n, r)
    # Train/val use the left column block, test rows only the right one: cold start = 1.
    left = lambda X: sparse.hstack([X, sparse.csr_matrix(X.shape)], format="csr")
    right = lambda X: sparse.hstack([sparse.csr_matrix(X.shape), X], format="csr")
    return ss.Encoded(left(enc.X_train), enc.y_train, left(enc.X_val), enc.y_val, right(enc.X_test), enc.y_test,
                      enc.meta)


def single_class_case(n, r):
    enc = ss.control_case("bounded")(n, r)
    return ss.Encoded(enc.X_train, -np.ones_like(enc.y_train), enc.X_val, enc.y_val, enc.X_test, enc.y_test, enc.meta)


BUILDERS = {"ctrl_bounded": ss.control_case("bounded"), "ctrl_dense_col": ss.control_case("dense_col"),
            "cold": cold_case, "single": single_class_case}
DESCRIPTIONS = {k: dict(dataset="synthetic", encoding=k, label="synthetic control") for k in BUILDERS}


@pytest.fixture(scope="module")
def screened(tmp_path_factory):
    out, _ = run.run_screen(tmp_path_factory.mktemp("s") / "screen", builders=BUILDERS,
                            descriptions=DESCRIPTIONS, **SMALL)
    return out, pd.read_csv(out / "summary.csv").set_index("case")


def test_verdicts_and_audit_only_for_sparsity_pass(screened):
    out, s = screened
    assert s.loc["ctrl_bounded", "sparsity_verdict"] == "PASS" and s.loc["ctrl_dense_col", "sparsity_verdict"] == "FAIL"
    assert (out / "audit/ctrl_bounded.json").exists() and not (out / "audit/ctrl_dense_col.json").exists()
    assert s.loc["ctrl_dense_col", "reason"].startswith("s grows with N (slope 1.00")
    assert math.isnan(s.loc["ctrl_dense_col", "kappa_reg"])


def test_cold_start_case_fails_gate_and_gamma(screened):
    _, s = screened
    row = s.loc["cold"]
    assert row.sparsity_verdict == "PASS" and row.cold_start_at_N_max == 1.0 and row.cold_start_gate == "FAIL"
    assert "cold start 1.000 > 0.10" in row.reason and "theorem inapplicable: gamma_p10 = 0" in row.reason
    assert row.gamma_p10 == 0 and math.isnan(row.log10_F61_proof)
    assert np.isfinite([row.slope, row.ci_lo, row.ci_hi, row.kappa_reg]).all()


def test_single_class_prefix_skips_audit_but_keeps_sparsity(screened):
    _, s = screened
    row = s.loc["single"]
    assert "audit not run: single class" in row.reason
    assert row.sparsity_verdict == "PASS" and np.isfinite(row.cold_start_at_N_max) and math.isnan(row.kappa_reg)


def test_cold_start_columns_precede_audit_columns(screened):
    cols = list(screened[1].reset_index().columns)
    assert cols.index("cold_start_at_N_max") < cols.index("kappa_reg") < cols.index("log10_F61_proof")
    assert cols[-2:] == ["kstar95_kib", "kstar98_kib"]


def test_passing_case_reports_bound_and_model_bytes(screened):
    row = screened[1].loc["ctrl_bounded"]
    assert row.gamma_p10 <= row.gamma_p10_bound and row.full_model_bytes == 8 * row.D_train
    assert row.NoverD_bound_at_N_max <= row.s_row_or_col_at_N_max


def test_outputs_are_immutable(screened):
    with pytest.raises(FileExistsError):
        run.run_screen(screened[0], builders=BUILDERS, descriptions=DESCRIPTIONS, **SMALL)


def test_external_rows_merge_controls_and_elliptic(tmp_path):
    import json
    (tmp_path / "c").mkdir(); (tmp_path / "e").mkdir()
    pd.DataFrame([dict(case="ctrl_bounded_margin", label="synthetic best case", D_train=10, N_max=8,
                       cold_start_mean_at_N_max=0.0, sparsity_verdict="PASS", kappa_reg=1.5)]).to_csv(
        tmp_path / "c/controls_summary.csv", index=False)
    (tmp_path / "e/structure.json").write_text(json.dumps(dict(
        labeled=dict(illicit=1, licit=3), cold_start_step_aligned=dict(rate=1.0), reason="no cross-step edges",
        cross_step_edges=0, n_edges=9)))
    rows = {r["case"]: r for r in run._external_rows(tmp_path / "c", tmp_path / "e")}
    assert rows["ctrl_bounded_margin"]["label"] == "synthetic best case" and rows["ctrl_bounded_margin"]["kappa_reg"] == 1.5
    assert rows["ctrl_bounded_margin"]["cold_start_gate"] == "PASS"
    e = rows["elliptic_structure"]
    assert e["N_max"] == 4 and e["cold_start_gate"] == "FAIL" and e["reason"].startswith("no cross-step edges")
