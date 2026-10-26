"""Sparsity-scaling screener: does s = max row-or-column nnz stay bounded as N grows?"""
import numpy as np
import pytest
from scipy import sparse

from src import sparsity_screen as ss

GRID = (10_000, 20_000, 40_000, 80_000)


# ---------------------------------------------------------------- synthetic controls
@pytest.mark.parametrize("n", GRID)
def test_bounded_control_is_bounded_at_every_grid_n(n):
    X, y = ss.bounded_control(n, s=16)
    p = ss.sparsity_point(X)
    assert X.shape == (n, 2 * n)
    assert p["s_row"] <= 16 and p["s_col"] <= 16 and p["s_row_or_col"] <= 16
    assert p["s_row"] >= 14  # Only collisions between maps can drop a row below s.
    assert set(np.unique(y).tolist()) <= {-1, 1} and len(y) == n


def test_bounded_control_replicate_seeds_differ():
    a, _ = ss.bounded_control(1_000, seed=[42, 0])
    b, _ = ss.bounded_control(1_000, seed=[42, 1])
    assert (a != b).nnz > 0
    c, _ = ss.bounded_control(1_000, seed=[42, 0])
    assert (a != c).nnz == 0


def test_dense_column_control_has_s_col_equal_n():
    X, _ = ss.dense_column_control(10_000)
    p = ss.sparsity_point(X)
    assert p["s_col"] == p["s_row_or_col"] == 10_000 and p["s_row"] <= 17


# ---------------------------------------------------------------- slope fit and verdict
def _grid(reps=5):
    return np.repeat(np.array(GRID, float), reps), np.tile(np.arange(reps), len(GRID))


def test_slope_recovers_linear_growth():
    N, rep = _grid()
    fit = ss.fit_loglog_slope(N, 3 * N, rep)
    assert fit["slope"] == pytest.approx(1.0)
    assert fit["ci_boot"][0] == pytest.approx(1.0) and fit["n_points"] == 20 and fit["n_replicates"] == 5
    assert ss.classify_slope(*fit["ci_boot"]) == "FAIL"


def test_slope_constant_is_exactly_zero():
    N, rep = _grid()
    fit = ss.fit_loglog_slope(N, np.full_like(N, 16.0), rep)
    assert fit["slope"] == 0.0 and fit["ci_boot"] == (0.0, 0.0) and fit["ci_t"] == (0.0, 0.0)
    assert ss.classify_slope(*fit["ci_boot"]) == "PASS"


def test_slope_bootstrap_ci_covers_noisy_truth():
    N, rep = _grid()
    noise = np.exp(np.random.default_rng(0).normal(0, .1, N.size))
    fit = ss.fit_loglog_slope(N, 5 * N**.5 * noise, rep)
    assert fit["ci_boot"][0] < .5 < fit["ci_boot"][1]
    assert fit["ci_t"][0] < .5 < fit["ci_t"][1]
    assert ss.classify_slope(*fit["ci_boot"]) == "INCONCLUSIVE"


def test_classify_slope_bands_are_strict():
    assert ss.classify_slope(-.1, .19) == "PASS"
    assert ss.classify_slope(-.1, .2) == "INCONCLUSIVE"   # Upper must be strictly < 0.2.
    assert ss.classify_slope(.51, 1.2) == "FAIL"
    assert ss.classify_slope(.5, 1.2) == "INCONCLUSIVE"   # Lower must be strictly > 0.5.
    assert ss.classify_slope(.1, .6) == "INCONCLUSIVE"
    assert ss.classify_slope(float("nan"), float("nan")) == "INCONCLUSIVE"


# ---------------------------------------------------------------- cold start
def test_cold_start_rate_hand_built():
    train = sparse.csr_matrix(np.array([[1., 0., 0.], [0., 1., 0.]]))  # Support: columns 0, 1.
    assert ss.cold_start_rate(train, sparse.csr_matrix([[1., 0., 0.], [0., 1., 1.]])) == 0.0
    assert ss.cold_start_rate(train, sparse.csr_matrix([[1., 0., 0.], [0., 0., 1.]])) == 0.5
    assert ss.cold_start_rate(train, sparse.csr_matrix([[0., 0., 1.], [0., 0., 0.]])) == 1.0


def test_cold_start_ignores_explicit_zeros():
    # Column 1 of train holds only a stored zero, so it is not training support.
    train = sparse.csr_matrix((np.array([1., 0.]), np.array([0, 1]), np.array([0, 1, 2])), shape=(2, 2))
    test = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    assert ss.cold_start_rate(train, test) == 0.5
    with pytest.raises(ValueError):
        ss.cold_start_rate(train, sparse.csr_matrix((0, 2)))
    with pytest.raises(ValueError):
        ss.cold_start_rate(train, sparse.csr_matrix((1, 3)))


# ---------------------------------------------------------------- splits and screening
def test_chronological_split_fits_and_truncates():
    tr, va, te, trunc = ss.chronological_split(10_000, start=0, n_train=1_000)
    assert not trunc and (tr, va, te) == (slice(0, 1000), slice(1000, 1150), slice(1150, 1300))
    tr, va, te, trunc = ss.chronological_split(1_000, start=900, n_train=500)
    assert trunc and tr.start == 900 and tr.stop <= va.start and va.stop <= te.start and te.stop <= 1_000
    assert va.stop - va.start == te.stop - te.start == int(np.ceil(.15 * (tr.stop - tr.start)))
    with pytest.raises(ValueError):
        ss.chronological_split(1_000, start=999, n_train=500)


def test_screen_case_controls_end_to_end():
    pos = ss.screen_case(ss.control_case("bounded"), GRID)
    neg = ss.screen_case(ss.control_case("dense_col"), GRID)
    assert len(pos) == len(neg) == 20 and not pos.truncated.any()
    fit = lambda df: ss.fit_loglog_slope(df.N_nonzero, df.s_row_or_col, df.replicate)
    assert ss.classify_slope(*fit(pos)["ci_boot"]) == "PASS"
    assert ss.classify_slope(*fit(neg)["ci_boot"]) == "FAIL"
    assert (neg.s_col == neg.N_nonzero).all()
    assert pos.cold_start_rate.max() < .01


def test_screen_case_records_truncation():
    df = ss.screen_case(ss.control_case("bounded", max_rows=30_000), (10_000, 40_000), replicates=2)
    assert not df.loc[df.N_requested == 10_000, "truncated"].any()
    big = df[df.N_requested == 40_000]
    assert big.truncated.all() and (big.N_train < 40_000).all()


def test_audit_tool_reports_bounded_control_as_sparse():
    # Spec step 1: the existing step-1 audit (unchanged) must see the control's s <= 16.
    from src import qos_audit as qa
    enc = ss.control_case("bounded")(2_000, 0)
    out = qa.audit_binary(enc.X_train, enc.y_train, enc.X_val, enc.y_val, enc.X_test, enc.y_test, alphas=(1.,))
    assert out["sparsity"]["s_row_or_col"] <= 16 and out["sigma_min_source"] == "shape_D_gt_N"
    row = out["alpha_sweep"][0]
    assert row["lsqr"]["converged"] and row["costs_gamma_p10"]["log10_F61_passes"] is not None
    assert out["test_sparsity"]["zero_rows_on_training_support"] == 0
