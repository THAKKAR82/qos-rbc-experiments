"""QOS applicability audit helpers: theorem conventions, not sklearn defaults."""
import hashlib
import json
import math

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from src.machine_size import max_row_or_col_sparsity, max_row_sparsity, qos_svm_machine_size
from src import qos_audit as qa


def random_sparse(n=40, d=120, density=.08, seed=42):
    rng = np.random.default_rng(seed)
    X = sparse.random(n, d, density=density, format='csr', random_state=rng, dtype=np.float64)
    y = np.where(rng.random(n) < .5, -1., 1.)
    return X, y


# ---------------------------------------------------------------- sparsity
@pytest.mark.parametrize("constructor", [np.array, sparse.csr_matrix, sparse.csc_matrix,
                                         sparse.coo_matrix, sparse.csr_array])
def test_max_row_or_col_sparsity_hand_built(constructor):
    X = constructor([[1., 0., 2.], [3., 0., 0.], [4., 0., 0.], [5., 0., 0.]])
    assert max_row_sparsity(X) == 2
    assert max_row_or_col_sparsity(X) == 4  # Column 0 has four nonzeros.
    assert max_row_or_col_sparsity(constructor([[1., 1., 1.], [0., 1., 0.]])) == 3  # Row wins.


def test_max_row_or_col_sparsity_counts_mathematical_nonzeros(monkeypatch):
    # Column 0 holds a cancelling duplicate and column 1 an explicit zero.
    X = sparse.csr_matrix((np.array([2., -2., 0., 5., 1.]), np.array([0, 0, 1, 2, 2]),
                           np.array([0, 3, 4, 5])), shape=(3, 3))
    before = (X.data.copy(), X.indices.copy(), X.indptr.copy())
    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse accounting must not densify")
    monkeypatch.setattr(sparse.csr_matrix, 'toarray', forbidden)
    assert max_row_or_col_sparsity(X) == 2  # Column 2 only.
    for actual, expected in zip((X.data, X.indices, X.indptr), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (3, 4)])
def test_max_row_or_col_sparsity_empty(shape):
    assert max_row_or_col_sparsity(np.zeros(shape)) == 0
    assert max_row_or_col_sparsity(sparse.csr_matrix(shape)) == 0


def test_sparsity_profile_and_nonzero_support():
    X = sparse.csr_matrix(np.array([[1., 0., 2., 0.], [0., 0., 0., 0.],
                                    [3., 0., 0., 0.], [4., 0., 0., 0.]]))
    rows, cols = qa.nonzero_support(X)
    np.testing.assert_array_equal(rows, [True, False, True, True])
    np.testing.assert_array_equal(cols, [True, False, True, False])
    p = qa.sparsity_profile(X)
    assert p['s_row_only'] == 2 and p['s_row_or_col'] == 3
    assert p['empty_rows'] == 1 and p['empty_cols'] == 2
    assert p['col_nnz']['max'] == 3 and p['row_nnz']['max'] == 2


def test_max_df_sweep_hand_built():
    # df per column over N=4 rows: [4, 2, 1, 1]; nnz = 8.
    X = sparse.csr_matrix(np.array([[1., 1., 1., 0.], [1., 1., 0., 0.],
                                    [1., 0., 0., 1.], [1., 0., 0., 0.]]))
    sweep = {r['max_df']: r for r in qa.max_df_sweep(X, fractions=(1., .5, .25, .1))}
    assert sweep[1.]['features_kept'] == 4 and sweep[1.]['nnz_share_lost'] == 0
    assert sweep[1.]['s_row_or_col'] == 4 and sweep[1.]['s_row_only'] == 3
    # df <= 2 keeps columns 1-3: loses 4 of 8 nonzeros; row 0 keeps two.
    assert sweep[.5]['features_kept'] == 3 and sweep[.5]['nnz_share_lost'] == .5
    assert sweep[.5]['s_row_or_col'] == 2 and sweep[.5]['s_row_only'] == 2
    assert sweep[.25]['features_kept'] == 2 and sweep[.25]['nnz_share_lost'] == .75
    assert sweep[.25]['s_row_or_col'] == 1
    assert sweep[.1]['features_kept'] == 0 and sweep[.1]['nnz_share_lost'] == 1
    assert sweep[.1]['s_row_or_col'] == sweep[.1]['s_row_only'] == 0


def test_zipf_slope_recovers_power_law():
    df = np.round(1e6 * np.arange(1, 3001) ** -1.0).astype(int)
    assert qa.zipf_slope(df) == pytest.approx(-1.0, abs=.01)
    assert math.isnan(qa.zipf_slope(np.array([5, 4, 3])))


# ---------------------------------------------------------------- normalization / kappa
def test_spectral_norm_matches_dense():
    X, _ = random_sparse()
    assert qa.spectral_norm(X) == pytest.approx(np.linalg.norm(X.toarray(), 2), rel=1e-10)


def test_kappa_reg_formula():
    assert qa.kappa_reg(.25) == pytest.approx(math.sqrt(1.25 / .25))
    assert qa.kappa_reg(.25, .5) == pytest.approx(math.sqrt(1.25 / .5))
    assert qa.kappa_reg(.01) <= math.sqrt(1 + 1 / .01) + 1e-12  # Eq. F50 bound.
    with pytest.raises(ValueError):
        qa.kappa_reg(0.)  # Rank-deficient and unregularized: unbounded.


def test_kappa_reg_matches_augmented_matrix_when_D_exceeds_N():
    # Lemma F.14: kappa_reg = ratio of extreme singular values of [X; sqrt(lam) I]
    # with ||X|| = 1. For D > N the D x D Gram has zero eigenvalues (Eq. F50).
    X, _ = random_sparse(n=10, d=30, density=.5)
    Xs = X.toarray() / np.linalg.norm(X.toarray(), 2)
    lam = .3
    A = np.vstack([Xs, math.sqrt(lam) * np.eye(30)])
    sv = np.linalg.svd(A, compute_uv=False)
    sigma_min = qa.sigma_min_scaled(n_rows=10, n_cols=30)
    assert sigma_min == 0.
    assert qa.kappa_reg(lam, sigma_min) == pytest.approx(sv.max() / sv.min(), rel=1e-10)
    with pytest.raises(ValueError):
        qa.sigma_min_scaled(n_rows=30, n_cols=30)


def test_alpha_to_lambda_ridge_matches_sklearn():
    X, y = random_sparse()
    X_test, _ = random_sparse(n=25, seed=7)
    sigma = qa.spectral_norm(X)
    for alpha in (.1, 1., 10., 100.):
        lam = qa.alpha_to_lambda(alpha, sigma)
        w, info = qa.ridge_no_intercept(X / sigma, y, lam)
        assert info['converged']
        reference = Ridge(alpha=alpha, fit_intercept=False, tol=1e-12).fit(X, y)
        # Unscaled weights are w / sigma; decision values must agree, not only signs.
        np.testing.assert_allclose(X_test @ w / sigma, reference.predict(X_test),
                                   rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(w / sigma, reference.coef_, rtol=1e-6, atol=1e-9)


def test_residual_ratio_is_squared_eq_f49():
    X, y = random_sparse(n=30, d=12, density=.4)
    Xs = X.toarray() / np.linalg.norm(X.toarray(), 2)
    lam = .2
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(12), Xs.T @ y)
    expected = (np.sum((Xs @ w - y) ** 2) + lam * w @ w) / (y @ y)
    assert qa.residual_ratio(sparse.csr_matrix(Xs), y, w, lam) == pytest.approx(expected, rel=1e-12)
    # Eq. F68: same quantity as ||(I - A A^+) b||^2 for the augmented system.
    A = np.vstack([Xs, math.sqrt(lam) * np.eye(12)]) / math.sqrt(1 + lam)
    b = np.concatenate([y, np.zeros(12)]) / np.linalg.norm(y)
    assert np.sum((b - A @ np.linalg.pinv(A) @ b) ** 2) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------- margins
def test_inf_rescaling_changes_margins_not_predictions():
    X, y = random_sparse()
    X_test, _ = random_sparse(n=50, seed=3)
    w, _ = qa.ridge_no_intercept(X / qa.spectral_norm(X), y, .05)
    m = qa.margins(X_test, w)
    nonzero = np.diff(sparse.csr_matrix(X_test).indptr) > 0
    assert not np.allclose(m['as_is'][nonzero], m['inf_rescaled'][nonzero])
    assert (m['inf_rescaled'] >= m['as_is'] - 1e-15).all()  # Entries were already <= 1.
    rescaled = sparse.diags(1 / np.where(m['row_max'] > 0, m['row_max'], 1.)) @ X_test
    assert abs(rescaled).max() == pytest.approx(1.)
    np.testing.assert_array_equal(qa.predict_sign(X_test, w), qa.predict_sign(rescaled, w))
    np.testing.assert_allclose(np.abs(rescaled @ w) / np.linalg.norm(w), m['inf_rescaled'])


def test_margin_summary_coverage():
    g = np.array([.5, .2, .05, .02, .001])
    yt = np.array([1, 1, -1, -1, 1]); yp = np.array([1, -1, -1, -1, -1])
    s = qa.margin_summary(g, yt, yp, thresholds=(.03,))
    assert s['coverage@0.03'] == .6 and s['acc@0.03'] == pytest.approx(2 / 3)
    assert s['q0'] == .001


# ---------------------------------------------------------------- accounting
def test_a9_equals_qos_svm_machine_size():
    for N, D, s in [(13459, 50000, 8776), (4904, 43446, 2916), (45211, 51, 16)]:
        costs = qa.qos_costs(N, D, s, kappa=10., gamma=.05, m=1000)
        assert costs['A9_qubits_single_prediction'] == qos_svm_machine_size(N, D, s)


def test_qos_costs_prefactors_and_zero_margin():
    N, D, s, k, g, m, delta = 100, 1024, 4, 3., .1, 256, .1
    c = qa.qos_costs(N, D, s, k, g, m=m, delta=delta)
    lD, lm, li = 10., 8., math.log2(10)
    S = s * lD**2 * (min(s * lD, lm) + li**2.5) / g**2 + math.log2(s * D) * math.log2(k) + math.log2(k)**2.5
    passes = s**7 * k**4 / (g**4 * delta) * min(s**2 * lD**2, lm**2)
    assert c['log10_F60_qubits_without_b'] == pytest.approx(math.log10(S))
    assert c['log10_F61_passes'] == pytest.approx(math.log10(passes))
    zero = qa.qos_costs(N, D, s, k, 0., m=m)
    assert zero['log10_F60_qubits_without_b'] is None and zero['log10_F61_passes'] is None


def test_binary_tasks_and_labels():
    tasks = qa.binary_tasks(('A', 'B', 'C', 'D'))
    assert len(tasks) == 10
    assert sum(t['negative'] is None for t in tasks) == 4
    labels = np.array([0, 1, 2, 3, 1, 0])
    mask, y = qa.task_labels(labels, positive=1, negative=3)
    np.testing.assert_array_equal(mask, [False, True, False, True, True, False])
    np.testing.assert_array_equal(y, [1, -1, 1])
    mask, y = qa.task_labels(labels, positive=0, negative=None)
    assert mask.all() and (y == np.array([1, -1, -1, -1, -1, 1])).all()


def test_audit_binary_rejects_bad_labels():
    X, y = random_sparse()
    with pytest.raises(ValueError):
        qa.audit_binary(X, (y > 0).astype(int), X, y, X, y)


def test_audit_binary_end_to_end_small():
    X, y = random_sparse(n=60, d=200, density=.05)
    Xv, yv = random_sparse(n=20, d=200, density=.05, seed=5)
    Xt, yt = random_sparse(n=30, d=200, density=.05, seed=6)
    out = qa.audit_binary(X, y, Xv, yv, Xt, yt, alphas=(1., 10.))
    assert out['N_full'] == 60 and out['D_full'] == 200
    assert out['D_nonzero'] <= 200 and out['N_nonzero'] <= 60
    for row in out['alpha_sweep']:
        assert row['lsqr']['converged']
        assert 0 < row['residual_ratio'] <= 1
        assert row['kappa_reg'] == pytest.approx(qa.kappa_reg(row['lam']))
        assert row['costs_gamma_min']['A9_qubits_single_prediction'] == qos_svm_machine_size(
            out['N_nonzero'], out['D_nonzero'], out['sparsity']['s_row_or_col'])


# ---------------------------------------------------------------- F61 decomposition
def test_f61_decomposition_sums_to_reported_passes():
    for N, D, s, k, g, m, delta in [(100, 1024, 4, 3., .1, 256, .1),
                                    (8472, 47138, 5714, 5.42, .0556, 3792, .1),
                                    (10, 64, 2, 1.5, .3, 2, .05)]:
        inputs = qa.qos_costs(N, D, s, k, g, m=m, delta=delta)
        d = qa.f61_decomposition(inputs['inputs'])
        parts = [d[f'log10_{f}'] for f in qa.F61_FACTORS]
        assert d['log10_total'] == pytest.approx(sum(parts), abs=1e-12)
        assert d['log10_total'] == pytest.approx(inputs['log10_F61_passes'], abs=1e-9)


def test_f61_decomposition_factors_and_s1_counterfactual():
    d = qa.f61_decomposition(dict(N=100, D=1024, s=4, kappa_reg=3., gamma_test=.1, m=256, delta=.1))
    assert d['log10_s^7'] == pytest.approx(7 * math.log10(4))
    assert d['log10_kappa^4'] == pytest.approx(4 * math.log10(3))
    assert d['log10_gamma^-4'] == pytest.approx(4.)
    assert d['log10_delta^-1'] == pytest.approx(1.)
    assert d['log10_min_term'] == pytest.approx(math.log10(min(16 * 100, 64)))  # log2 D=10, log2 m=8
    # s = 1: s^7 vanishes and the min term becomes min{log^2 D, log^2 m}.
    assert d['log10_total_s1'] == pytest.approx(4 * math.log10(3) + 4 + 1 + math.log10(64))
    assert d['min_term_branch'] == 'log2^2 m'
    with pytest.raises(ValueError):
        qa.f61_decomposition(dict(N=1, D=4, s=1, kappa_reg=1., gamma_test=0., m=2, delta=.1))


def test_derive_f61_decomposition_reads_json_only(tmp_path):
    def task(task_id, alphas):
        sweep = []
        for a in alphas:
            costs = qa.qos_costs(50, 300, 7, 2. + a, .05, m=40)
            sweep.append({'alpha': a, 'costs_gamma_p10': costs})
        return {'task': {'task_id': task_id, 'kind': 'pair'}, 'alpha_sweep': sweep}
    for t in ('a__vs__b', 'c__vs__rest'):
        (tmp_path / f'{t}.json').write_text(json.dumps(task(t, [1., 10.])))
    (tmp_path / 'manifest.json').write_text(json.dumps({'tasks': ['a__vs__b', 'c__vs__rest']}))
    table = qa.derive_f61_decomposition(tmp_path)
    assert len(table) == 4 and set(table['task_id']) == {'a__vs__b', 'c__vs__rest'}
    row = table[(table.task_id == 'a__vs__b') & (table.alpha == 10.)].iloc[0]
    assert row['log10_total'] == pytest.approx(qa.qos_costs(50, 300, 7, 12., .05, m=40)['log10_F61_passes'])
    assert row['gamma'] == .05 and row['s'] == 7


def test_write_derived_manifest_is_immutable(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'run_id': 'r1'}))
    (tmp_path / 'figures').mkdir()
    fig = tmp_path / 'figures' / 'f.png'
    fig.write_bytes(b'abc')
    first = qa.write_derived_manifest(tmp_path, ['figures/f.png'])
    assert first['artifacts_sha256']['figures/f.png'] == hashlib.sha256(b'abc').hexdigest()
    assert first['source_run_id'] == 'r1'
    assert qa.write_derived_manifest(tmp_path, ['figures/f.png']) == first  # Verified, unchanged.
    fig.write_bytes(b'abd')
    with pytest.raises(ValueError):
        qa.write_derived_manifest(tmp_path, ['figures/f.png'])


# ---------------------------------------------------------------- D <= N (rank-checked sigma_min)
def one_hot_with_numeric(n=300, groups=(3, 4, 2), seed=42):
    rng = np.random.default_rng(seed)
    blocks = [np.eye(g)[rng.integers(0, g, n)] for g in groups]  # each block sums to ones
    return np.hstack(blocks + [rng.standard_normal((n, 2))])


def test_svd_sigma_min_full_rank_tall():
    X = np.random.default_rng(42).standard_normal((200, 12))
    info = qa.svd_sigma_min(X)
    sv = np.linalg.svd(X, compute_uv=False)
    assert info['rank'] == 12 == np.linalg.matrix_rank(X)
    assert info['sigma_min'] == pytest.approx(sv[-1]) == info['sigma_min_raw']
    assert info['sigma_max'] == pytest.approx(sv[0])
    assert info['tol'] == pytest.approx(sv[0] * max(X.shape) * np.finfo(np.float64).eps)


def test_svd_sigma_min_rank_deficient_one_hot():
    X = one_hot_with_numeric()
    info = qa.svd_sigma_min(sparse.csr_matrix(X))
    D = X.shape[1]
    assert info['rank'] == D - 2 == np.linalg.matrix_rank(X)  # three groups: two dependencies
    assert info['sigma_min'] == 0.0
    assert 0 <= info['sigma_min_raw'] <= info['tol'] < info['smallest_nonzero']
    with pytest.raises(ValueError):
        qa.svd_sigma_min(sparse.random(10, qa.DENSE_SVD_MAX_D + 1, density=.01, format='csr'))


def test_audit_binary_D_below_N_full_rank_uses_svd_sigma_min():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((120, 6)); y = np.where(X @ rng.standard_normal(6) > 0, 1, -1)
    Xt = rng.standard_normal((40, 6)); yt = np.where(Xt[:, 0] > 0, 1, -1)
    out = qa.audit_binary(X, y, None, None, Xt, yt, alphas=(1.,))
    sv = np.linalg.svd(X, compute_uv=False)
    assert out['sigma_min_scaled'] == pytest.approx(sv[-1] / sv[0], rel=1e-8)
    assert out['sigma_min_source'] == 'dense_svd_rank_check'
    row = out['alpha_sweep'][0]
    Xs = X / sv[0]
    A = np.vstack([Xs, math.sqrt(row['lam']) * np.eye(6)])
    s = np.linalg.svd(A, compute_uv=False)
    assert row['kappa_reg'] == pytest.approx(s.max() / s.min(), rel=1e-8)
    assert row['validation'] is None and row['test']['accuracy'] >= 0


def test_audit_binary_D_below_N_rank_deficient_kappa():
    X = one_hot_with_numeric(n=200)
    rng = np.random.default_rng(1)
    y = np.where(rng.random(200) < .4, 1, -1)
    out = qa.audit_binary(X, y, None, None, X[:50], y[:50], alphas=(10.,))
    assert out['svd']['rank'] == X.shape[1] - 2 and out['sigma_min_scaled'] == 0.0
    row = out['alpha_sweep'][0]
    assert row['kappa_reg'] == pytest.approx(math.sqrt((1 + row['lam']) / row['lam']))
    assert out['sparsity']['s_row_or_col'] == 200  # dense numeric columns: s = N


def test_audit_binary_D_above_N_path_unchanged():
    X, y = random_sparse(n=30, d=200, density=.1)
    out = qa.audit_binary(X, y, X, y, X, y, alphas=(1.,))
    assert out['sigma_min_source'] == 'shape_D_gt_N' and out['svd'] is None
    assert out['sigma_min_scaled'] == 0.0


def test_analytic_dense_column_s7():
    row = qa.analytic_dense_column_row('PaySim', 6_362_620)
    assert row['s_row_or_col'] == 6_362_620 and row['s_over_N'] == 1.0
    assert row['log10_s^7'] == pytest.approx(7 * math.log10(6_362_620)) == pytest.approx(47.63, abs=.01)
    assert row['status'].startswith('analytic, not measured')


def test_family_summary_selected_and_cost_min_alpha():
    import pandas as pd
    summary = pd.DataFrame(dict(task_id=['a', 'a', 'b', 'b'], alpha=[1., 10.] * 2, N_nonzero=[10] * 4,
                                s_row_only=[3] * 4, s_row_or_col=[8, 8, 6, 6], kappa_reg=[5., 2., 6., 3.],
                                gamma_p10=[.1, .2, .05, .3], val_accuracy=[.9, .8, .7, .7],
                                test_accuracy=[.6, .5, .9, .8]))
    decomposition = pd.DataFrame(dict(task_id=['a', 'a', 'b', 'b'], alpha=[1., 10.] * 2,
                                      log10_total=[30., 25., 28., 29.], log10_total_s1=[9., 7., 8., 10.]))
    row = qa.family_summary(summary, decomposition, dataset='X', family='f', status='measured',
                            select_by='val_accuracy')
    assert row['n_tasks'] == 2 and row['s_over_N_max'] == .8 and row['kappa_reg_min'] == 2.
    # Per-task selection: a -> alpha 1 (0.9); b ties at 0.7 -> larger alpha 10.
    assert row['selected_alphas'] == [1., 10.] and row['selection'] == 'per task, max val_accuracy; ties -> larger alpha'
    assert row['log10_F61_p10_selected_alpha_min'] == 29. and row['log10_F61_p10_selected_alpha_max'] == 30.
    assert row['log10_s1_floor_selected_alpha_min'] == 9. and row['log10_s1_floor_selected_alpha_max'] == 10.
    assert row['log10_F61_p10_costmin_alpha_grid_edge_min'] == 25.
    assert row['log10_F61_p10_costmin_alpha_grid_edge_max'] == 28.
    assert row['costmin_alphas_grid_edge'] == [1., 10.]
    # Pooled selection (Bank: mean CV-fold accuracy): mean test acc alpha 1 = .75 > .65.
    pooled = qa.family_summary(summary, decomposition, dataset='X', family='f', status='measured',
                               select_by='test_accuracy', pooled=True)
    assert pooled['selected_alphas'] == [1.]
    assert pooled['log10_F61_p10_selected_alpha_min'] == 28. and pooled['log10_F61_p10_selected_alpha_max'] == 30.
