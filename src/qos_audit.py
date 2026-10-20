"""QOS applicability audit, step 1: measure the parameters QOS's guarantees depend on.

Reference: Zhao et al., "Exponential quantum advantage in processing massive
classical data", arXiv:2604.07639v2 (1 Oct 2026), Appendix F:
  - Eq. F49: r_reg = min_w (||Xw - y||^2 + lam ||w||^2) / ||y||^2 <= 0.99 (squared).
  - Eq. F50: kappa_reg = sqrt((sigma_max^2 + lam) / (sigma_min^2 + lam)) with
    sigma_min^2 = lambda_min(X^T X); zero eigenvalues count when D > N.
  - Theorem F.13 (p.146): qubits S (Eq. F60) and samples M (Eq. F61).
    s = max nonzeros per row OR column of X; ||X|| = 1; ||x'||_inf <= 1.
    b in F60 is the bit length of each matrix entry; it is additive in S and
    is NOT included here.
  - "We take N and D to count only nonzero training samples and nonzero
    features" (p.143): N_nonzero / D_nonzero are primary, full sizes recorded.

Conventions follow the theorem, not sklearn defaults: labels in {-1, +1},
unweighted ridge, no intercept, training matrix scaled to spectral norm 1, and
sklearn alpha on the unscaled matrix mapped to lam = alpha / sigma_max^2.

This is a calculated / theoretical accounting study, not a quantum simulation.
Cost numbers are polynomial PREFACTORS: constants, b, R (taken as 1) and the
polylogs hidden in O-tilde are dropped; all logs are base 2 and log(m) is
floored at 1 bit. Read them as orders of magnitude, never as resource
estimates, and never as evidence of quantum advantage.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import re
import sys
import uuid

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, svds
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .machine_size import (_countable_matrix, matrix_nnz, max_row_or_col_sparsity,
                           max_row_sparsity, qos_svm_machine_size)

SEED = 42
ALPHAS = (0.1, 1.0, 10.0, 100.0)
DELTA = 0.1
MARGIN_THRESHOLDS = (0.3, 0.1, 0.03, 0.01, 0.003)
MAX_DF_FRACTIONS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.01)
ZIPF_RANGE = (10, 1000)
LSQR_TOL = 1e-10
LSQR_ITER_LIM = 20_000
INTERPRETATION = "calculated / theoretical accounting"


# ---------------------------------------------------------------- normalization
def _float64_csr(X):
    """Canonical float64 CSR copy (mathematical nonzeros); never densifies."""
    matrix = _countable_matrix(X)
    return sparse.csr_matrix(matrix, dtype=np.float64, copy=True)


def spectral_norm(X, seed: int = SEED) -> float:
    """Largest singular value of a sparse matrix (seeded start vector)."""
    X = _float64_csr(X)
    v0 = np.random.default_rng(seed).standard_normal(min(X.shape))
    return float(svds(X, k=1, v0=v0, return_singular_vectors=False)[0])


def alpha_to_lambda(alpha: float, sigma_max: float) -> float:
    """sklearn Ridge alpha on unscaled X  ->  theorem's lam for X / sigma_max."""
    return alpha / sigma_max**2


def sigma_min_scaled(n_rows: int, n_cols: int) -> float:
    """sigma_min(X) per Eq. F50 when it is implied by shape alone.

    sigma_min^2 = lambda_min(X^T X) for the D x D Gram matrix, so D > N forces
    rank <= N < D and sigma_min = 0. Otherwise it must be computed, not assumed.
    """
    if n_cols > n_rows:
        return 0.0
    raise ValueError(f"D={n_cols} <= N={n_rows}: sigma_min is not implied; compute it explicitly")


DENSE_SVD_MAX_D = 2_000


def svd_sigma_min(X) -> dict:
    """Exact singular values by dense SVD, for the D <= N case (small D only).

    Uses numpy's matrix_rank tolerance, sigma_max * max(N, D) * eps. Eq. F50
    counts zero eigenvalues of X^T X, so sigma_min = 0 whenever rank < D; the
    raw smallest value and the smallest nonzero value are kept as diagnostics.
    """
    if X.shape[1] > DENSE_SVD_MAX_D:
        raise ValueError(f"D={X.shape[1]} > {DENSE_SVD_MAX_D}: dense SVD not supported here")
    dense = X.toarray() if sparse.issparse(X) else np.asarray(X)
    sv = np.linalg.svd(np.asarray(dense, dtype=np.float64), compute_uv=False)
    tol = float(sv[0] * max(dense.shape) * np.finfo(np.float64).eps)
    nonzero = sv[sv > tol]
    rank = int(nonzero.size)
    return {"method": "numpy.linalg.svd (dense, float64)", "tol": tol, "rank": rank,
            "D": int(dense.shape[1]), "sigma_max": float(sv[0]), "sigma_min_raw": float(sv[-1]),
            "smallest_nonzero": float(nonzero[-1]) if rank else 0.0,
            "sigma_min": float(sv[-1]) if rank == dense.shape[1] else 0.0,
            "singular_values": sv.tolist()}


def kappa_reg(lam: float, sigma_min_scaled: float = 0.0) -> float:
    """Regularized condition number for ||X|| = 1 (Eq. F50 / Lemma F.14)."""
    denominator = sigma_min_scaled**2 + lam
    if denominator <= 0:
        raise ValueError("kappa_reg is unbounded for sigma_min = lam = 0")
    return math.sqrt((1.0 + lam) / denominator)


# ---------------------------------------------------------------- sparsity
def _quantiles(a: np.ndarray) -> dict:
    if not a.size:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    out = {f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 99)}
    out["max"] = int(a.max())
    return out


def _row_col_counts(X):
    matrix = _float64_csr(X)
    return np.diff(matrix.indptr), np.bincount(matrix.indices, minlength=matrix.shape[1])


def nonzero_support(X):
    """Boolean masks of rows and columns holding any mathematical nonzero."""
    rows, columns = _row_col_counts(X)
    return rows > 0, columns > 0


def zipf_slope(col_nnz: np.ndarray, lo: int = ZIPF_RANGE[0], hi: int = ZIPF_RANGE[1]) -> float:
    """Slope of log(document frequency) vs log(rank) over ranks [lo, hi).

    A slope near -1 is classic Zipf: column sparsity is then set by the most
    common features and grows with N, not with polylog N. NaN if too few ranks.
    """
    df = np.sort(np.asarray(col_nnz)[np.asarray(col_nnz) > 0])[::-1].astype(float)
    hi = min(hi, df.size)
    if hi - lo < 5:
        return float("nan")
    rank = np.arange(lo, hi) + 1.0
    return float(np.polyfit(np.log(rank), np.log(df[lo:hi]), 1)[0])


def sparsity_profile(X) -> dict:
    """Row/column nonzero counts. Quantiles exclude empty rows/columns."""
    rows, columns = _row_col_counts(X)
    return {
        "nnz": matrix_nnz(X),
        "row_nnz": _quantiles(rows[rows > 0]),
        "col_nnz": _quantiles(columns[columns > 0]),
        "s_row_only": max_row_sparsity(X),
        "s_row_or_col": max_row_or_col_sparsity(X),
        "empty_rows": int((rows == 0).sum()),
        "empty_cols": int((columns == 0).sum()),
    }


def max_df_sweep(X, fractions=MAX_DF_FRACTIONS) -> list[dict]:
    """Column sparsity if features with document frequency > f*N were masked.

    A post-hoc column mask on the fitted matrix (sklearn max_df semantics,
    keep df <= f*N): no vocabulary, IDF or L2 refit. Reports what making the
    data 'QOS-sparse' would cost in lost nonzeros; no accuracy is computed.
    """
    matrix = _float64_csr(X)
    n = matrix.shape[0]
    df = np.bincount(matrix.indices, minlength=matrix.shape[1])
    total = int(df.sum())
    out = []
    for f in fractions:
        keep = (df > 0) & (df <= f * n)
        kept = matrix[:, keep]
        out.append({
            "max_df": float(f),
            "df_cutoff": float(f * n),
            "features_kept": int(keep.sum()),
            "nnz_share_lost": float(1 - df[keep].sum() / total) if total else 0.0,
            "s_row_only": max_row_sparsity(kept),
            "s_row_or_col": max_row_or_col_sparsity(kept),
            "rows_emptied": int((np.diff(kept.indptr) == 0).sum() - (np.diff(matrix.indptr) == 0).sum()),
        })
    return out


# ---------------------------------------------------------------- ridge + margins
def ridge_no_intercept(Xs, y, lam: float, tol: float = LSQR_TOL,
                       iter_lim: int = LSQR_ITER_LIM) -> tuple[np.ndarray, dict]:
    """argmin ||Xs w - y||^2 + lam ||w||^2 by LSQR (Xs already has ||Xs|| = 1)."""
    Xs = _float64_csr(Xs)
    y = np.asarray(y, dtype=np.float64)
    result = lsqr(Xs, y, damp=math.sqrt(lam), atol=tol, btol=tol, iter_lim=iter_lim)
    w, istop, itn = result[0], int(result[1]), int(result[2])
    gradient = Xs.T @ (Xs @ w - y) + lam * w
    info = {"istop": istop, "iterations": itn, "converged": istop in (1, 2, 4, 5),
            "normal_equation_relative_residual": float(np.linalg.norm(gradient) /
                                                       np.linalg.norm(Xs.T @ y))}
    return w, info


def residual_ratio(Xs, y, w, lam: float) -> float:
    """Normalized regularized residual r_reg at w (Eq. F49, squared ratio)."""
    y = np.asarray(y, dtype=np.float64)
    r = Xs @ w - y
    return float((r @ r + lam * (w @ w)) / (y @ y))


def predict_sign(X, w) -> np.ndarray:
    """sgn(x'.w) with ties (including all-zero rows) mapped to +1."""
    return np.where(np.asarray(X @ w).ravel() >= 0, 1, -1)


def margins(X_test, w, columns=None) -> dict:
    """Geometric margins gamma = |x'.w| / ||w|| (Eq. F48), as-is and inf-rescaled.

    Rescaling each test row to ||x'||_inf = 1 keeps every prediction and gives
    the largest margin the theorem's ||x'||_inf <= 1 allows. With `columns`, the
    infinity norm is taken over the training support only: discarded all-zero
    training columns carry w = 0 and are outside the theorem's D_nonzero space.
    """
    X_test = _float64_csr(X_test)
    support = X_test if columns is None else X_test[:, columns]
    raw = np.abs(X_test @ w) / np.linalg.norm(w)
    row_max = np.asarray(abs(support).max(axis=1).todense()).ravel()
    scale = np.where(row_max > 0, row_max, 1.0)
    return {"max_abs_entry": float(row_max.max(initial=0)), "row_max": row_max,
            "as_is": raw, "inf_rescaled": raw / scale}


def margin_summary(g: np.ndarray, y_true, y_pred, thresholds=MARGIN_THRESHOLDS) -> dict:
    """Quantiles, plus coverage/accuracy if QOS only answers when gamma >= g0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {f"q{p}": float(np.percentile(g, p)) for p in (0, 1, 5, 10, 25, 50)}
    for g0 in thresholds:
        m = g >= g0
        out[f"coverage@{g0}"] = float(m.mean())
        out[f"acc@{g0}"] = float((y_true[m] == y_pred[m]).mean()) if m.any() else None
    return out


def binary_metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}


# ---------------------------------------------------------------- cost prefactors
def qos_costs(N: int, D: int, s: int, kappa: float, gamma: float,
              m: int = 1, delta: float = DELTA) -> dict:
    """A9 machine size plus log10 polynomial prefactors of F60 and F61.

    F60 is reported WITHOUT its additive b term. log10_F61_passes is
    log10(M / (R N)). Both are None when gamma <= 0 (theorem inapplicable).
    """
    out = {"A9_qubits_single_prediction": qos_svm_machine_size(N, D, s),
           "inputs": {"N": int(N), "D": int(D), "s": int(s), "kappa_reg": float(kappa),
                      "gamma_test": float(gamma), "m": int(m), "delta": float(delta)}}
    if not (gamma > 0 and math.isfinite(gamma)) or s < 1:
        out.update(log10_F60_qubits_without_b=None, log10_F61_passes=None)
        return out
    lD, lm, li, lk = math.log2(D), math.log2(max(m, 2)), math.log2(1 / delta), math.log2(kappa)
    qubits = (s * lD**2 * (min(s * lD, lm) + li**2.5) / gamma**2
              + math.log2(s * D) * lk + lk**2.5)
    log10_passes = (7 * math.log10(s) + 4 * math.log10(kappa) - 4 * math.log10(gamma)
                    - math.log10(delta) + math.log10(min(s**2 * lD**2, lm**2)))
    out.update(log10_F60_qubits_without_b=math.log10(qubits), log10_F61_passes=log10_passes)
    return out


F61_FACTORS = ("s^7", "kappa^4", "gamma^-4", "delta^-1", "min_term")


def f61_decomposition(inputs: dict) -> dict:
    """log10 of each F61 factor of M/(RN), their sum, and an s = 1 counterfactual.

    `inputs` is qos_costs(...)["inputs"]. min_term is min{s^2 log^2 D, log^2 m}
    (log2, log m floored at 1 bit, as in qos_costs). The s = 1 total keeps every
    other measured factor: it is the floor set by kappa, gamma, delta and m.
    Pure arithmetic: same prefactor caveats as qos_costs.
    """
    s, D, m = int(inputs["s"]), int(inputs["D"]), int(inputs["m"])
    kappa, gamma, delta = float(inputs["kappa_reg"]), float(inputs["gamma_test"]), float(inputs["delta"])
    if not (gamma > 0 and math.isfinite(gamma)) or s < 1:
        raise ValueError("F61 is undefined for gamma <= 0 or s < 1")
    lD2, lm2 = math.log2(D) ** 2, math.log2(max(m, 2)) ** 2
    def min_term(sparsity):
        return min(sparsity**2 * lD2, lm2)
    out = {"log10_s^7": 7 * math.log10(s), "log10_kappa^4": 4 * math.log10(kappa),
           "log10_gamma^-4": -4 * math.log10(gamma), "log10_delta^-1": -math.log10(delta),
           "log10_min_term": math.log10(min_term(s)),
           "min_term_branch": "s^2 log2^2 D" if s**2 * lD2 < lm2 else "log2^2 m"}
    out["log10_total"] = sum(out[f"log10_{f}"] for f in F61_FACTORS)
    out["log10_total_s1"] = out["log10_total"] - out["log10_s^7"] - out["log10_min_term"] + math.log10(min_term(1))
    return out


def derive_f61_decomposition(output_dir, gamma_label: str = "p10"):
    """Decomposition table for every task and alpha, read from saved task JSON only."""
    import pandas as pd
    out = Path(output_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    rows = []
    for task_id in manifest["tasks"]:
        result = json.loads((out / f"{task_id}.json").read_text())
        for r in result["alpha_sweep"]:
            inputs = r[f"costs_gamma_{gamma_label}"]["inputs"]
            rows.append(dict(task_id=task_id, kind=result["task"]["kind"], alpha=r["alpha"],
                             gamma_label=gamma_label, gamma=inputs["gamma_test"], s=inputs["s"],
                             kappa_reg=inputs["kappa_reg"], D=inputs["D"], m=inputs["m"],
                             delta=inputs["delta"], **f61_decomposition(inputs)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- one binary task
def _check_labels(*labels):
    for y in labels:
        values = set(np.unique(np.asarray(y)).tolist())
        if not values <= {-1, 1}:
            raise ValueError(f"Labels must be -1/+1, got {sorted(values)}")


def audit_binary(X_train, y_train, X_val, y_val, X_test, y_test, *, alphas=ALPHAS,
                 delta: float = DELTA, thresholds=MARGIN_THRESHOLDS,
                 max_df_fractions=MAX_DF_FRACTIONS, zipf_range=ZIPF_RANGE,
                 seed: int = SEED) -> dict:
    """Step-1 audit for ONE binary task (labels in {-1, +1}).

    Sweeps sklearn-style alphas because kappa_reg and accuracy trade off: the
    whole curve is reported, not a selected point.
    """
    has_val = X_val is not None
    _check_labels(y_train, y_test, *([y_val] if has_val else []))
    y_train, y_test = np.asarray(y_train, dtype=np.int64), np.asarray(y_test, dtype=np.int64)
    X_full = _float64_csr(X_train)
    rows, cols = nonzero_support(X_full)
    X, y = X_full[rows], y_train[rows]  # All-zero rows cannot change w (p.143).
    N_nz, D_nz = int(rows.sum()), int(cols.sum())
    sigma = spectral_norm(X, seed=seed)
    if D_nz > N_nz:
        s_min, svd, source = sigma_min_scaled(N_nz, D_nz), None, "shape_D_gt_N"
    else:  # Rank check on the nonzero columns: zero columns would add spurious zeros.
        svd = svd_sigma_min(X[:, cols])
        if not math.isclose(svd["sigma_max"], sigma, rel_tol=1e-8):
            raise ValueError(f"svds/SVD sigma_max disagree: {sigma} vs {svd['sigma_max']}")
        s_min, source = svd["sigma_min"] / sigma, "dense_svd_rank_check"
    Xs = X / sigma
    sparsity = sparsity_profile(X)
    s = sparsity["s_row_or_col"]
    _, col_nnz = _row_col_counts(X)
    X_test = _float64_csr(X_test)
    if has_val:
        X_val, y_val = _float64_csr(X_val), np.asarray(y_val, dtype=np.int64)
    test_support_nnz = np.diff(X_test[:, cols].indptr)

    sweep = []
    for alpha in alphas:
        lam = alpha_to_lambda(alpha, sigma)
        kappa = kappa_reg(lam, s_min)
        w, info = ridge_no_intercept(Xs, y, lam)
        pt = predict_sign(X_test, w)
        mg = margins(X_test, w, columns=cols)
        g = mg["inf_rescaled"]
        row = {"alpha": float(alpha), "lam": lam, "kappa_reg": kappa, "lsqr": info,
               "residual_ratio": residual_ratio(Xs, y, w, lam),
               "validation": binary_metrics(y_val, predict_sign(X_val, w)) if has_val else None,
               "test": binary_metrics(y_test, pt),
               "test_zero_decisions": int((X_test @ w == 0).sum()),
               "test_max_abs_entry_as_is": mg["max_abs_entry"],
               "margins_as_is": margin_summary(mg["as_is"], y_test, pt, thresholds),
               "margins_inf_rescaled": margin_summary(g, y_test, pt, thresholds),
               "margins_inf_rescaled_sorted": np.sort(g).tolist()}
        # Literal requirement (min test margin) and a 'serve 90%' relaxation,
        # where m counts only the test points actually served.
        for label, gamma in (("min", float(g.min())), ("p10", float(np.percentile(g, 10)))):
            row[f"costs_gamma_{label}"] = qos_costs(N_nz, D_nz, s, kappa, gamma,
                                                    m=int((g >= gamma).sum()), delta=delta)
        sweep.append(row)

    return {
        "N_full": int(X_full.shape[0]), "D_full": int(X_full.shape[1]),
        "N_nonzero": N_nz, "D_nonzero": D_nz, "sigma_max": sigma,
        "sigma_min_scaled": s_min, "sigma_min_source": source, "svd": svd, "log2_N_nonzero": math.log2(N_nz), "log2_D_nonzero": math.log2(D_nz),
        "train_label_counts": {"+1": int((y == 1).sum()), "-1": int((y == -1).sum())},
        "validation_N": int(X_val.shape[0]) if has_val else 0, "test_N": int(X_test.shape[0]),
        "sparsity": sparsity,
        "zipf": {"slope": zipf_slope(col_nnz, *zipf_range), "rank_range": list(zipf_range),
                 "column_df_sorted": np.sort(col_nnz[col_nnz > 0])[::-1].tolist()},
        "test_sparsity": {"max_row_nnz_on_training_support": int(test_support_nnz.max(initial=0)),
                          "test_rows_s_sparse": bool(test_support_nnz.max(initial=0) <= s),
                          "zero_rows_on_training_support": int((test_support_nnz == 0).sum())},
        "A9_variants": {
            "nonzero_N_D_row_or_col": qos_svm_machine_size(N_nz, D_nz, s),
            "full_N_D_row_or_col": qos_svm_machine_size(X_full.shape[0], X_full.shape[1], s),
            "nonzero_N_D_row_only": qos_svm_machine_size(N_nz, D_nz, sparsity["s_row_only"])},
        "max_df_sweep": max_df_sweep(X, max_df_fractions),
        "alpha_sweep": sweep,
    }


# ---------------------------------------------------------------- CFPB tasks
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def binary_tasks(classes) -> list[dict]:
    """All pairs (lower class index = +1), then one-vs-rest (named class = +1)."""
    tasks = [{"task_id": f"{_slug(classes[a])}__vs__{_slug(classes[b])}", "kind": "pair",
              "positive": a, "negative": b}
             for a, b in itertools.combinations(range(len(classes)), 2)]
    tasks += [{"task_id": f"{_slug(classes[c])}__vs__rest", "kind": "one_vs_rest",
               "positive": c, "negative": None} for c in range(len(classes))]
    return tasks


def task_labels(labels, positive: int, negative: int | None):
    """Row mask and {-1, +1} labels for one task (negative=None: one-vs-rest)."""
    labels = np.asarray(labels)
    mask = (labels == positive) | (labels == negative) if negative is not None else np.ones(len(labels), bool)
    return mask, np.where(labels[mask] == positive, 1, -1)


def _clean(value):
    """JSON-safe: numpy scalars to Python, NaN/inf to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary_rows(task: dict, result: dict) -> list[dict]:
    rows = []
    for r in result["alpha_sweep"]:
        mi = r["margins_inf_rescaled"]
        rows.append({
            "task_id": task["task_id"], "kind": task["kind"], "alpha": r["alpha"], "lam": r["lam"],
            "N_nonzero": result["N_nonzero"], "D_nonzero": result["D_nonzero"],
            "N_full": result["N_full"], "D_full": result["D_full"], "sigma_max": result["sigma_max"],
            "s_row_only": result["sparsity"]["s_row_only"],
            "s_row_or_col": result["sparsity"]["s_row_or_col"], "zipf_slope": result["zipf"]["slope"],
            "kappa_reg": r["kappa_reg"], "residual_ratio": r["residual_ratio"],
            "lsqr_converged": r["lsqr"]["converged"],
            "val_accuracy": r["validation"]["accuracy"] if r["validation"] else None,
            "val_balanced_accuracy": r["validation"]["balanced_accuracy"] if r["validation"] else None,
            "test_accuracy": r["test"]["accuracy"],
            "test_balanced_accuracy": r["test"]["balanced_accuracy"],
            "gamma_min": mi["q0"], "gamma_p10": mi["q10"], "gamma_p50": mi["q50"],
            "coverage@0.03": mi.get("coverage@0.03"), "acc@0.03": mi.get("acc@0.03"),
            "A9_qubits": r["costs_gamma_min"]["A9_qubits_single_prediction"],
            "log10_F60_qubits_without_b_gamma_min": r["costs_gamma_min"]["log10_F60_qubits_without_b"],
            "log10_F60_qubits_without_b_gamma_p10": r["costs_gamma_p10"]["log10_F60_qubits_without_b"],
            "log10_F61_passes_gamma_min": r["costs_gamma_min"]["log10_F61_passes"],
            "log10_F61_passes_gamma_p10": r["costs_gamma_p10"]["log10_F61_passes"],
        })
    return rows


CFPB_DATASET = "CFPB Jan-Feb 2026 primary4 binary"
CFPB_PREPROCESSING = "frozen January TF-IDF (accepted 50k uni+bigram); task rows only; no refit"


def ledger_rows(task: dict, result: dict, *, run_id: str, date: str, commit: str | None,
                output_dir: str, notes_common: dict, dataset: str = CFPB_DATASET,
                preprocessing: str = CFPB_PREPROCESSING,
                experiment_prefix: str = "cfpb-qos-audit-step1") -> list[dict]:
    """Append-only experiment_log.csv rows for one task (or CV fold)."""
    base = dict(dataset=f"{dataset}:{task['task_id']}", date=date,
                git_commit=commit or "",
                preprocessing=preprocessing,
                n_samples=result["N_nonzero"], n_features=result["D_nonzero"],
                max_sparsity=result["sparsity"]["s_row_or_col"], sketch_method="none",
                sketch_dimension="", seed=SEED,
                model="unweighted ridge, no intercept, ||X||=1, lam=alpha/sigma_max^2 (LSQR)")
    def row(suffix, metric, value, interpretation, **extra):
        notes = dict(notes_common, interpretation=interpretation, task_id=task["task_id"],
                     artifact=f"{output_dir}/{task['task_id']}.json", run_id=run_id, **extra)
        return dict(experiment_id=f"{experiment_prefix}-{run_id}-{task['task_id']}{suffix}",
                    **base, metric=metric, result=value,
                    notes=json.dumps(_clean(notes), sort_keys=True))
    out = [row("", "s_row_or_col", result["sparsity"]["s_row_or_col"], INTERPRETATION,
               s_row_only=result["sparsity"]["s_row_only"]),
           row("", "qos_A9_machine_size", result["A9_variants"]["nonzero_N_D_row_or_col"],
               INTERPRETATION + "; single prediction; not measured hardware usage",
               A9_variants=result["A9_variants"])]
    for r in result["alpha_sweep"]:
        a = f"-a{r['alpha']:g}"
        extra = dict(alpha=r["alpha"], lam=r["lam"])
        out.append(row(a, "kappa_reg", r["kappa_reg"], INTERPRETATION, **extra))
        for label in ("min", "p10"):
            out.append(row(a, f"log10_F61_passes_gamma_{label}", r[f"costs_gamma_{label}"]["log10_F61_passes"],
                           INTERPRETATION + "; polynomial prefactor, constants/polylogs/b dropped; order of magnitude only",
                           gamma=r[f"costs_gamma_{label}"]["inputs"]["gamma_test"], **extra))
        for split in ("validation", "test"):
            if r[split] is None:
                continue
            for metric in ("accuracy", "balanced_accuracy"):
                out.append(row(a, f"{split}_{metric}", r[split][metric],
                               "measured holdout score of the audit ridge (theorem conventions)", **extra))
    return out


def run_step1(output_dir, *, alphas=ALPHAS, delta=DELTA, thresholds=MARGIN_THRESHOLDS,
              max_df_fractions=MAX_DF_FRACTIONS, zipf_range=ZIPF_RANGE, seed=SEED,
              gate_record="docs/decisions/cfpb_modeling_gate.md"):
    """Rebuild the frozen CFPB primary TF-IDF and audit all ten binary tasks.

    Writes <task>.json, summary.csv, ledger_rows.csv and manifest.json into a
    NEW directory; refuses to overwrite. Does not touch experiment_log.csv.
    """
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from .cfpb import CLASSES, SOURCE_SHA256
    from .cfpb_baselines import ACCEPTED, build_representation, load_rows
    from .cfpb_prepare import ROOT, git_commit, write_json

    out = Path(output_dir)
    out = out if out.is_absolute() else ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; audit outputs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    source = ROOT / "data/CCDB_Export_15_January_2026_through_February_2026.csv"
    commit = git_commit(ROOT)
    code = ["src/qos_audit.py", "src/machine_size.py", "src/cfpb_baselines.py",
            "src/text_preprocessing.py", "src/cfpb.py"]
    config = dict(alphas=list(alphas), delta=delta, margin_thresholds=list(thresholds),
                  max_df_fractions=list(max_df_fractions), zipf_rank_range=list(zipf_range),
                  seed=seed, lsqr_tol=LSQR_TOL, lsqr_iter_lim=LSQR_ITER_LIM, threads=1)
    manifest = dict(
        run_id=run_id, date_utc=now.isoformat(), interpretation=INTERPRETATION,
        paper="Zhao et al., arXiv:2604.07639v2 (1 Oct 2026), Theorem F.13, Eqs. F48-F50, F60, F61, A9",
        git_commit=commit,
        code_provenance="git_commit plus code_sha256 (audit code uncommitted at run time if hashes differ from HEAD)",
        code_sha256={p: _sha256(ROOT / p) for p in code},
        source_sha256=_sha256(source), preparation=str(ACCEPTED.relative_to(ROOT)),
        preparation_manifest_sha256=_sha256(ACCEPTED / "manifest.json"),
        gate_record=gate_record, gate_record_sha256=_sha256(ROOT / gate_record),
        python=sys.version, platform=platform.platform(),
        packages={n: importlib.metadata.version(n) for n in
                  ["numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl"]},
        config=config, classes=list(CLASSES),
        conventions=dict(
            labels="+1 = lower CLASSES index (pairs) or the named class (one-vs-rest)",
            model="unweighted ridge, no intercept; uniform-sample QOS computes unweighted ridge, "
                  "so one-vs-rest balanced accuracy may be poor; no weighted variant in this task",
            N_D="N_nonzero/D_nonzero over the task's training rows are primary (paper p.143); full sizes recorded",
            sigma_min="0 because D_nonzero > N_nonzero (Eq. F50 counts zero eigenvalues of X^T X)",
            r_reg="squared ratio, Eq. F49",
            margins="test rows inf-norm rescaled over training-support columns; predictions unchanged",
            costs="F60 without additive b; F61 as log10(M/(R N)) with R=1; log2 throughout; "
                  "log m floored at 1 bit; constants and hidden polylogs dropped: orders of magnitude only",
            max_df="post-hoc column mask, no refit"),
        status="running")
    if manifest["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Source CSV differs from frozen SOURCE_SHA256")
    write_json(out / "manifest.json", _clean(manifest))

    with threadpool_limits(limits=1):
        rows = load_rows(source, ACCEPTED)
        partitions, matrices, _ = build_representation("primary", rows, ACCEPTED, [])
        labels = {k: partitions[k].label.to_numpy() for k in ("train", "validation", "test")}
        summary, ledger = [], []
        notes_common = dict(code_sha256=manifest["code_sha256"], gate_record=gate_record,
                            preparation=manifest["preparation"], source_sha256=manifest["source_sha256"])
        for task in binary_tasks(CLASSES):
            masks = {k: task_labels(labels[k], task["positive"], task["negative"]) for k in labels}
            result = audit_binary(matrices["train"][masks["train"][0]], masks["train"][1],
                                  matrices["validation"][masks["validation"][0]], masks["validation"][1],
                                  matrices["test"][masks["test"][0]], masks["test"][1],
                                  alphas=alphas, delta=delta, thresholds=thresholds,
                                  max_df_fractions=max_df_fractions, zipf_range=zipf_range, seed=seed)
            payload = dict(task=dict(task, positive_class=CLASSES[task["positive"]],
                                     negative_class=(CLASSES[task["negative"]] if task["negative"] is not None
                                                     else "rest")),
                           run_id=run_id, interpretation=INTERPRETATION, gate_record=gate_record,
                           conventions=manifest["conventions"], **result)
            write_json(out / f"{task['task_id']}.json", _clean(payload))
            summary += summary_rows(task, result)
            ledger += ledger_rows(task, result, run_id=run_id, date=now.date().isoformat(),
                                  commit=commit, output_dir=str(out.relative_to(ROOT)),
                                  notes_common=notes_common)
            print("audited", task["task_id"], flush=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "ledger_rows.csv", index=False)
    manifest.update(status="completed", tasks=[t["task_id"] for t in binary_tasks(CLASSES)],
                    artifacts_sha256={p.name: _sha256(p) for p in sorted(out.iterdir())
                                      if p.name != "manifest.json"})
    write_json(out / "manifest.json", _clean(manifest))
    return out, run_id


def write_derived_manifest(output_dir, paths, *, name="derived_manifest.json"):
    """Hash post-run artifacts (figures, derived tables) without touching manifest.json.

    Written once; later calls only verify, and raise on any hash drift.
    """
    out = Path(output_dir)
    source = out / "manifest.json"
    hashes = {p: _sha256(out / p) for p in sorted(paths)}
    target = out / name
    if target.exists():
        recorded = json.loads(target.read_text())
        if recorded["artifacts_sha256"] != hashes:
            raise ValueError(f"Derived artifact drift in {out}; do not overwrite {name}")
        return recorded
    data = dict(source_run_id=json.loads(source.read_text()).get("run_id"),
                source_manifest_sha256=_sha256(source), interpretation=INTERPRETATION,
                note="Post-run figures and derived tables; pure functions of the task JSONs",
                artifacts_sha256=hashes)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


# ---------------------------------------------------------------- Bank (D < N)
BANK_CATEGORICAL = ("job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "poutcome")
BANK_NUMERIC = ("age", "balance", "day", "duration", "campaign", "pdays", "previous")
BANK_CAVEATS = (
    "Known leakage: the validated encoded matrix is notebook 05's global fit "
    "(OneHotEncoder + StandardScaler on all 45,211 rows before CV), so fold "
    "validation rows influenced the encoding (AGENTS.md).",
    "Bank's validated accounting convention is row-only sparsity; this audit "
    "reports row-only and row-or-column s side by side.",
    "Bank is a pipeline-validation dataset, not evidence of practical quantum advantage.",
    "Audit ridge (LSQR on CSR, no intercept, ||X|| = 1) is not notebook 05's "
    "RidgeClassifier; its scores are not the regression-fixture scores.",
)


def bank_encoded():
    """Notebook 05's validated global encoding and +/-1 labels (yes = +1). Read-only."""
    import pandas as pd
    from .cfpb_prepare import ROOT
    from .preprocessing import make_preprocessor
    df = pd.read_csv(ROOT / "data/bank_marketing_qos_encoded.csv")
    X = make_preprocessor(list(BANK_CATEGORICAL), list(BANK_NUMERIC)).fit_transform(
        df[list(BANK_CATEGORICAL) + list(BANK_NUMERIC)])
    y01 = df["y"].map({"no": 0, "yes": 1}).to_numpy()
    if np.isnan(y01.astype(float)).any():
        raise ValueError("Unexpected Bank target values")
    return X, y01.astype(int), np.where(y01 == 1, 1, -1)


def run_bank_step1(output_dir, *, alphas=ALPHAS, delta=DELTA, thresholds=MARGIN_THRESHOLDS,
                   max_df_fractions=MAX_DF_FRACTIONS, zipf_range=ZIPF_RANGE, seed=SEED,
                   n_splits=5, cv_seed=42):
    """Audit Bank on the validated global encoding; each CV fold's validation rows are the test set."""
    import pandas as pd
    from .cfpb_prepare import ROOT, git_commit, write_json
    from .evaluation import make_stratified_cv
    from .machine_size import max_row_sparsity as _row_s

    out = Path(output_dir)
    out = out if out.is_absolute() else ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; audit outputs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    data = ROOT / "data/bank_marketing_qos_encoded.csv"
    X, y01, y = bank_encoded()
    if X.shape != (45211, 51) or _row_s(X) != 16 or matrix_nnz(X) != 723376:
        raise ValueError(f"Bank representation drift: {X.shape}")
    full_svd = svd_sigma_min(X)
    code = ["src/qos_audit.py", "src/machine_size.py", "src/preprocessing.py", "src/evaluation.py"]
    folds = list(make_stratified_cv(n_splits=n_splits, seed=cv_seed).split(np.zeros(len(y)), y01))
    tasks = [f"cv_fold_{k + 1}" for k in range(len(folds))]
    manifest = dict(
        run_id=run_id, date_utc=now.isoformat(), interpretation=INTERPRETATION,
        dataset="Bank Marketing (bank-full)", git_commit=git_commit(ROOT),
        code_sha256={p: _sha256(ROOT / p) for p in code}, data_sha256=_sha256(data),
        representation=dict(source="notebook 05 cell 6: make_preprocessor(...).fit_transform(X_raw), global fit",
                            N=int(X.shape[0]), D=int(X.shape[1]), dtype=str(np.asarray(X).dtype),
                            max_row_sparsity=_row_s(X), nnz=matrix_nnz(X),
                            max_row_or_col_sparsity=max_row_or_col_sparsity(X)),
        full_matrix_svd={k: v for k, v in full_svd.items()},
        cv=dict(splitter="src.evaluation.make_stratified_cv", n_splits=n_splits, seed=cv_seed,
                test_set="validation rows of each fold", validation_split="none"),
        caveats=list(BANK_CAVEATS),
        python=sys.version, platform=platform.platform(),
        packages={n: importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn", "pandas"]},
        config=dict(alphas=list(alphas), delta=delta, margin_thresholds=list(thresholds),
                    max_df_fractions=list(max_df_fractions), zipf_rank_range=list(zipf_range), seed=seed,
                    lsqr_tol=LSQR_TOL, lsqr_iter_lim=LSQR_ITER_LIM),
        conventions=dict(labels="+1 = subscribed (y = yes)",
                         model="unweighted ridge, no intercept, ||X|| = 1, lam = alpha / sigma_max^2 (LSQR on CSR)",
                         sigma_min="dense SVD of each fold's training matrix; numpy matrix_rank tolerance; "
                                   "0 if rank < D (Eq. F50)",
                         margins="fold validation rows, inf-norm rescaled; standardized entries exceed 1 as-is",
                         costs="as CFPB step 1: F60 without b, F61 as log10(M/(R N)); orders of magnitude only"),
        status="running")
    write_json(out / "manifest.json", _clean(manifest))
    summary, ledger = [], []
    notes_common = dict(code_sha256=manifest["code_sha256"], data_sha256=manifest["data_sha256"],
                        caveat="global preprocessing before CV (known leakage)")
    Xc = sparse.csr_matrix(np.asarray(X, dtype=np.float64))  # Audit-internal; Bank pipeline untouched.
    for task_id, (train, test) in zip(tasks, folds):
        result = audit_binary(Xc[train], y[train], None, None, Xc[test], y[test], alphas=alphas,
                              delta=delta, thresholds=thresholds, max_df_fractions=max_df_fractions,
                              zipf_range=zipf_range, seed=seed)
        task = dict(task_id=task_id, kind="cv_fold", positive_class="yes", negative_class="no")
        payload = dict(task=task, run_id=run_id, interpretation=INTERPRETATION,
                       caveats=list(BANK_CAVEATS), conventions=manifest["conventions"], **result)
        write_json(out / f"{task_id}.json", _clean(payload))
        summary += summary_rows(task, result)
        ledger += ledger_rows(task, result, run_id=run_id, date=now.date().isoformat(),
                              commit=manifest["git_commit"], output_dir=str(out.relative_to(ROOT)),
                              notes_common=notes_common, dataset="Bank Marketing (bank-full)",
                              preprocessing="validated global encoding (notebook 05 cell 6; known leakage); "
                                            "fold train rows; dense->CSR inside audit only",
                              experiment_prefix="bank-qos-audit-step1")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "ledger_rows.csv", index=False)
    manifest.update(status="completed", tasks=tasks,
                    artifacts_sha256={p.name: _sha256(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"})
    write_json(out / "manifest.json", _clean(manifest))
    return out, run_id


# ---------------------------------------------------------------- cross-dataset summary
ANALYTIC_STATUS = "analytic, not measured; representation not onboarded in this repo"


def analytic_dense_column_row(dataset: str, n_rows: int) -> dict:
    """With any fully dense column, s_row_or_col = N, so log10 s^7 = 7 log10 N."""
    return dict(dataset=dataset, family="all (any dense column)", status=ANALYTIC_STATUS,
                n_tasks=None, N=int(n_rows), s_row_or_col=int(n_rows), s_over_N=1.0,
                **{"log10_s^7": 7 * math.log10(n_rows)})


def _select_alpha(frame, column):
    """Row with the highest `column`; ties go to the larger alpha (smaller kappa_reg)."""
    return frame.sort_values([column, "alpha"], ascending=[False, False]).iloc[0]


def family_summary(summary, decomposition, *, dataset: str, family: str, status: str,
                   select_by: str = "val_accuracy", pooled: bool = False) -> dict:
    """One row per dataset/task family.

    Headline: log10 F61 at gamma = p10 at the alpha selected by `select_by`
    (per task, or one alpha for the family by mean across tasks when pooled,
    e.g. Bank's mean CV-fold accuracy). Secondary: the cost-minimizing alpha,
    which sits at the grid edge.
    """
    import pandas as pd
    s = summary.assign(s_over_N=summary.s_row_or_col / summary.N_nonzero)
    merged = s.merge(decomposition[["task_id", "alpha", "log10_total", "log10_total_s1"]],
                     on=["task_id", "alpha"], validate="one_to_one")
    if pooled:
        alpha = float(_select_alpha(merged.groupby("alpha", as_index=False)[select_by].mean(), select_by).alpha)
        chosen = merged[merged.alpha == alpha]
        selection = f"one alpha for all tasks, max mean {select_by}; ties -> larger alpha"
    else:
        chosen = pd.DataFrame([_select_alpha(g, select_by) for _, g in merged.groupby("task_id", sort=False)])
        selection = f"per task, max {select_by}; ties -> larger alpha"
    costmin = merged.loc[merged.groupby("task_id").log10_total.idxmin()]
    return dict(dataset=dataset, family=family, status=status, n_tasks=int(s.task_id.nunique()),
                selection=selection, selected_alphas=sorted(set(chosen.alpha.astype(float))),
                log10_F61_p10_selected_alpha_min=float(chosen.log10_total.min()),
                log10_F61_p10_selected_alpha_max=float(chosen.log10_total.max()),
                log10_s1_floor_selected_alpha_min=float(chosen.log10_total_s1.min()),
                log10_s1_floor_selected_alpha_max=float(chosen.log10_total_s1.max()),
                selected_alpha_kappa_reg_min=float(chosen.kappa_reg.min()),
                selected_alpha_kappa_reg_max=float(chosen.kappa_reg.max()),
                N_min=int(s.N_nonzero.min()), N_max=int(s.N_nonzero.max()),
                s_row_only_min=int(s.s_row_only.min()), s_row_only_max=int(s.s_row_only.max()),
                s_row_or_col_min=int(s.s_row_or_col.min()), s_row_or_col_max=int(s.s_row_or_col.max()),
                s_over_N_min=float(s.s_over_N.min()), s_over_N_max=float(s.s_over_N.max()),
                kappa_reg_min=float(s.kappa_reg.min()), kappa_reg_max=float(s.kappa_reg.max()),
                gamma_p10_min=float(s.gamma_p10.min()), gamma_p10_max=float(s.gamma_p10.max()),
                costmin_alphas_grid_edge=sorted(set(costmin.alpha.astype(float))),
                log10_F61_p10_costmin_alpha_grid_edge_min=float(costmin.log10_total.min()),
                log10_F61_p10_costmin_alpha_grid_edge_max=float(costmin.log10_total.max()),
                log10_s1_floor_costmin_alpha_min=float(costmin.log10_total_s1.min()),
                log10_s1_floor_costmin_alpha_max=float(costmin.log10_total_s1.max()))


# ---------------------------------------------------------------- empirical F61 (close-out)
EMPIRICAL_LABEL = "extrapolation from small-size simulation; orders of magnitude only"
PROOF_BE_S_EXPONENT = 5.0  # Theorem D.23: O~(R N s^5 / eps) per block-encoding query


def empirical_f61(row, alpha_s):
    """log10 F61 at gamma = p10 with the block-encoding s^5 replaced by a measured s^alpha.

    F104: s^7 = s^5 (per-query block encoding, Theorem D.23) x s^2 (Q^2 via F97). Only the s^5 is
    replaced; kappa^4, gamma^-4, delta^-1, the min term and the readout s^2 keep their proof values,
    which are NOT measured. Pure arithmetic on a saved decomposition row.
    """
    s = float(row["s"])
    ls = math.log10(s)
    if not math.isclose(float(row["log10_s^7"]), 7 * ls, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("decomposition row's log10_s^7 is not 7 log10(s)")
    be_proof, be_meas = PROOF_BE_S_EXPONENT * ls, float(alpha_s) * ls
    return dict(alpha_s=float(alpha_s), log10_s_block_encoding_proof=be_proof,
                log10_s_block_encoding_measured=be_meas, log10_s_readout_proof=2 * ls,
                log10_total_proof=float(row["log10_total"]),
                log10_total_empirical=float(row["log10_total"]) - (be_proof - be_meas),
                kappa_gamma_delta_status="proof values (not measured)", label=EMPIRICAL_LABEL)


def padded_dimension(N, D):
    """2^n with n = ceil(log2(max(N, D))), the embedding dimension used by the proof (F99)."""
    return 1 << (int(max(N, D)) - 1).bit_length()


def anchored_per_query_samples(s, dim, alpha_s, anchor_samples, anchor_s=32, anchor_dim=64):
    """Samples per A/2 block-encoding query, scaled from a measured anchor by (s/s0)^alpha (dim/dim0).

    The linear padded-dimension factor is the theorem's (D211), not measured. Extrapolation only.
    """
    return float(anchor_samples) * (float(s) / anchor_s) ** float(alpha_s) * (float(dim) / anchor_dim)
