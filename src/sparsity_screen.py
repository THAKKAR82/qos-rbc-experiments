"""Sparsity-condition screen: does Theorem F.13's s stay bounded as N grows?

Reference: Zhao et al., arXiv:2604.07639v2, Theorem F.13: s = max nonzeros per
row OR column of the training matrix. QOS's cost is polynomial in s, so the
theorem is only interesting if s stays (poly)logarithmic as N grows.

Key bound (calculated): sum_j c_j = nnz >= N for N nonzero rows, so
max_j c_j >= N / D. Any fixed-D encoding therefore has s growing at least
linearly in N, and bounded s needs D to grow with N.

The screen measures s on chronological training prefixes of growing N, fits
log s vs log N across replicate start offsets, and applies a pre-registered
verdict (experiments/sparsity_screen/PROTOCOL.md). It also measures the
cold-start rate: the share of test rows with no overlap with the training
features, which have margin gamma = 0. These are measured sparsity statistics,
not quantum resource estimates.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .qos_audit import SEED, nonzero_support, sparsity_profile

PASS_BELOW, FAIL_ABOVE = 0.2, 0.5  # Approved 2026-10-26, fixed before any data is read.
VAL_FRAC = TEST_FRAC = 0.15
N_BOOT = 2000


@dataclass
class Encoded:
    """One chronological train/val/test split of one case at one N (CSR, labels +/-1)."""
    X_train: sparse.csr_matrix
    y_train: np.ndarray
    X_val: sparse.csr_matrix
    y_val: np.ndarray
    X_test: sparse.csr_matrix
    y_test: np.ndarray
    meta: dict = field(default_factory=lambda: {"truncated": False})


# ---------------------------------------------------------------- splits
def chronological_split(n_total: int, start: int, n_train: int, val_frac: float = VAL_FRAC,
                        test_frac: float = TEST_FRAC) -> tuple[slice, slice, slice, bool]:
    """Train = n_train rows from `start`, then ceil(val_frac n) validation and ceil(test_frac n) test.

    If the stream ends first, n_train shrinks to the largest size that fits and
    truncated=True is returned, so a short prefix is recorded, never reused silently.
    """
    def need(m):
        return m + math.ceil(val_frac * m) + math.ceil(test_frac * m)
    available = n_total - start
    m, truncated = int(n_train), False
    if need(m) > available:
        m, truncated = int(available / (1 + val_frac + test_frac)), True
        while m > 0 and need(m) > available:
            m -= 1
    if m < 1:
        raise ValueError(f"no training rows fit after start={start} in n_total={n_total}")
    v, t = start + m, start + m + math.ceil(val_frac * m)
    return slice(start, v), slice(v, t), slice(t, t + math.ceil(test_frac * m)), truncated


# ---------------------------------------------------------------- measurements
def sparsity_point(X) -> dict:
    """Row/column sparsity of one training matrix (mathematical nonzeros, never densified)."""
    p = sparsity_profile(X)
    rows, cols = nonzero_support(X)
    return {"N_nonzero": int(rows.sum()), "D_nonzero": int(cols.sum()), "nnz": p["nnz"],
            "s_row": p["s_row_only"], "s_col": p["col_nnz"]["max"],
            "s_row_or_col": p["s_row_or_col"], "col_nnz_p99": p["col_nnz"]["p99"]}


def cold_start_rate(X_train, X_test) -> float:
    """Share of test rows with no nonzero in any training-support column.

    Such rows have x'.w = 0 for every model trained on X_train, so their margin
    gamma is 0 and Theorem F.13's guarantee does not apply to them.
    """
    if X_test.shape[0] == 0:
        raise ValueError("cold start is undefined for an empty test set")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError(f"column spaces differ: {X_train.shape[1]} vs {X_test.shape[1]}")
    _, support = nonzero_support(X_train)
    served, _ = nonzero_support(sparse.csr_matrix(X_test)[:, support])
    return float(1 - served.mean())


# ---------------------------------------------------------------- slope fit and verdict
def _ols(x, y):
    xm = x - x.mean()
    slope = float(xm @ (y - y.mean()) / (xm @ xm))
    return slope, float(y.mean() - slope * x.mean())


def fit_loglog_slope(N, s, replicate, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """OLS slope of log s on log N; 95% CIs by cluster bootstrap over replicates and by t.

    Slope ~0: s bounded in N (condition can hold); ~1: s grows linearly (fails).
    Identical s everywhere gives exactly 0 with CI (0, 0).
    """
    x, y = np.log(np.asarray(N, float)), np.log(np.asarray(s, float))
    replicate = np.asarray(replicate)
    reps = np.unique(replicate)
    out = {"n_points": int(x.size), "n_replicates": int(reps.size)}
    if np.ptp(y) == 0:
        return dict(out, slope=0.0, intercept=float(y[0]), ci_boot=(0.0, 0.0), ci_t=(0.0, 0.0), n_boot_used=0)
    slope, intercept = _ols(x, y)
    dof = x.size - 2
    resid = y - (intercept + slope * x)
    se = math.sqrt(float(resid @ resid) / dof / float(((x - x.mean()) ** 2).sum()))
    half = float(stats.t.ppf(.975, dof)) * se
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(replicate == r) for r in reps]
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, reps.size, reps.size)])
        if np.ptp(x[idx]) > 0:
            boot.append(_ols(x[idx], y[idx])[0])
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (math.nan, math.nan)
    return dict(out, slope=slope, intercept=intercept, ci_boot=(float(lo), float(hi)),
                ci_t=(slope - half, slope + half), n_boot_used=len(boot))


def classify_slope(ci_lo: float, ci_hi: float, *, pass_below: float = PASS_BELOW,
                   fail_above: float = FAIL_ABOVE) -> str:
    """Pre-registered verdict: PASS if CI upper < 0.2, FAIL if CI lower > 0.5, else INCONCLUSIVE."""
    if ci_hi < pass_below:
        return "PASS"
    if ci_lo > fail_above:
        return "FAIL"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------- synthetic controls
def bounded_control(n_total: int, s: int = 16, d_ratio: int = 2,
                    seed: int | Sequence[int] = SEED) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Positive control: sum of s injective row->column maps, so every row and column has <= s nonzeros.

    Labels y = sgn(X w* + 0.1 noise) (ties -> +1) are planted only so the
    matrix can stand in for a task; the control validates the screener, not QOS.
    """
    entropy = [int(v) for v in np.atleast_1d(seed)]
    rng = np.random.default_rng(entropy)
    d = d_ratio * n_total
    cols = np.concatenate([rng.permutation(d)[:n_total] for _ in range(s)])
    rows = np.tile(np.arange(n_total), s)
    X = sparse.csr_matrix((np.ones(cols.size), (rows, cols)), shape=(n_total, d))
    X.sum_duplicates()
    X.data[:] = 1.0
    label_rng = np.random.default_rng(entropy + [1])
    score = X @ label_rng.standard_normal(d) + .1 * label_rng.standard_normal(n_total)
    return X, np.where(score >= 0, 1, -1)


def dense_column_control(n_total: int, s: int = 16, d_ratio: int = 2,
                         seed: int | Sequence[int] = SEED) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Negative control: bounded_control plus one all-ones column, so s_col = N."""
    X, y = bounded_control(n_total, s, d_ratio, seed)
    return sparse.hstack([X, np.ones((n_total, 1))], format="csr"), y


def margin_control(n_total: int, s: int = 16, d_ratio: int = 1, threshold_nnz: int | None = None,
                   seed: int | Sequence[int] = SEED) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Synthetic best case: bounded support (<= s per row and column) AND a planted margin.

    Rows get a random class; each class owns a disjoint column block and every
    row draws s columns from its own block through s injective maps, so labels
    are y = sgn(x . w*) for w* = +1 / -1 on the two blocks. A row's planted margin
    is nnz / sqrt(D); rows below the fixed threshold threshold_nnz / sqrt(D)
    (default s - 1, i.e. at most one map collision) are dropped. This is a
    reference point for kappa, gamma and F61, not evidence about real data.
    """
    threshold_nnz = s - 1 if threshold_nnz is None else threshold_nnz
    rng = np.random.default_rng([int(v) for v in np.atleast_1d(seed)])
    n_gen = n_total + math.ceil(.1 * n_total) + 200
    y = np.where(rng.random(n_gen) < .5, 1, -1)
    rows, cols, offset = [], [], 0
    for label in (1, -1):
        members = np.flatnonzero(y == label)
        d = d_ratio * members.size
        for _ in range(s):
            rows.append(members)
            cols.append(offset + rng.permutation(d)[:members.size])
        offset += d
    X = sparse.csr_matrix((np.ones(n_gen * s), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_gen, offset))
    X.sum_duplicates()
    X.data[:] = 1.0
    keep = np.flatnonzero(np.diff(X.indptr) >= threshold_nnz)
    if keep.size < n_total:
        raise ValueError(f"only {keep.size} of {n_total} rows meet the margin threshold")
    keep = keep[:n_total]
    return X[keep], y[keep]


def gamma_upper_bound(X_train, X_test, n_served: int) -> float:
    """Calculated ceiling on the margin that n_served test rows can all reach.

    With test entries |x| <= 1 on the training support S (true after the audit's
    inf-rescaling), sum_i gamma_i^2 = ||X'_S w||^2 / ||w||^2 <= ||X'_S||^2
    <= s'_row s'_col (Schur bound). So if n_served rows have gamma_i >= g, then
    g <= sqrt(s'_row s'_col / n_served): with column-bounded test data, the
    served margin shrinks like 1/sqrt(m). Sparsity patterns are unchanged by
    row rescaling, so the bound applies to the audit's gamma as reported.
    """
    if n_served < 1:
        return math.inf
    _, support = nonzero_support(X_train)
    p = sparsity_profile(sparse.csr_matrix(X_test)[:, support])
    return math.sqrt(p["s_row_only"] * p["col_nnz"]["max"] / n_served)


CONTROLS = {"bounded": bounded_control, "dense_col": dense_column_control, "bounded_margin": margin_control}


def control_case(kind: str, s: int = 16, max_rows: int | None = None) -> Callable[[int, int], Encoded]:
    """build(n, replicate) for a synthetic control; replicate r uses seed [42, r].

    max_rows caps the stream length to exercise truncation.
    """
    make = CONTROLS[kind]

    def build(n: int, replicate: int) -> Encoded:
        wanted = n + math.ceil(VAL_FRAC * n) + math.ceil(TEST_FRAC * n)
        total = wanted if max_rows is None else min(wanted, max_rows)
        tr, va, te, truncated = chronological_split(total, 0, n)
        X, y = make(te.stop, s, seed=[SEED, replicate])
        return Encoded(X[tr], y[tr], X[va], y[va], X[te], y[te],
                       meta={"truncated": truncated, "kind": kind, "seed": [SEED, replicate]})
    return build


# ---------------------------------------------------------------- screening
def screen_case(build: Callable[[int, int], Encoded], n_grid: Sequence[int],
                replicates: int = 5) -> pd.DataFrame:
    """One row per (N_requested, replicate): training-matrix sparsity and test cold-start rate."""
    rows = []
    for n in n_grid:
        for r in range(replicates):
            enc = build(int(n), r)
            rows.append(dict(N_requested=int(n), replicate=r, truncated=bool(enc.meta["truncated"]),
                             N_train=int(enc.X_train.shape[0]),
                             cold_start_rate=cold_start_rate(enc.X_train, enc.X_test),
                             **sparsity_point(enc.X_train)))
    return pd.DataFrame(rows)
