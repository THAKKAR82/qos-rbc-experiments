"""Public transaction datasets and encodings for the QOS sparsity screen.

Each case turns a time-sorted transaction table into chronological
train/validation/test CSR matrices (sparsity_screen.Encoded). Vocabularies,
amount deciles and standardization are fit on the training slice only.

- paysim_naive: fixed D = 10 (type one-hot, log1p amount and balances,
  standardized). It illustrates s_col >= N / D: s grows with N by construction.
- paysim_time_local / aml_acct_window: entity x time-window indicator features,
  uncapped. D grows with N, so s can stay bounded; the price is that later
  rows can share no feature with training (cold start, gamma = 0).

Labels: fraud / laundering = +1, else -1.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import cache
import math

import numpy as np
import pandas as pd
from scipy import sparse

from .sparsity_screen import TEST_FRAC, VAL_FRAC, Encoded, chronological_split

PAYSIM_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
PAYSIM_MONEY = ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")
PAYSIM_WINDOW = 24       # steps (hours) per entity window
AML_WINDOW = 1440        # minutes per entity window (1 day)
AML_COLUMNS = ("Timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_received",
               "receiving_currency", "amount_paid", "payment_currency", "payment_format", "is_laundering")
N_MAX, REPLICATES = 80_000, 5
CASE_IDS = ("paysim_naive", "paysim_time_local", "aml_acct_window")


# ---------------------------------------------------------------- loaders
def _shared_codes(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    codes, _ = pd.factorize(pd.concat([a, b], ignore_index=True))
    return codes[:len(a)].astype(np.int64), codes[len(a):].astype(np.int64)


def load_paysim(path) -> pd.DataFrame:
    """PaySim rows sorted stably by step (hour); origin/destination accounts share one code space."""
    raw = pd.read_csv(path, usecols=["step", "type", *PAYSIM_MONEY, "nameOrig", "nameDest", "isFraud"])
    raw = raw.sort_values("step", kind="stable", ignore_index=True)
    orig, dest = _shared_codes(raw.nameOrig, raw.nameDest)
    return pd.DataFrame({"t": raw.step.to_numpy(np.int64), "type": raw["type"].to_numpy(),
                         **{c: raw[c].to_numpy(np.float64) for c in PAYSIM_MONEY},
                         "orig": orig, "dest": dest, "y": np.where(raw.isFraud == 1, 1, -1)})


def load_amlworld(path) -> pd.DataFrame:
    """AMLworld transactions sorted stably by time; t in minutes; an account is (bank, account id)."""
    raw = pd.read_csv(path, header=0, names=list(AML_COLUMNS), dtype=str)
    ts = pd.to_datetime(raw.Timestamp, format="%Y/%m/%d %H:%M")
    minutes = ((ts - ts.min()).dt.total_seconds() // 60).astype(np.int64)
    order = np.argsort(minutes.to_numpy(), kind="stable")
    raw, minutes = raw.iloc[order].reset_index(drop=True), minutes.iloc[order].to_numpy()
    src, dst = _shared_codes(raw.from_bank + "/" + raw.from_account, raw.to_bank + "/" + raw.to_account)
    return pd.DataFrame({"t": minutes, "src": src, "dst": dst, "fmt": raw.payment_format.to_numpy(),
                         "currency": raw.payment_currency.to_numpy(),
                         "amount": raw.amount_paid.astype(np.float64).to_numpy(),
                         "y": np.where(raw.is_laundering == "1", 1, -1)})


# ---------------------------------------------------------------- encoding primitives
def keyed_csr(rows: np.ndarray, keys: np.ndarray, n_rows: int,
              vocabulary: np.ndarray | None = None) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Indicator CSR from (row, int64 key) pairs; duplicates collapse to 1.

    vocabulary=None fits the sorted key vocabulary on these pairs (training);
    with a vocabulary, keys outside it are dropped (no peeking at later data).
    """
    rows, keys = np.asarray(rows, np.int64), np.asarray(keys, np.int64)
    if vocabulary is None:
        vocabulary = np.unique(keys)
    cols = np.searchsorted(vocabulary, keys)
    known = cols < vocabulary.size
    known[known] = vocabulary[cols[known]] == keys[known]
    X = sparse.csr_matrix((np.ones(int(known.sum())), (rows[known], cols[known])),
                          shape=(n_rows, vocabulary.size))
    X.sum_duplicates()
    X.data[:] = 1.0
    return X, vocabulary


def _families(frames, family_keys: Callable[[pd.DataFrame], list[np.ndarray]]):
    """Namespace each feature family's key, fit the vocabulary on frames[0], encode all frames."""
    out, vocabulary = [], None
    for frame in frames:
        fams = family_keys(frame)
        F, n = len(fams), len(frame)
        keys = np.concatenate([k * F + f for f, k in enumerate(fams)]) if n else np.zeros(0, np.int64)
        X, vocabulary = keyed_csr(np.tile(np.arange(n), F), keys, n, vocabulary)
        out.append(X)
    return out


# ---------------------------------------------------------------- encodings
def _last_t(*frames) -> int:
    return int(max(df.t.to_numpy().max(initial=0) for df in frames))


def encode_paysim_naive(train, val, test):
    """Fixed D = 10: type one-hot (5), log1p of amount and the 4 balances, standardized on train."""
    def raw(df):
        onehot = (df["type"].to_numpy()[:, None] == np.array(PAYSIM_TYPES)[None, :]).astype(np.float64)
        return np.hstack([onehot, np.log1p(np.column_stack([df[c].to_numpy() for c in PAYSIM_MONEY]))])
    base = raw(train)
    mean, std = base[:, 5:].mean(axis=0), base[:, 5:].std(axis=0)
    std = np.where(std > 0, std, 1.0)
    def encode(df):
        Z = raw(df)
        Z[:, 5:] = (Z[:, 5:] - mean) / std
        return sparse.csr_matrix(Z)  # D = 10: dense columns are the point, densification is cheap.
    Xs = [encode(df) for df in (train, val, test)]
    return *Xs, {"D": 10, "families": ["type one-hot", "log1p money (standardized on train)"]}


def encode_paysim_time_local(train, val, test):
    """(orig, 24-step window), (dest, 24-step window), (type, train amount decile, step) indicators."""
    edges = np.quantile(train.amount.to_numpy(), np.linspace(.1, .9, 9)) if len(train) else np.zeros(9)
    n_steps = _last_t(train, val, test) + 1
    n_windows = n_steps // PAYSIM_WINDOW + 1
    type_code = {t: i for i, t in enumerate(PAYSIM_TYPES)}
    def keys(df):
        window = df.t.to_numpy() // PAYSIM_WINDOW
        decile = np.searchsorted(edges, df.amount.to_numpy(), side="right")
        cell = df["type"].map(type_code).to_numpy(np.int64) * 10 + decile
        return [df.orig.to_numpy() * n_windows + window, df.dest.to_numpy() * n_windows + window,
                cell * n_steps + df.t.to_numpy()]
    Xs = _families((train, val, test), keys)
    return *Xs, {"families": ["orig x 24-step window", "dest x 24-step window", "type x amount decile x step"],
                 "amount_decile_edges": edges.tolist()}


def encode_aml_acct_window(train, val, test):
    """(from account, day) and (to account, day) indicators; accounts are (bank, id)."""
    n_days = _last_t(train, val, test) // AML_WINDOW + 1
    def keys(df):
        day = df.t.to_numpy() // AML_WINDOW
        return [df.src.to_numpy() * n_days + day, df.dst.to_numpy() * n_days + day]
    Xs = _families((train, val, test), keys)
    return *Xs, {"families": ["from account x day", "to account x day"]}


# ---------------------------------------------------------------- cases
def make_case(frame: pd.DataFrame, encoder, n_max: int = N_MAX,
              replicates: int = REPLICATES) -> Callable[[int, int], Encoded]:
    """build(n, replicate): chronological split of `frame` from the replicate's start offset.

    Start offsets are spaced evenly so that the largest grid N still fits after
    the last one; the same offsets are used at every N (nested prefixes).
    """
    need = n_max + math.ceil(VAL_FRAC * n_max) + math.ceil(TEST_FRAC * n_max)
    starts = np.linspace(0, max(len(frame) - need, 0), replicates).round().astype(int)

    def build(n: int, replicate: int) -> Encoded:
        start = int(starts[replicate])
        tr, va, te, truncated = chronological_split(len(frame), start, n)
        parts = [frame.iloc[s] for s in (tr, va, te)]
        Xtr, Xva, Xte, meta = encoder(*parts)
        return Encoded(Xtr, parts[0].y.to_numpy(), Xva, parts[1].y.to_numpy(), Xte, parts[2].y.to_numpy(),
                       meta=dict(meta, truncated=truncated, start=start,
                                 t_range=[int(parts[0].t.iloc[0]), int(parts[2].t.iloc[-1])]))
    return build


ENCODERS = {"paysim_naive": ("paysim", encode_paysim_naive),
            "paysim_time_local": ("paysim", encode_paysim_time_local),
            "aml_acct_window": ("aml", encode_aml_acct_window)}


@cache
def _frame(source: str) -> pd.DataFrame:
    from .cfpb_prepare import ROOT
    if source == "paysim":
        return load_paysim(ROOT / "data/paysim.csv")
    return load_amlworld(ROOT / "data/raw/screen/amlworld/HI-Small_Trans.csv")


def case(case_id: str) -> Callable[[int, int], Encoded]:
    """Builder for a registered real-data case (loads and caches the source table)."""
    source, encoder = ENCODERS[case_id]
    return make_case(_frame(source), encoder)
