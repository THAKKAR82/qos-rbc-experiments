"""Screen loaders and encodings on tiny invented CSVs (no real data)."""
import numpy as np
import pytest
from scipy import sparse

from src import screen_datasets as sd
from src import sparsity_screen as ss

PAYSIM_HEADER = ("step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,"
                 "newbalanceDest,isFraud,isFlaggedFraud\n")
AML_HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
              "Amount Paid,Payment Currency,Payment Format,Is Laundering\n")


def paysim_csv(tmp_path, rows):
    path = tmp_path / "paysim.csv"
    path.write_text(PAYSIM_HEADER + "".join(
        f"{st},{ty},{amt},{o},100.0,90.0,{d},0.0,10.0,{f},0\n" for st, ty, amt, o, d, f in rows))
    return path


def aml_csv(tmp_path, rows):
    path = tmp_path / "aml.csv"
    path.write_text(AML_HEADER + "".join(
        f"{ts},{fb},{fa},{tb},{ta},{amt},US Dollar,{amt},US Dollar,{fmt},{y}\n" for ts, fb, fa, tb, ta, amt, fmt, y in rows))
    return path


# ---------------------------------------------------------------- loaders
def test_load_paysim_sorts_stably_by_step_and_maps_labels(tmp_path):
    path = paysim_csv(tmp_path, [(2, "PAYMENT", 5., "C1", "M1", 0), (1, "TRANSFER", 7., "C2", "C3", 1),
                                 (1, "CASH_OUT", 9., "C3", "C1", 0)])
    df = sd.load_paysim(path)
    assert df.t.tolist() == [1, 1, 2] and df.amount.tolist() == [7., 9., 5.]   # Stable within a step.
    assert df.y.tolist() == [1, -1, -1]
    # One shared code space for accounts, so C1 as origin and as destination is the same entity.
    assert df.orig.iloc[2] == df.dest.iloc[1]


def test_load_amlworld_keys_accounts_by_bank_and_times_in_minutes(tmp_path):
    path = aml_csv(tmp_path, [("2022/09/01 00:20", "010", "A1", "011", "A1", 5.0, "Cheque", 0),
                              ("2022/09/01 00:05", "011", "A1", "010", "A2", 6.0, "ACH", 1),
                              ("2022/09/02 00:05", "010", "A1", "010", "A1", 7.0, "Wire", 0)])
    df = sd.load_amlworld(path)
    assert df.t.tolist() == [0, 15, 1440]
    assert df.y.tolist() == [1, -1, -1]
    # Account A1 at bank 010 and at bank 011 are different entities.
    assert df.src.iloc[1] == df.src.iloc[2] != df.src.iloc[0]
    assert df.dst.iloc[1] == df.src.iloc[0]


# ---------------------------------------------------------------- keyed CSR
def test_keyed_csr_collapses_duplicates_to_one():
    X, vocab = sd.keyed_csr(np.array([0, 0, 1]), np.array([7, 7, 9]), n_rows=2)
    assert X.shape == (2, 2) and X.nnz == 2 and X.max() == 1 and vocab.tolist() == [7, 9]


def test_keyed_csr_with_vocabulary_drops_unseen_keys():
    _, vocab = sd.keyed_csr(np.array([0]), np.array([7]), n_rows=1)
    X, _ = sd.keyed_csr(np.array([0, 1, 1]), np.array([7, 8, 9]), n_rows=2, vocabulary=vocab)
    assert X.shape == (2, 1) and np.diff(X.indptr).tolist() == [1, 0]


# ---------------------------------------------------------------- encodings
def _paysim_frame(tmp_path, n=40):
    rng = np.random.default_rng(0)
    rows = [(1 + i // 4, ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"][i % 5], float(rng.integers(1, 999)),
             f"C{i % 7}", f"C{(i * 3) % 11}", int(i % 9 == 0)) for i in range(n)]
    return sd.load_paysim(paysim_csv(tmp_path, rows))


def test_paysim_naive_is_fixed_d_and_dense(tmp_path):
    df = _paysim_frame(tmp_path)
    Xtr, Xva, Xte, meta = sd.encode_paysim_naive(df.iloc[:20], df.iloc[20:23], df.iloc[23:26])
    assert Xtr.shape[1] == Xva.shape[1] == Xte.shape[1] == 10 == meta["D"]
    assert ss.sparsity_point(Xtr)["s_col"] == 20                     # log1p(amount) is dense: s = N.
    assert abs(Xtr[:, 5].toarray().mean()) < 1e-12                   # Standardized with training statistics.


def test_paysim_time_local_columns_are_entity_windows(tmp_path):
    rows = [(1, "PAYMENT", 5., "C1", "M1", 0), (2, "PAYMENT", 6., "C1", "M2", 0),   # Same origin, same day.
            (30, "PAYMENT", 7., "C1", "M3", 1)]                                     # Same origin, next day.
    df = sd.load_paysim(paysim_csv(tmp_path, rows))
    Xtr, _, Xte, meta = sd.encode_paysim_time_local(df.iloc[:2], df.iloc[:0], df.iloc[2:])
    shared = Xtr[0].multiply(Xtr[1])
    assert shared.nnz == 1                                           # Only (C1, day 0) is shared.
    assert Xte.nnz == 0 and ss.cold_start_rate(Xtr, Xte) == 1.0      # Next-day row is cold start.
    assert meta["families"] == ["orig x 24-step window", "dest x 24-step window", "type x amount decile x step"]


def test_aml_acct_window_columns(tmp_path):
    path = aml_csv(tmp_path, [("2022/09/01 00:00", "010", "A1", "010", "B1", 5.0, "ACH", 0),
                              ("2022/09/01 10:00", "010", "A1", "010", "B2", 6.0, "ACH", 0),
                              ("2022/09/01 23:00", "010", "A9", "010", "B1", 6.0, "ACH", 1),
                              ("2022/09/02 01:00", "010", "A1", "010", "B1", 7.0, "ACH", 0)])
    df = sd.load_amlworld(path)
    Xtr, _, Xte, meta = sd.encode_aml_acct_window(df.iloc[:3], df.iloc[:0], df.iloc[3:])
    # Columns: from (A1, d0), (A9, d0); to (B1, d0), (B2, d0). (A1, d0) and (B1, d0) each appear twice.
    assert Xtr.shape[1] == 4 and ss.sparsity_point(Xtr)["s_col"] == 2
    assert ss.cold_start_rate(Xtr, Xte) == 1.0 and meta["families"] == ["from account x day", "to account x day"]


# ---------------------------------------------------------------- case builder
def test_make_case_spaces_replicate_starts_and_records_truncation(tmp_path):
    df = _paysim_frame(tmp_path, n=40)
    build = sd.make_case(df, sd.encode_paysim_naive, n_max=10, replicates=3)
    a, b = build(10, 0), build(10, 2)
    assert a.meta["start"] == 0 and b.meta["start"] == 40 - 14 and not a.meta["truncated"]
    assert a.X_train.shape[0] == 10 and a.X_test.shape[0] == 2
    assert build(20, 2).meta["truncated"]
    assert isinstance(a.X_train, sparse.csr_matrix) and set(np.unique(a.y_train)) <= {-1, 1}


def test_case_ids():
    assert sd.CASE_IDS == ("paysim_naive", "paysim_time_local", "aml_acct_window")


# ---------------------------------------------------------------- per-family diagnostics (review fix)
def test_entity_window_meta_labels_column_families_and_time_spans(tmp_path):
    path = aml_csv(tmp_path, [("2022/09/01 00:00", "010", "A1", "010", "B1", 5.0, "ACH", 0),
                              ("2022/09/01 10:00", "010", "A1", "010", "B2", 6.0, "ACH", 0),
                              ("2022/09/01 23:00", "010", "A9", "010", "B1", 6.0, "ACH", 1),
                              ("2022/09/02 01:00", "010", "A1", "010", "B1", 7.0, "ACH", 0)])
    df = sd.load_amlworld(path)
    Xtr, _, _, meta = sd.encode_aml_acct_window(df.iloc[:3], df.iloc[:0], df.iloc[3:])
    fam = np.asarray(meta["column_family"])
    assert fam.shape == (Xtr.shape[1],) and np.bincount(fam).tolist() == [2, 2]   # 2 from- and 2 to-columns.
    enc = sd.make_case(df, sd.encode_aml_acct_window, n_max=2, replicates=1)(2, 0)
    assert enc.meta["t_train"] == [0, 600] and enc.meta["t_test"][0] >= enc.meta["t_train"][1]


def test_family_breakdown_max_column_and_served_share():
    X_train = sparse.csr_matrix(np.array([[1., 1., 0.], [1., 0., 1.], [1., 0., 0.]]))
    X_test = sparse.csr_matrix(np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]))
    enc = ss.Encoded(X_train, np.array([1, -1, 1]), X_train, np.array([1, -1, 1]), X_test, np.array([1, -1, 1]),
                     meta={"truncated": False, "column_family": [0, 1, 1]})
    out = {r["family"]: r for r in sd.family_breakdown(enc, ["a", "b"])}
    assert out["a"]["max_col_nnz"] == 3 and out["b"]["max_col_nnz"] == 1
    assert out["a"]["test_rows_served"] == pytest.approx(1 / 3) and out["b"]["test_rows_served"] == pytest.approx(1 / 3)
