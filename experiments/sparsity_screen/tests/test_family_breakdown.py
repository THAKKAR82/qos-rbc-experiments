"""Per-family breakdown: rebuilt points must reproduce the committed scaling table."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import sparsity_screen as ss
from experiments.sparsity_screen import family_breakdown as fb


def build(n, r):
    enc = ss.control_case("bounded")(n, r)
    return ss.Encoded(enc.X_train, enc.y_train, enc.X_val, enc.y_val, enc.X_test, enc.y_test,
                      meta=dict(enc.meta, column_family=[0] * enc.X_train.shape[1], t_train=[0, n], t_test=[n, n + 1]))


def scaling_for(grid, reps):
    return ss.screen_case(build, grid, reps).assign(case="toy")


def test_breakdown_rows_reproduce_scaling_and_report_spans():
    scaling = scaling_for((200, 400), 2)
    fams, spans = fb.breakdown_rows("toy", build, ["all"], scaling)
    assert len(fams) == 4 and len(spans) == 4
    assert all(f["max_col_nnz"] == s for f, s in zip(fams, scaling.s_col))
    assert spans[0]["t_train_span"] == 200 and spans[0]["test_after_train_window"] is None


def test_breakdown_rows_reject_drift_from_committed_scaling():
    scaling = scaling_for((200,), 1)
    scaling.loc[0, "s_row_or_col"] += 1
    with pytest.raises(ValueError, match="does not reproduce"):
        fb.breakdown_rows("toy", build, ["all"], scaling)


def test_breakdown_tolerates_csv_float_round_trip_but_not_real_drift():
    scaling = scaling_for((200,), 1)
    scaling.loc[0, "cold_start_rate"] += 1e-14        # What a CSV round trip can do to a float.
    fb.breakdown_rows("toy", build, ["all"], scaling)
    scaling.loc[0, "cold_start_rate"] += 1e-6
    with pytest.raises(ValueError, match="does not reproduce"):
        fb.breakdown_rows("toy", build, ["all"], scaling)
