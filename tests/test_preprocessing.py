"""Explicit Bank column configuration stays here, not in the generic factory.

Global fitting below characterizes the existing encoded matrix only; it does
not correct or endorse the sketch sweep's known preprocessing leakage.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import make_preprocessor
from src.machine_size import matrix_nnz, max_row_sparsity, qos_svm_machine_size

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]
NUMERIC = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]


def test_bank_feature_count_order_values_and_accounting():
    frame = pd.read_csv(ROOT / "data/bank_marketing_qos_encoded.csv")
    before = frame.copy(deep=True)
    transformer = make_preprocessor(CATEGORICAL, NUMERIC)
    actual = transformer.fit_transform(frame)
    # Compare with the frozen original assignment, independent of the refactor.
    fixture = json.loads((ROOT / "tests/fixtures/bank_original_cells.json").read_text())
    source = fixture["cells"]["4"]
    assert hashlib.sha256(source.encode()).hexdigest() == "0487ef0b08aeda42421f281ad6da4f2dd51d124de25219fe7aa738f8fb1f08a3"
    tree = ast.parse(source)
    assignment = tree.body[0]
    assert isinstance(assignment, ast.Assign)
    assert ast.unparse(assignment.targets[0]) == "preprocessor"
    namespace = dict(ColumnTransformer=ColumnTransformer, OneHotEncoder=OneHotEncoder,
                     StandardScaler=StandardScaler, categorical_columns=CATEGORICAL,
                     numeric_columns=NUMERIC)
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "notebook-preprocessor", "exec"), namespace)
    original = namespace["preprocessor"]
    expected = original.fit_transform(frame)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (45211, 51)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(transformer.get_feature_names_out(), original.get_feature_names_out())
    assert transformer.get_feature_names_out()[-7:].tolist() == [f"numeric__{c}" for c in NUMERIC]
    assert matrix_nnz(actual) == 723376
    assert max_row_sparsity(actual) == 16
    assert qos_svm_machine_size(*actual.shape, max_row_sparsity(actual)) == 41
    pd.testing.assert_frame_equal(frame, before)


def test_feature_order_unknown_categories_and_non_mutation():
    frame = pd.DataFrame({"second": ["z", "a"], "first": ["b", "a"],
                          "n2": [10., 14.], "n1": [1., 3.]})
    before = frame.copy(deep=True)
    transformer = make_preprocessor(["first", "second"], ["n1", "n2"])
    first = transformer.fit_transform(frame)
    other = make_preprocessor(["first", "second"], ["n1", "n2"])
    np.testing.assert_array_equal(first, other.fit_transform(frame))
    assert transformer.get_feature_names_out().tolist() == [
        "categorical__first_a", "categorical__first_b", "categorical__second_a",
        "categorical__second_z", "numeric__n1", "numeric__n2"]
    np.testing.assert_array_equal(first[:, -2:], [[-1., -1.], [1., 1.]])
    unseen = pd.DataFrame({"first": ["unknown"], "second": ["unknown"],
                           "n1": [2.], "n2": [12.]})
    unseen_before = unseen.copy(deep=True)
    np.testing.assert_array_equal(transformer.transform(unseen), np.zeros((1, 6)))
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(unseen, unseen_before)


def test_low_density_output_remains_sparse():
    frame = pd.DataFrame({"category": [f"c{i}" for i in range(20)],
                          "number": np.arange(20, dtype=float)})
    transformer = make_preprocessor(["category"], ["number"])
    output = transformer.fit_transform(frame)
    assert sparse.isspmatrix_csr(output)
    assert output.shape == (20, 21)
    assert output.nnz == 40
    assert sparse.issparse(transformer.transform(frame.iloc[:1]))
