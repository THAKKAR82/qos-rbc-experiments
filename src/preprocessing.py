"""Tabular preprocessing matching the validated Bank notebook.

Column selection, feature exclusions, and target mapping belong to experiment
configuration. The caller controls fitting: Bank baselines fit inside CV folds,
whereas the existing sketch sweeps fit globally before CV (a known leakage
risk). This factory deliberately does not change either protocol.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(categorical_columns, numeric_columns):
    """Return an unfitted transformer with explicit, ordered column lists.

Categorical one-hot features precede centered/scaled numeric features. Encoder
categories retain sklearn's default ordering. Unknown categories map to zeros.
The original sparse_threshold=0.3 is preserved: Bank produces a dense array;
lower-density encodings may produce sparse output. Do not force CSR here,
because that can change Ridge's automatic solver and validated scores.
"""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             list(categorical_columns)),
            ("numeric", StandardScaler(), list(numeric_columns)),
        ],
        sparse_threshold=0.3,
    )
