"""Sparse text representations with explicit training-only fitting.

These helpers do not select CFPB classes, curate text, or call Bank preprocessing.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .machine_size import max_row_sparsity, matrix_nnz, qos_svm_machine_size


def make_tfidf_vectorizer(*, ngram_range=(1, 2), max_features=50000, min_df=2):
    """Construct the explicit float32/L2/English-stop-word TF-IDF configuration."""
    return TfidfVectorizer(
        analyzer="word", ngram_range=ngram_range, max_features=max_features,
        min_df=min_df, max_df=1.0, stop_words="english", lowercase=True,
        strip_accents=None, token_pattern=r"(?u)\b\w\w+\b", binary=False,
        sublinear_tf=False, use_idf=True, smooth_idf=True, norm="l2", dtype=np.float32,
    )


def make_unigram_vectorizer():
    """IMDb-style uncapped unigram control; curation/splits remain caller-owned."""
    return make_tfidf_vectorizer(ngram_range=(1, 1), max_features=None, min_df=1)


def _canonical(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def fit_text_partitions(training_texts, evaluation_texts, *, reference=False):
    """Fit a fresh vectorizer only on training texts; transform named heldouts.

    Returns (fitted vectorizer, {'train': CSR, ...}). Empty heldouts are supported.
    The function neither mutates the inputs nor drops zero-feature rows.
    """
    if "train" in evaluation_texts:
        raise ValueError("Evaluation mapping cannot replace the training partition")
    vectorizer = make_unigram_vectorizer() if reference else make_tfidf_vectorizer()
    matrices = {"train": _canonical(vectorizer.fit_transform(training_texts))}
    for name, texts in evaluation_texts.items():
        texts = list(texts)
        matrices[name] = (_canonical(vectorizer.transform(texts)) if texts else
                          sparse.csr_matrix((0, len(vectorizer.vocabulary_)), dtype=np.float32))
    return vectorizer, matrices


def text_matrix_stats(matrix):
    """Measured sparse payload and mathematical sparsity, never dense conversion.

    QOS estimates are theoretical per-binary-subproblem proxies, not measured
    memory or total multiclass costs. Callers must label population/partition.
    """
    if not sparse.issparse(matrix):
        raise TypeError("Expected a sparse text matrix")
    x = matrix.tocsr(copy=True)
    x.sum_duplicates()
    x.eliminate_zeros()
    if not np.isfinite(x.data).all():
        raise ValueError("Nonfinite text features")
    n, d = x.shape
    nnz = matrix_nnz(x)
    row = max_row_sparsity(x)
    column = int(np.bincount(x.indices, minlength=d).max()) if d else 0
    result = dict(N=n, D=d, nnz=nnz, mean_row_nnz=nnz / n if n else None,
                  max_row_nnz=row, max_column_nnz=column,
                  density=nnz / (n * d) if n and d else None,
                  zero_rows=int((np.diff(x.indptr) == 0).sum()), dtype=str(x.dtype),
                  index_dtype=str(x.indices.dtype),
                  csr_bytes=x.data.nbytes + x.indices.nbytes + x.indptr.nbytes,
                  dense_equivalent_bytes=n * d * x.dtype.itemsize)
    result["qos_row_only"] = qos_svm_machine_size(n, d, row) if n and d else None
    result["qos_row_or_column"] = qos_svm_machine_size(n, d, max(row, column)) if n and d else None
    return result
