import numpy as np
import pytest
from scipy import sparse

from src.text_preprocessing import (
    make_tfidf_vectorizer, make_unigram_vectorizer, fit_text_partitions, text_matrix_stats,
)


def test_explicit_configurations():
    primary = make_tfidf_vectorizer().get_params()
    for key, value in dict(ngram_range=(1, 2), max_features=50000, min_df=2,
                           max_df=1.0, norm="l2", stop_words="english", dtype=np.float32,
                           sublinear_tf=False, binary=False, use_idf=True,
                           smooth_idf=True, strip_accents=None).items():
        assert primary[key] == value
    control = make_unigram_vectorizer().get_params()
    assert control["ngram_range"] == (1, 1)
    assert control["min_df"] == 1 and control["max_features"] is None


@pytest.mark.parametrize("reference", [False, True])
def test_training_only_idf_sparse_dtype_determinism_and_no_mutation(reference):
    train = ["mortgage payment error", "mortgage payment delayed", "credit card error"]
    heldout = {"validation": ["futureexclusive mortgage"], "test": ["futureexclusive", "credit"],
               "empty": []}
    before = list(train), {k: list(v) for k, v in heldout.items()}
    v, matrices = fit_text_partitions(train, heldout, reference=reference)
    assert "futureexclusive" not in v.vocabulary_
    assert v.idf_[v.vocabulary_["mortgage"]] == pytest.approx(np.log(4 / 3) + 1)
    assert matrices["test"].getrow(0).nnz == 0
    assert matrices["empty"].shape == (0, len(v.vocabulary_))
    for x in matrices.values():
        assert sparse.isspmatrix_csr(x) and x.dtype == np.float32 and x.has_canonical_format
    v2, again = fit_text_partitions(train, heldout, reference=reference)
    assert v.vocabulary_ == v2.vocabulary_
    np.testing.assert_array_equal(v.idf_, v2.idf_)
    for k in matrices:
        assert (matrices[k] != again[k]).nnz == 0
    assert (train, heldout) == before


def test_cap_and_meaningful_stats_without_densifying(monkeypatch):
    v = make_tfidf_vectorizer(max_features=3, min_df=1)
    x = v.fit_transform(["apple banana apple", "banana orange pear"])
    assert x.shape == (2, 3)
    original = x.copy()
    monkeypatch.setattr(sparse.csr_matrix, "toarray", lambda *a, **k: pytest.fail("densified"))
    s = text_matrix_stats(x)
    assert s["N"] == 2 and s["D"] == 3 and s["nnz"] == x.nnz
    assert s["csr_bytes"] == x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
    assert s["dense_equivalent_bytes"] == 24
    assert (x != original).nnz == 0
    assert text_matrix_stats(sparse.csr_matrix((0, 3), dtype=np.float32))["qos_row_only"] is None


def test_no_training_partition_override():
    with pytest.raises(ValueError, match="cannot replace"):
        fit_text_partitions(["apple banana"], {"train": ["leaked"]})
