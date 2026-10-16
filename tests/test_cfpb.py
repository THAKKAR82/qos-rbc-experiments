"""Frozen CFPB policies, with independent exhaustive checks of similarity joins."""

import itertools
import random

import pandas as pd
import pytest

from src.cfpb import (
    CLASSES, clean_narrative, normalize_narrative, chronological_split,
    curate_cfpb, exact_jaccard_edges, word_trigrams, source_metadata,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["Complaint ID", "Date received", "Product",
                                      "Consumer complaint narrative"])


def test_normalization_and_artifacts():
    original = "  CrÉdit\u00a0CARD\nXXXX!  123 xx x xylophone Ａ  "
    assert normalize_narrative(original) == "crédit card xxxx! 123 xx x xylophone ａ"
    assert clean_narrative(original) == "crédit card ! 123 x xylophone ａ"
    assert clean_narrative(clean_narrative(original)) == clean_narrative(original)
    assert normalize_narrative(None) == normalize_narrative(pd.NA) == ""
    with pytest.raises(TypeError):
        normalize_narrative(10)
    d = frame([(1, "2026-01-01", CLASSES[0], "Product: mortgage Trouble paying"),
               (2, "2026-01-02", CLASSES[0], "Mortgage account collection problem"),
               (3, "2026-01-03", CLASSES[0], "XXXX xx !")])
    a = curate_cfpb(d).audit.set_index("complaint_id")
    assert a.loc["1", "reason"] == "structured_heading"
    assert a.loc["2", "primary"]
    assert a.loc["3", "reason"] == "empty_model_text"


@pytest.mark.parametrize("date,expected", [("2026-01-01", "train"), ("2026-01-31", "train"),
                                          ("2026-02-01", "validation"), ("2026-02-14", "validation"),
                                          ("2026-02-15", "test"), ("2026-02-28", "test")])
def test_split_boundaries(date, expected):
    assert chronological_split(date) == expected


@pytest.mark.parametrize("date", ["2025-12-31", "2026-03-01", "2026-02-30", "bad", None])
def test_invalid_split(date):
    with pytest.raises(ValueError):
        chronological_split(date)


def test_all_product_conflicts_before_class_filter_and_no_majority_vote():
    d = frame([(1, "2026-01-01", CLASSES[0], "My account has a payment problem"),
               (2, "2026-02-02", CLASSES[0], " MY account has a payment problem "),
               (3, "2026-02-20", "Student loan", "My account has a payment problem"),
               (4, "2026-01-04", "Student loan", "Other unrelated student debt")])
    a = curate_cfpb(d).audit
    assert not a.primary.any()
    assert a.reason.tolist() == ["raw_conflict", "nonprimary_class", "raw_conflict", "raw_conflict"]
    assert not a.natural_stream.any()


def test_earliest_occurrence_group_disjoint_numeric_tie_and_no_mutation():
    d = frame([(10, "2026-01-03", CLASSES[0], "This payment is wrong"),
               (2, "2026-01-03", CLASSES[0], "This payment is wrong"),
               (3, "2026-02-03", CLASSES[0], " this PAYMENT is wrong "),
               (4, "2026-02-22", CLASSES[0], "This payment is wrong"),
               (5, "2026-02-01", CLASSES[1], "Card transaction error"),
               (6, "2026-02-15", CLASSES[1], "Card transaction error")])
    before = d.copy(deep=True)
    r = curate_cfpb(d)
    pd.testing.assert_frame_equal(d, before)
    a = r.audit.set_index("complaint_id")
    assert set(a.index[a.primary]) == {"2", "5"}
    assert a.loc["5", "split"] == "validation"  # Never backdated.
    assert a.loc["4", "stratum"] == "seen_in_train"
    assert a.loc["6", "stratum"] == "seen_in_validation"
    assert a.loc["4", "natural_stream"]
    assert not a.loc["2", "natural_stream"]
    assert r.groups.query("kind == 'exact'").member_count.max() == 3
    assert r.groups.query("kind == 'normalized'").member_count.max() == 4
    assert r.audit.loc[r.audit.primary, "template_group_id"].is_unique
    again = curate_cfpb(d.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(r.audit, again.audit)
    pd.testing.assert_frame_equal(r.groups, again.groups)
    pd.testing.assert_frame_equal(r.edges, again.edges)


def test_conflict_created_by_redaction_cleaning():
    d = frame([(1, "2026-01-01", CLASSES[0], "XXXX payment error"),
               (2, "2026-01-02", CLASSES[1], "xx payment error")])
    r = curate_cfpb(d)
    assert set(r.audit.reason) == {"model_conflict"}
    assert not r.audit.raw_conflict.any()


def test_near_template_transitive_chain_and_conflicts():
    base = [f"word{i}" for i in range(100)]
    a = " ".join(base)
    b = " ".join(base + [f"tail{i}" for i in range(10)])
    c = " ".join(base + [f"tail{i}" for i in range(20)])
    d = frame([(1, "2026-01-01", CLASSES[0], a),
               (2, "2026-02-01", CLASSES[0], b),
               (3, "2026-02-15", CLASSES[0], c)])
    r = curate_cfpb(d)
    assert len(r.edges) == 2
    assert r.audit.template_group_id.nunique() == 1
    assert r.audit.primary.sum() == 1
    assert r.diagnostics["complete"]
    d.loc[2, "Product"] = "Student loan"
    r = curate_cfpb(d)
    assert set(r.audit.reason) == {"template_conflict"}
    assert not r.audit.primary.any()


def test_prefix_join_matches_exhaustive_including_threshold_and_short_sets():
    rng = random.Random(42)
    sets = []
    for _ in range(100):
        base = set(rng.sample(range(200), rng.randint(20, 70)))
        sets.extend([frozenset(base), frozenset(base | set(rng.sample(range(200), 5)))])
    # Boundary 51/60 == .85, below boundary 50/61; both are >=20 tokens.
    sets.extend([frozenset(range(60)), frozenset(range(51)), frozenset(range(50)) | {99},
                 frozenset(range(19)), frozenset(range(19))])
    before = list(sets)
    expected = []
    for i, j in itertools.combinations(range(len(sets)), 2):
        if min(len(sets[i]), len(sets[j])) < 20:
            continue
        inter = len(sets[i] & sets[j]); union = len(sets[i] | sets[j])
        if 20 * inter >= 17 * union:
            expected.append((i, j, inter, union))
    observed, diagnostics = exact_jaccard_edges(sets)
    assert observed == expected
    assert sets == before
    assert diagnostics["short_texts"] == 2
    assert diagnostics["complete"]
    assert exact_jaccard_edges([])[0] == []


def test_trigrams_use_words_no_stopword_removal():
    assert word_trigrams("a, b c a b") == frozenset([("a", "b", "c"), ("b", "c", "a"), ("c", "a", "b")])


def test_missing_invalid_and_duplicate_ids():
    d = frame([("bad", "2026-01-01", CLASSES[0], "payment trouble"),
               (2, "bad", CLASSES[0], "payment trouble"),
               (3, "2026-01-03", CLASSES[0], "\u2003 "),
               (4, "2026-01-04", "", "payment problem")])
    a = curate_cfpb(d).audit
    assert set(a.reason) == {"invalid_id", "invalid_date", "empty_narrative", "missing_product"}
    with pytest.raises(ValueError, match="Duplicate complaint"):
        curate_cfpb(frame([(1, "2026-01-01", CLASSES[0], "first"),
                          ("01", "2026-02-01", CLASSES[0], "second")]))
    with pytest.raises(ValueError, match="outside"):
        source_metadata(frame([(1, "2027-01-01", CLASSES[0], "first")]))


def test_empty_and_singleton_populations():
    with pytest.raises(ValueError, match="No source"):
        curate_cfpb(frame([]))
    r = curate_cfpb(frame([(1, "2026-01-01", CLASSES[0], "")]))
    assert r.groups.empty and r.edges.empty and not r.model_texts
