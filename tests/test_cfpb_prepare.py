"""End-to-end output-contract test on a small immutable synthetic CSV."""

import hashlib
import json

import pandas as pd

from src.cfpb import CLASSES, SOURCE_COLUMNS
from src import cfpb_prepare


def test_preparation_counts_membership_and_rebuild_determinism(tmp_path, monkeypatch):
    rows = []
    cid = 0
    for product in CLASSES:
        for date in ("2026-01-01", "2026-01-20", "2026-02-04", "2026-02-20"):
            cid += 1
            rows.append([cid, date, product, f"Payment account error customer number {cid} banking"])
    rows.append([100, "2026-02-02", CLASSES[0], ""])
    path = tmp_path / "input.csv"
    pd.DataFrame(rows, columns=SOURCE_COLUMNS).to_csv(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(cfpb_prepare, "SOURCE_SHA256", hashlib.sha256(before).hexdigest())
    monkeypatch.setattr(cfpb_prepare, "check_resources", lambda **kwargs: None)
    # This fixture tests integration, not minimum real-benchmark sample sizes.
    monkeypatch.setattr(cfpb_prepare, "curation_gates", lambda audit: dict(
        blocking=[], validity_review_required=[], modeling_permitted=True))
    first = cfpb_prepare.prepare(path, tmp_path / "runs")
    second = cfpb_prepare.prepare(path, tmp_path / "runs")
    assert path.read_bytes() == before
    counts = pd.read_csv(first / "curated_class_counts.csv", index_col=0)
    assert list(counts.index) == list(CLASSES)
    assert counts.train.tolist() == [2, 2, 2, 2]
    assert counts.validation.tolist() == counts.test.tolist() == [1, 1, 1, 1]
    audit = pd.read_csv(first / "curation_audit.csv")
    assert len(audit) == 17 and audit.primary.sum() == 16
    assert audit.loc[audit.complaint_id == 100, "reason"].item() == "empty_narrative"
    for filename in ("curation_audit.csv", "curated_class_counts.csv", "split_manifest.csv",
                     "population_counts.csv", "vocabulary.csv", "idf.npy",
                     "unigram_vocabulary.csv", "unigram_idf.npy", "duplicate_groups.csv",
                     "near_template_edges.csv", "vectorizer_manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    stats = pd.read_csv(first / "representation_stats.csv")
    assert stats.query("partition == 'train'").N.tolist() == [8, 8]
    assert stats.query("partition != 'train'").qos_row_only.isna().all()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["classifiers_trained"] is False
    assert manifest["source_sha256_after"] == manifest["source_sha256"]
    assert manifest["status"] == "text_validated"
    assert not list(first.glob("*.npz"))


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     7654.
Pageins:                                      979764.
Pageouts:                                       9036.
Swapins:                                          12.
Swapouts:                                         30.
"""


def test_vm_stat_swap_counts_swap_not_file_pageins():
    assert cfpb_prepare._vm_stat_swap_pages(VM_STAT) == 42


def _gate_with_counters(monkeypatch, counters):
    values = iter(counters)
    clock = iter(range(10, 100, 2))
    monkeypatch.setattr(cfpb_prepare, "_RESOURCE_STATE", {})
    monkeypatch.setattr(cfpb_prepare, "_swap_counter", lambda: next(values))
    monkeypatch.setattr(cfpb_prepare.time, "monotonic", lambda: next(clock))


def test_gate_ignores_flat_swap_counter(monkeypatch):
    _gate_with_counters(monkeypatch, [5] * 6)
    for _ in range(6):
        cfpb_prepare.check_resources()


def test_gate_stops_on_three_successive_swap_increases(monkeypatch):
    import pytest
    _gate_with_counters(monkeypatch, [5, 6, 7, 8])
    for _ in range(3):
        cfpb_prepare.check_resources()
    with pytest.raises(RuntimeError, match="swap I/O"):
        cfpb_prepare.check_resources()
