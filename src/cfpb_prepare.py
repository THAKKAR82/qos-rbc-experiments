"""Prepare/audit CFPB v1 text only: python -m src.cfpb_prepare.

Fresh, separate output directories; never fits classifiers or writes Bank files.
Sparse matrices are rebuilt from immutable source + manifests, not persisted.
"""

import os
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from threadpoolctl import threadpool_limits

from .cfpb import (CLASSES, SOURCE_COLUMNS, SOURCE_SHA256, SPEC_VERSION, SPLITS,
                   curate_cfpb, curation_gates, source_metadata)
from .text_preprocessing import fit_text_partitions, text_matrix_stats

ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_STATE = {}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git_commit(directory):
    if not (directory / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def check_resources(*, before_stage=False):
    available = psutil.virtual_memory().available
    rss = psutil.Process().memory_info().rss
    if available < (4 if before_stage else 2) * 2**30 or rss > 4 * 2**30:
        raise RuntimeError(f"Memory gate: available={available}, process_RSS={rss}")
    now = time.monotonic()
    if _RESOURCE_STATE.get("swap_available") is not False and now - _RESOURCE_STATE.get("sample_time", 0) >= 1:
        try:
            swap = psutil.swap_memory()
        except (OSError, PermissionError) as exc:
            _RESOURCE_STATE.update(swap_available=False, swap_error=str(exc))
        else:
            counter = swap.sin + swap.sout
            previous = _RESOURCE_STATE.get("swap_counter", counter)
            rising = _RESOURCE_STATE.get("rising_samples", 0) + 1 if counter > previous else 0
            _RESOURCE_STATE.update(swap_available=True, swap_counter=counter,
                                   sample_time=now, rising_samples=rising)
            if rising >= 3:
                raise RuntimeError("Memory gate: swap I/O increased in three successive >=1s samples")


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _read_source(source, output):
    """Stream raw rows, retain only valid narratives; audit every original row."""
    parts, identifiers = [], []
    raw_counts = Counter()
    total = 0
    path = output / "_source_rows.csv"
    for chunk in pd.read_csv(source, usecols=list(SOURCE_COLUMNS), dtype=str,
                             keep_default_na=False, chunksize=100000):
        check_resources()
        meta = source_metadata(chunk)
        meta.to_csv(path, mode="a", header=total == 0, index=False)
        total += len(chunk)
        identifiers.extend(meta.loc[~meta.invalid_id, "complaint_id"])
        for (product, split, reason), n in meta.groupby(["product", "split", "reason"],
                                                       dropna=False).size().items():
            raw_counts[(product, split, reason)] += int(n)
        keep = meta.reason.eq("").to_numpy()
        parts.append(chunk.loc[keep].copy())
        print(f"Source scan: {total:,} complaints", flush=True)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate complaint IDs in full source block release")
    if not parts or not any(len(p) for p in parts):
        raise ValueError("No eligible narrative rows")
    return pd.concat(parts, ignore_index=True), raw_counts, total


def _write_complete_audit(output, curated):
    """Merge narrative audit onto streamed source metadata without raw prose."""
    extra = curated.rename(columns={"reason": "curation_reason"})
    base_columns = set(pd.read_csv(output / "_source_rows.csv", nrows=0).columns)
    extra_columns = ["complaint_id"] + [c for c in extra if c not in base_columns]
    first = True
    reason_counts = Counter()
    for source in pd.read_csv(output / "_source_rows.csv", dtype=str, keep_default_na=False,
                              chunksize=100000):
        merged = source.merge(extra[extra_columns], on="complaint_id", how="left", validate="many_to_one")
        merged["reason"] = merged.curation_reason.fillna(merged.reason)
        if merged.reason.eq("").any():
            raise AssertionError("Unmapped narrative-bearing source row")
        merged = merged.drop(columns="curation_reason")
        for col in ("primary", "natural_stream", "structured_heading", "empty_model_text",
                    "missing_product", "raw_conflict", "model_conflict", "template_conflict"):
            # Unavailable rows were never assessed for downstream artifact/group flags.
            if col in ("primary", "natural_stream"):
                merged[col] = merged[col].eq(True)
        reason_counts.update(merged.reason.value_counts().to_dict())
        merged.to_csv(output / "curation_audit.csv", index=False, mode="a", header=first)
        first = False
    (output / "_source_rows.csv").unlink()
    return dict(reason_counts)


def prepare(source, output_root):
    """Run the frozen text-only stage; return its fresh output path."""
    _RESOURCE_STATE.clear()
    check_resources(before_stage=True)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    spec = ROOT / "docs/cfpb_benchmark_spec.md"
    manifest = dict(run_id=run_id, spec_version=SPEC_VERSION, spec_sha256=sha256_file(spec),
                    source_path=str(Path(source).resolve()), source_sha256=sha256_file(source),
                    date_utc=now.isoformat(), root_commit=git_commit(ROOT),
                    reference_commit=git_commit(ROOT / "reference/quantum-oracle-sketching"),
                    acquisition_provenance=None, status="running_text_only", classifiers_trained=False,
                    python=sys.version, platform=platform.platform(), seed=42, chunksize=100000,
                    threads={k: os.environ[k] for k in os.environ if k in
                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
                    packages={name: importlib.metadata.version(name) for name in
                              ("numpy", "pandas", "scipy", "scikit-learn", "psutil")},
                    code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in
                                 (Path(__file__), ROOT / "src/cfpb.py", ROOT / "src/text_preprocessing.py")},
                    memory_monitoring="RSS/available RAM checked by chunk and join; swap sampled at >=1s intervals where accessible",
                    swap_monitoring=dict(_RESOURCE_STATE),
                    sparse_matrices_saved=False)
    write_json(output / "manifest.json", manifest)
    runtime = []
    started = time.perf_counter()
    try:
        if manifest["source_sha256"] != SOURCE_SHA256:
            raise ValueError("Source SHA-256 differs from frozen specification")
        t = time.perf_counter()
        narratives, raw_counts, total = _read_source(source, output)
        runtime.append(dict(stage="source_scan", seconds=time.perf_counter() - t, peak_rss_bytes=peak_rss()))
        check_resources(before_stage=True)
        print(f"Curating {len(narratives):,} narratives across all products; complete prefix join", flush=True)
        t = time.perf_counter()
        result = curate_cfpb(narratives, check_resources=check_resources)
        del narratives
        runtime.append(dict(stage="curation", seconds=time.perf_counter() - t, peak_rss_bytes=peak_rss()))
        audit = result.audit
        gates = curation_gates(audit)
        reasons = _write_complete_audit(output, audit)
        result.groups.to_csv(output / "duplicate_groups.csv", index=False)
        result.edges.to_csv(output / "near_template_edges.csv", index=False)
        primary = audit[audit.primary].copy()
        split_rows = audit[audit.primary | audit.natural_stream].copy()
        split_rows[["complaint_id", "date_received", "product", "label", "split",
                    "template_group_id", "primary", "natural_stream", "stratum"]].to_csv(
                        output / "split_manifest.csv", index=False)
        population = []
        for (product, split, reason), n in sorted(raw_counts.items()):
            population.append(dict(stage="source", product=product, split=split,
                                   reason=reason or "narrative_candidate", n=n))
        for name, a in (("primary", primary), ("natural_stream", audit[audit.natural_stream])):
            for product in CLASSES:
                for split in SPLITS:
                    population.append(dict(stage=name, product=product, split=split, reason="",
                                           n=int(((a["product"] == product) & (a["split"] == split)).sum())))
        pd.DataFrame(population).to_csv(output / "population_counts.csv", index=False)
        counts = pd.crosstab(primary["product"], primary["split"]).reindex(index=CLASSES, columns=SPLITS, fill_value=0)
        counts.to_csv(output / "curated_class_counts.csv")
        daily = []
        for name, a in (("raw_narratives", audit), ("primary", primary)):
            x = a.groupby(["date_received", "product"]).size().rename("n").reset_index()
            x["stage"] = name
            daily.append(x)
        pd.concat(daily).to_csv(output / "daily_class_counts.csv", index=False)
        duplicate_summary = {}
        for kind, g in result.groups.groupby("kind"):
            repeated = g[g.member_count > 1]
            duplicate_summary[kind] = dict(groups=len(g), repeated_groups=len(repeated),
                                           excess_members=int((repeated.member_count - 1).sum()),
                                           conflicting_groups=int(g.conflicting.sum()),
                                           conflicting_members=int(g.loc[g.conflicting, "member_count"].sum()))
        summary = dict(total_source_rows=total, narrative_candidates=len(audit),
                       curated_N=len(primary), per_class_counts=counts.to_dict(),
                       primary_split_counts=primary.split.value_counts().to_dict(),
                       exclusion_reason_counts=reasons, duplicate_groups=duplicate_summary,
                       near_templates=result.diagnostics, gates=gates,
                       primary_class_exclusions=pd.crosstab(
                           audit.loc[audit["product"].isin(CLASSES), "product"],
                           audit.loc[audit["product"].isin(CLASSES), "reason"]).to_dict(),
                       flag_counts={c: int(audit[c].sum()) for c in
                                    ("structured_heading", "empty_model_text", "raw_conflict",
                                     "model_conflict", "template_conflict")})
        write_json(output / "curation_summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        if gates["blocking"]:
            raise RuntimeError("Curation release gates failed; inspect curation_summary.json")
        # Validity review gates concern future model fitting. No model is fitted here.
        stats, vectorizers, zero_rows = [], {}, []
        train = primary[primary.split == "train"]
        heldout_rows = {s: primary[primary.split == s] for s in ("validation", "test")}
        heldout_rows.update({"natural_stream_" + s: audit[audit.natural_stream & audit.split.eq(s)]
                             for s in ("validation", "test")})
        train_texts = [result.model_texts[c] for c in train.complaint_id]
        heldout_texts = {name: [result.model_texts[c] for c in rows.complaint_id]
                        for name, rows in heldout_rows.items()}
        for config, reference in (("primary", False), ("unigram", True)):
            check_resources(before_stage=True)
            t = time.perf_counter()
            vectorizer, matrices = fit_text_partitions(train_texts, heldout_texts, reference=reference)
            runtime.append(dict(stage=config + "_fit_transform", seconds=time.perf_counter() - t,
                                peak_rss_bytes=peak_rss()))
            prefix = "" if config == "primary" else "unigram_"
            vocabulary = output / (prefix + "vocabulary.csv")
            idf_path = output / (prefix + "idf.npy")
            pd.DataFrame({"feature_index": range(len(vectorizer.vocabulary_)),
                          "token": vectorizer.get_feature_names_out()}).to_csv(vocabulary, index=False)
            np.save(idf_path, vectorizer.idf_, allow_pickle=False)
            params = vectorizer.get_params()
            params["dtype"] = "float32"
            vectorizers[config] = dict(parameters=params, D=len(vectorizer.vocabulary_),
                                       vocabulary_file=vocabulary.name, vocabulary_sha256=sha256_file(vocabulary),
                                       idf_file=idf_path.name, idf_sha256=sha256_file(idf_path),
                                       training_ids_sha256=hashlib.sha256(
                                           "\n".join(train.complaint_id).encode()).hexdigest(),
                                       english_stop_words_sha256=hashlib.sha256(
                                           "\n".join(sorted(ENGLISH_STOP_WORDS)).encode()).hexdigest())
            for name, matrix in matrices.items():
                check_resources()
                s = text_matrix_stats(matrix)
                population_name = "natural_stream" if name.startswith("natural_stream_") else "primary"
                split = name.removeprefix("natural_stream_")
                s.update(configuration_id=config, population=population_name, partition=split,
                         run_id=run_id, spec_version=SPEC_VERSION, seed=None)
                # Holdouts have descriptive sparsity only, not training QOS estimates.
                if name != "train":
                    s["qos_row_only"] = s["qos_row_or_column"] = None
                stats.append(s)
                rows = train if name == "train" else heldout_rows[name]
                for cid in rows.loc[np.diff(matrix.indptr) == 0, "complaint_id"]:
                    zero_rows.append(dict(configuration_id=config, population=population_name,
                                          partition=split, complaint_id=cid))
                if population_name == "primary" and s["N"] and s["zero_rows"] / s["N"] > .01:
                    gates["blocking"].append(f"over_one_percent_zero_features:{config}:{name}")
            print(f"{config}: D={len(vectorizer.vocabulary_):,}; train nnz={matrices['train'].nnz:,}", flush=True)
            del matrices, vectorizer
        pd.DataFrame(stats).to_csv(output / "representation_stats.csv", index=False)
        accounting = pd.DataFrame(stats).query("partition == 'train'").copy()
        accounting["interpretation"] = "theoretical LS-SVM per-binary-subproblem accounting; not hardware memory or quantum advantage"
        accounting.to_csv(output / "resource_accounting.csv", index=False)
        pd.DataFrame(zero_rows, columns=["configuration_id", "population", "partition", "complaint_id"]).to_csv(
            output / "zero_feature_rows.csv", index=False)
        write_json(output / "vectorizer_manifest.json", vectorizers)
        pd.DataFrame(runtime).to_csv(output / "runtime_memory.csv", index=False)
        gates["modeling_permitted"] = not gates["blocking"] and not gates["validity_review_required"]
        write_json(output / "release_gates.json", gates)
        # No classifier metrics are fabricated. Ledger contains accounting only.
        ledger = []
        for row in stats:
            if row["partition"] != "train":
                continue
            for field, metric in (("qos_row_only", "qos_size_row_only"),
                                  ("qos_row_or_column", "qos_size_row_or_column")):
                ledger.append(dict(experiment_id=run_id + "-" + row["configuration_id"],
                                   dataset="CFPB Jan-Feb 2026 primary4", date=now.date().isoformat(),
                                   git_commit=manifest["root_commit"],
                                   preprocessing=row["configuration_id"] + ": train-only TF-IDF; spec 1.0",
                                   n_samples=row["N"], n_features=row["D"], max_sparsity=row["max_row_nnz"],
                                   sketch_method="none", sketch_dimension=None, seed=None,
                                   model="none; theoretical LS-SVM subproblem", metric=metric,
                                   result=row[field], notes=json.dumps(dict(
                                       run_id=run_id, spec_sha256=manifest["spec_sha256"],
                                       source_sha256=manifest["source_sha256"], partition="train",
                                       population="primary", dtype=row["dtype"],
                                       max_column_nnz=row["max_column_nnz"],
                                       source="resource_accounting.csv", classifiers_trained=False,
                                       vocabulary_sha256=vectorizers[row["configuration_id"]]["vocabulary_sha256"],
                                       interpretation="theoretical per-subproblem accounting; not hardware usage or quantum advantage"))))
        pd.DataFrame(ledger).to_csv(output / "experiment_log.csv", index=False)
        report = ["# CFPB curation and text-only validation", "",
                  f"Run: `{run_id}`; spec {SPEC_VERSION}.", "",
                  f"Curated primary N: **{len(primary)}**. No classifiers trained.", "",
                  "## Primary class counts", "", "```csv", counts.to_csv().strip(), "```", "",
                  "## Complete grouping", "", json.dumps(result.diagnostics, sort_keys=True), "",
                  "## Release gates", "", "```json", json.dumps(gates, indent=2), "```", "",
                  "Any validity-review gate must be resolved before classifier fitting. "
                  "The grouping is complete; review flags are not waived by text feasibility.", "",
                  "## Representations", "", "```csv",
                  pd.DataFrame(stats).query("population == 'primary'").to_csv(index=False).strip(), "```", "",
                  "Vocabulary and IDF were fitted only on curated January rows. "
                  "All subsequent rows were transformed without refitting. No matrices were densified or saved.", "",
                  "QOS quantities refer only to training matrices and are theoretical binary-subproblem "
                  "accounting under the two stated sparsity conventions, not hardware memory, total "
                  "multiclass cost, or evidence of quantum advantage.", "",
                  "Source/model fitting uses offline globally audited groups; future labels/text inform "
                  "curation, not vocabulary or IDF fitting. Raw narratives are absent from the audit CSVs. "
                  "Vocabulary files remain derived text and are not privacy-cleared publication artifacts.", ""]
        (output / "benchmark_report.md").write_text("\n".join(report))
        final_hash = sha256_file(source)
        if final_hash != manifest["source_sha256"]:
            raise RuntimeError("Source changed during preparation")
        manifest.update(status="text_validated_review_required" if gates["validity_review_required"] else
                        "text_validated", source_sha256_after=final_hash,
                        total_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss(),
                        swap_monitoring=dict(_RESOURCE_STATE),
                        curation_complete=result.diagnostics["complete"], gates=gates,
                        artifacts_sha256={p.name: sha256_file(p) for p in sorted(output.iterdir())
                                          if p.is_file() and p.name != "manifest.json"})
        if gates["blocking"]:
            raise RuntimeError("Representation release gates failed; inspect release_gates.json")
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        total_seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/CCDB_Export_15_January_2026_through_February_2026.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "validation/cfpb")
    args = parser.parse_args()
    # CLI-only configuration: importing helpers/tests must not change Bank's
    # environment. threadpool_limits also covers already loaded BLAS libraries.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    with threadpool_limits(limits=1):
        print(prepare(args.source, args.output_root))


if __name__ == "__main__":
    main()
