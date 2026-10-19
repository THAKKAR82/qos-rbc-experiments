"""Offline integrity and non-streaming reference preparation, validation only."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr
from src.cfpb import CLASSES, SOURCE_SHA256, clean_narrative
from src.cfpb_prepare import check_resources
from src.cfpb_baselines import measured
from src.holdout_evaluation import classification_metrics
from src.text_preprocessing import make_tfidf_vectorizer
from .data import ROOT, PREPARATION, TASKS, Encoder, atomic_json, sha256, task_mask, write_spool
from .state import deep_bytes, totals

SOURCE = ROOT / 'data/CCDB_Export_15_January_2026_through_February_2026.csv'


def score(y, pred):
    metrics, perclass, confusion, _ = classification_metrics(y, pred, labels=(-1, 1))
    return dict(**metrics, per_class=perclass, confusion=confusion.tolist(), N=len(y))


def protected_hashes():
    import subprocess
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    return {p: sha256(ROOT / p) for p in paths if p and p != 'results/experiment_log.csv'}


def verify_frozen():
    count = 0
    for d in ('validation/cfpb/20261014T192947Z-7b6668f9',
              'validation/cfpb/20261014T204807Z-2270924a',
              'validation/cfpb/20261016T125619Z-30526eac', 'results/audit/cfpb_step1'):
        root = ROOT / d
        for p, expected in json.loads((root / 'manifest.json').read_text())['artifacts_sha256'].items():
            if sha256(root / p) != expected:
                raise ValueError('Frozen artifact integrity failure: ' + d + '/' + p)
            count += 1
    return count


def prepare(output):
    output = Path(output); cache = output / 'data_cache'; cache.mkdir(exist_ok=True)
    timings = []
    with measured('source_integrity_and_train_validation_rehydration', timings):
        if sha256(SOURCE) != SOURCE_SHA256:
            raise ValueError('Source SHA mismatch')
        rows = pd.read_csv(PREPARATION / 'split_manifest.csv', dtype=str, keep_default_na=False)
        rows = rows[((rows.split == 'train') & (rows.primary == 'True')) |
                    ((rows.split == 'validation') & ((rows.primary == 'True') | (rows.natural_stream == 'True')))].copy()
        assert not rows.split.eq('test').any()
        wanted = set(rows.complaint_id)
        parts = []
        for chunk in pd.read_csv(SOURCE, usecols=['Complaint ID', 'Consumer complaint narrative', 'Product', 'Date received'],
                                 dtype=str, keep_default_na=False, chunksize=100000):
            check_resources()
            parts.append(chunk[chunk['Complaint ID'].isin(wanted)].copy())
        raw = pd.concat(parts, ignore_index=True)
        rows = rows.merge(raw, left_on='complaint_id', right_on='Complaint ID', validate='one_to_one')
        if set(rows.complaint_id) != wanted or not rows.Product.eq(rows['product']).all() or not rows['Date received'].eq(rows.date_received).all():
            raise ValueError('Frozen source membership/metadata mismatch')
        rows['text'] = rows['Consumer complaint narrative'].map(clean_narrative)
        accepted = {}
        for chunk in pd.read_csv(PREPARATION / 'curation_audit.csv', usecols=['complaint_id', 'model_text_hash'],
                                 dtype=str, keep_default_na=False, chunksize=100000):
            subset = chunk[chunk.complaint_id.isin(wanted)]
            accepted.update(zip(subset.complaint_id, subset.model_text_hash))
        if not all(hashlib.sha256(r.text.encode()).hexdigest() == accepted[r.complaint_id] for r in rows.itertuples()):
            raise ValueError('Frozen cleaned-text hash mismatch')
        rows['numeric_id'] = rows.complaint_id.astype(np.int64)
        rows = rows.sort_values(['date_received', 'numeric_id']).reset_index(drop=True)
        rows['label'] = rows.label.astype(int)
        for col in ('primary', 'natural_stream'):
            rows[col] = rows[col].eq('True')
        if not rows['product'].eq(rows.label.map(dict(enumerate(CLASSES)))).all():
            raise ValueError('Label mapping drift')
        train, val = rows[rows.split.eq('train')], rows[rows.split.eq('validation')]
        assert len(train) == 13459 and int(val.primary.sum()) == 3033 and int(val.natural_stream.sum()) == 3634
        if rows[rows.primary].template_group_id.duplicated().any():
            raise ValueError('Primary text-group leakage')
        del raw, parts, accepted
    with measured('frozen_representation_and_rowwise_correctness', timings):
        vm = json.loads((PREPARATION / 'vectorizer_manifest.json').read_text())['primary']
        assert hashlib.sha256('\n'.join(train.complaint_id).encode()).hexdigest() == vm['training_ids_sha256']
        v = make_tfidf_vectorizer()
        X = v.fit_transform(train.text.tolist()).tocsr()
        X.sum_duplicates(); X.eliminate_zeros(); X.sort_indices()
        V = v.transform(val.text.tolist()).tocsr(); V.sort_indices()
        encoder = Encoder('A_frozen')
        assert v.vocabulary_ == encoder.vectorizer.vocabulary_
        np.testing.assert_array_equal(v.idf_, encoder.vectorizer.idf_)
        for i, text in enumerate(train.text):
            if i % 256 == 0:
                check_resources()
            k, x = encoder.encode(text)
            start, end = X.indptr[i:i+2]
            np.testing.assert_array_equal(k[:, 0], X.indices[start:end])
            np.testing.assert_array_equal(x, X.data[start:end].astype(np.float64))
        preprocessing = {rep: totals(Encoder(rep).state_inventory()) for rep in ('A_frozen', 'B_causal')}
    with measured('binary_ridge_reference_and_validation_only_equivalence', timings):
        # Never load test-performance columns into the comparison dataframe.
        audit = pd.read_csv(ROOT / 'results/audit/cfpb_step1/summary.csv',
                            usecols=['task_id', 'alpha', 'sigma_max', 'val_accuracy', 'val_balanced_accuracy'])
        references, spool_info = {}, {}
        for task, spec in TASKS.items():
            tm, vm = task_mask(train.label.to_numpy(), task), task_mask(val.label.to_numpy(), task)
            tr, va = train[tm], val[vm]
            xt, xv = X[tm].astype(np.float64), V[vm].astype(np.float64)
            yt = np.where(tr.label == spec['positive'], 1, -1)
            yv = np.where(va.label == spec['positive'], 1, -1)
            primary = va.primary.to_numpy(); natural = va.natural_stream.to_numpy()
            assert len(yt) == spec['N'] and int(primary.sum()) == spec['validation_N']
            options = []
            for alpha in (.1, 1., 10., 100.):
                check_resources()
                original = audit[(audit.task_id == spec['audit_id']) & (audit.alpha == alpha)].iloc[0]
                sigma = float(original.sigma_max)
                fit_start = time.perf_counter()
                raw = lsqr(xt, yt, damp=np.sqrt(alpha), atol=1e-10, btol=1e-10, iter_lim=20000)
                scaled = lsqr(xt / sigma, yt, damp=np.sqrt(alpha / sigma**2), atol=1e-10, btol=1e-10, iter_lim=20000)
                if raw[1] not in (1, 2, 4, 5) or scaled[1] not in (1, 2, 4, 5):
                    raise ValueError('Ridge LSQR did not converge')
                logits = np.asarray(xv @ raw[0]); comparison = np.asarray(xv @ (scaled[0] / sigma))
                pred = np.where(logits >= 0, 1, -1)
                np.testing.assert_array_equal(pred, np.where(comparison >= 0, 1, -1))
                np.testing.assert_allclose(logits, comparison, atol=1e-7, rtol=0)
                metrics = score(yv[primary], pred[primary])
                np.testing.assert_allclose([metrics['accuracy'], metrics['balanced_accuracy']],
                                           [original.val_accuracy, original.val_balanced_accuracy], atol=1e-12, rtol=0)
                residual = np.linalg.norm(xt.T @ (xt @ raw[0] - yt) + alpha * raw[0]) / np.linalg.norm(xt.T @ yt)
                if residual > 1e-7:
                    raise ValueError('Ridge normal-equation residual too large')
                options.append(dict(alpha=alpha, primary=metrics, natural_stream=score(yv[natural], pred[natural]),
                                    lsqr_iterations=int(raw[2]), normal_equation_residual=float(residual),
                                    scaled_unscaled_max_logit_difference=float(np.max(np.abs(logits-comparison))),
                                    fit_and_equivalence_seconds=time.perf_counter()-fit_start,
                                    model_coefficient_bytes=int(raw[0].nbytes), model_allocated_bytes=deep_bytes(raw[0])))
            chosen = min(options, key=lambda r: (-r['primary']['accuracy'], -r['alpha']))
            majority = 1 if np.count_nonzero(yt == 1) >= np.count_nonzero(yt == -1) else -1
            references[task] = dict(selected_alpha=chosen['alpha'], selected=chosen, alpha_sweep=options,
                                    majority_primary=score(yv[primary], np.full(primary.sum(), majority)),
                                    task=spec, balanced_model=False, fit_intercept=False,
                                    reference_kind='non-streaming full frozen-feature Ridge')
            spool_info[task] = {}
            for name, frame in [('train', tr), ('validation', va)]:
                path = cache / f'{task}_{name}.jsonl'
                def records():
                    for r in frame.itertuples():
                        yield dict(id=r.complaint_id, date=r.date_received, split=r.split,
                                   y=1 if r.label == spec['positive'] else -1, text=r.text,
                                   primary=bool(r.primary), natural_stream=bool(r.natural_stream), stratum=r.stratum)
                n = write_spool(path, records())
                offsets = path.with_suffix('.offsets.npy')
                spool_info[task][name] = dict(N=n, path=str(path.relative_to(output)),
                                             sha256=sha256(path), bytes=path.stat().st_size,
                                             offsets_bytes=offsets.stat().st_size, offsets_sha256=sha256(offsets))
    result = dict(references=references, spools=spool_info, preprocessing=preprocessing,
                  preparation_timings=timings, source_sha256=SOURCE_SHA256,
                  source_bytes=SOURCE.stat().st_size, source_integrity_byte_passes=1,
                  source_extraction_passes=1, historical_A_preprocessing_unique_documents=13459,
                  pilot_A_reference_fit_documents=13459, rowwise_correctness_documents=13459,
                  no_test_rows_rehydrated=True, no_test_predictions=True,
                  offline_dataframe_bytes=int(rows.memory_usage(deep=True).sum()),
                  fitted_train_csr_payload_bytes=int(X.data.nbytes+X.indices.nbytes+X.indptr.nbytes))
    atomic_json(output / 'preparation.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    prepare(parser.parse_args().output)
