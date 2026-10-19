"""Validation-only pilot summaries and idempotent, byte-preserving ledger append.

This module fits no models and accesses no source documents or test outcomes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from .data import ROOT, atomic_json, sha256
from .state import totals


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def append_exactly_once(ledger, rows, original_length, original_sha):
    """Reject changed historical bytes, conflicting IDs, or a partial append."""
    ledger = Path(ledger)
    before = ledger.read_bytes()
    if hashlib.sha256(before[:original_length]).hexdigest() != original_sha:
        raise ValueError('Historical ledger prefix changed')
    existing = list(csv.DictReader(io.StringIO(before.decode())))
    by_id = {r['experiment_id']: r for r in existing}
    fields = list(existing[0])
    new = []
    for row in rows:
        row = {k: str(row.get(k, '')) for k in fields}
        previous = by_id.get(row['experiment_id'])
        if previous is not None:
            if previous != row:
                raise ValueError('Conflicting existing ledger ID')
        else:
            new.append(row)
        by_id[row['experiment_id']] = row
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writerows(new)
    if new:
        if before and not before.endswith(b'\n'):
            raise ValueError('Existing ledger lacks final newline')
        temporary = ledger.with_suffix('.append.tmp')
        temporary.write_bytes(before + buffer.getvalue().encode())
        temporary.replace(ledger)
    if ledger.read_bytes()[:len(before)] != before:
        raise AssertionError('Ledger append changed prior bytes')
    return len(new)


def summarize(output, publish_ledger=False):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['status'] in ('preparing', 'calibration', 'pilot'):
        raise ValueError('Cannot finalize an active benchmark')
    plan = json.loads((output/'run_plan.json').read_text())
    prep = json.loads((output/'preparation.json').read_text())
    rows, checkpoints, states, costs, records = [], [], [], [], []
    for config in plan['trajectories']:
        status = manifest['trajectory_status'][config['id']]
        row = dict(**config, status=status)
        path = output/'trials'/f"{config['id']}.json"
        if path.exists():
            record = json.loads(path.read_text()); records.append(record)
            for field in ('failure_reason', 'examples_processed', 'data_passes', 'peak_model_state_bytes',
                          'model_allocated_bytes', 'model_state_registers64', 'preprocessing_state_bytes',
                          'peak_stream_state_bytes', 'peak_total_persistent_bytes', 'training_seconds',
                          'update_seconds', 'encoding_seconds', 'input_seconds', 'evaluation_seconds',
                          'worker_seconds', 'subprocess_wall_seconds', 'peak_rss_bytes',
                          'validation_presentations', 'input_coordinate_visits', 'historical_preprocessing_documents',
                          'preprocessing_passes_before_stream', 'preprocessing_initialization_seconds',
                          'external_training_spool_bytes', 'external_training_index_bytes',
                          'random_order_setup_extra_array_payload_bytes', 'training_workspace_inventory_lower_bound',
                          'harness_record_bytes'):
                row[field] = record.get(field)
            row['unique_training_examples'] = record['examples_processed']
            spools = prep['spools'][config['task']]
            row['input_integrity_bytes_scanned'] = sum(s['bytes']+s['offsets_bytes'] for s in spools.values()) * record['attempts']
            row['logical_training_spool_bytes'] = spools['train']['bytes'] if status=='completed' else None
            row['logical_validation_spool_bytes'] = spools['validation']['bytes'] * len(record['checkpoints'])
            row['io_note'] = 'Logical complete reads/checksum scans; physical random-seek read-ahead and interrupted validation bytes are not instrumented.'
            row['total_persistent_registers64'] = math.ceil(record['peak_total_persistent_bytes']/8)
            row['preprocessing_registers64'] = math.ceil(record.get('preprocessing_state_bytes', 0)/8)
            row['stream_access_registers64'] = math.ceil(record['peak_stream_state_bytes']/8)
            row['state_budget_met'] = record['peak_model_state_bytes'] <= config['budget_bytes']
            row['inference_retained_algorithm_bytes'] = record['model_allocated_bytes']
            row['inference_state_note'] = 'Unpruned learner object retained; no reduced inference export measured'
            row['examples_per_training_second'] = record['examples_processed']/record['training_seconds'] if record['training_seconds'] else None
            row['validation_examples_per_second'] = record['validation_presentations']/record['evaluation_seconds'] if record['evaluation_seconds'] else None
            row['baseline_rss_bytes'] = record['worker_timings'][0]['start_rss_bytes'] if record['worker_timings'] else None
            row['reference_accuracy'] = record['reference']['primary']['accuracy']
            row['reference_balanced_accuracy'] = record['reference']['primary']['balanced_accuracy']
            row['reference_macro_f1'] = record['reference']['primary']['macro_f1']
            for pop in ('primary', 'natural_stream'):
                for metric in ('accuracy', 'balanced_accuracy', 'macro_f1'):
                    row[f'{pop}_{metric}'] = record.get('final_'+pop, {}).get(metric)
            for group, value in totals(record['state_inventory'])['by_category'].items():
                row[group+'_bytes'] = value
                row[group+'_registers64'] = math.ceil(value/8)
            for entry in record['state_inventory'] + record.get('preprocessing_inventory', []):
                states.append(dict(trajectory_id=config['id'], **entry))
            for cp in record['checkpoints']:
                for population, metrics in cp['metrics'].items():
                    checkpoints.append(dict(**config, status=status, population=population,
                        examples_processed=cp['examples_processed'], passes=cp['passes'],
                        model_state_bytes=cp['model_state_bytes'], validation_N=metrics['N'],
                        **{k:metrics[k] for k in ('accuracy','balanced_accuracy','macro_f1')}))
            costs.append(dict(trajectory_id=config['id'], stage='training', seconds=record['training_seconds'],
                              presentations=record['examples_processed'], peak_rss_bytes=record['peak_rss_bytes']))
            costs.append(dict(trajectory_id=config['id'], stage='validation', seconds=record['evaluation_seconds'],
                              presentations=record['validation_presentations'], peak_rss_bytes=record['peak_rss_bytes']))
            costs.append(dict(trajectory_id=config['id'], stage='worker_other',
                seconds=record['worker_seconds']-record['training_seconds']-record['evaluation_seconds'],
                presentations=0, peak_rss_bytes=record['peak_rss_bytes']))
        rows.append(row)
    for stage in prep['preparation_timings']:
        costs.append(dict(trajectory_id='shared_nonstreaming_preparation', **stage))
    write_csv(output/'trajectory_summary.csv', rows)
    write_csv(output/'checkpoint_metrics.csv', checkpoints)
    write_csv(output/'state_inventory.csv', states)
    write_csv(output/'cost_ledger.csv', costs)
    completed = [r for r in rows if r['status'] == 'completed']
    groups = defaultdict(list)
    for row in completed:
        groups[(row['representation'],row['method'])].append(row)
    resources = []
    for (rep, method), rr in sorted(groups.items()):
        rates = [r['examples_per_training_second'] for r in rr]
        resources.append(dict(representation=rep, method=method, completed=len(rr),
            train_seconds=sum(r['training_seconds'] for r in rr),
            worker_seconds=sum(r['worker_seconds'] for r in rr),
            throughput_min=min(rates), throughput_median=statistics.median(rates), throughput_max=max(rates),
            algorithm_bytes_min=min(r['peak_model_state_bytes'] for r in rr),
            algorithm_bytes_max=max(r['peak_model_state_bytes'] for r in rr),
            total_persistent_bytes_min=min(r['peak_total_persistent_bytes'] for r in rr),
            total_persistent_bytes_max=max(r['peak_total_persistent_bytes'] for r in rr),
            peak_rss_bytes=max(r['peak_rss_bytes'] for r in rr)))
    write_csv(output/'method_resources.csv', resources)
    paired = []
    by_key = {(r['task'],r['representation'],r['method'],r['budget_bytes'],r['ordering']):r for r in completed}
    for key, random in by_key.items():
        if key[-1] != 'random' or key[:-1]+('chronological',) not in by_key:
            continue
        chronological = by_key[key[:-1]+('chronological',)]
        paired.append(dict(task=key[0],representation=key[1],method=key[2],budget_bytes=key[3],seed=42,
            **{m+'_random_minus_chronological':random['primary_'+m]-chronological['primary_'+m]
               for m in ('accuracy','balanced_accuracy','macro_f1')}))
    write_csv(output/'order_differences.csv', paired)
    crossings=[]
    # These are individual pilot point crossings, not k-star estimates or confidence tests.
    for cp in checkpoints:
        if cp['population'] != 'primary':
            continue
        ref = prep['references'][cp['task']]['selected']['primary']
        baseline = prep['references'][cp['task']]['majority_primary']
        for retention in (.95,.98,.99):
            crossings.append(dict(**{k:cp[k] for k in ('id','task','representation','method','ordering','budget_bytes','examples_processed','passes','model_state_bytes')},
                retention=retention, accuracy_crossing=cp['accuracy'] >= retention*ref['accuracy'],
                balanced_accuracy_crossing=cp['balanced_accuracy'] >= retention*ref['balanced_accuracy'],
                joint_crossing=all(cp[m] >= retention*ref[m] for m in ('accuracy','balanced_accuracy')),
                excess_accuracy_retention=(cp['accuracy']-baseline['accuracy'])/(ref['accuracy']-baseline['accuracy']),
                evidence='single-seed untuned validation point only; not k_star_grid or confirmation'))
    write_csv(output/'descriptive_retention_crossings.csv', crossings)
    # Full proposed matrix: 230 effective training passes and 211 validation
    # checkpoints per cell, derived from the plan's HPO + five evaluation seeds.
    forecast=[]
    for (rep,method),rr in sorted(groups.items()):
        training_per_example=[r['training_seconds']/r['examples_processed'] for r in rr]
        eval_per_example=[r['evaluation_seconds']/r['validation_presentations'] for r in rr]
        starts=[r['subprocess_wall_seconds']-r['training_seconds']-r['evaluation_seconds'] for r in rr]
        values={}
        for name,fn in [('low_observed',min),('median_observed',statistics.median),('high_observed',max)]:
            # 24 initial candidates + four continuation launches + four second
            # tuning streams + five evaluation seeds = 37 serial launches/cell.
            values[name+'_seconds']=(346703840*fn(training_per_example)+85878688*fn(eval_per_example)+5920*fn(starts))
        forecast.append(dict(representation=rep,method=method,**values))
    write_csv(output/'full_matrix_runtime_projection.csv',forecast)
    summary=dict(status=manifest['status'], planned=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
        completed=len(completed), benchmark_wall_seconds=manifest['benchmark_wall_seconds'],
        training_presentations=sum(r.get('examples_processed',0) or 0 for r in rows),
        validation_presentations=sum(r.get('validation_presentations',0) or 0 for r in rows),
        training_seconds=sum(r.get('training_seconds',0) or 0 for r in rows),
        evaluation_seconds=sum(r.get('evaluation_seconds',0) or 0 for r in rows),
        all_recorded_algorithm_states_fit=all(r.get('state_budget_met',True) for r in rows),
        process_peak_rss_bytes=max([r.get('peak_rss_bytes',0) or 0 for r in rows]+[s['peak_rss_bytes'] for s in prep['preparation_timings']]),
        preservation_mismatches=manifest['preservation_mismatches'],frozen_artifacts_verified=manifest['frozen_artifacts_verified_after'],
        test_predictions=0, tuning_trajectories=0, evaluation_seeds=[42], across_seed_uncertainty='not estimable',
        batch_reference_fits=8, additional_raw_scaled_equivalence_fits=8,
        k_star_grid='not estimated: untuned one-seed diagnostic pilot',
        raw_data_or_model_cache_committed=False, analyzer_sha256=sha256(Path(__file__)),
        full_forecast=dict(training_presentations=2773630720, validation_presentations=85878688*8,
            worker_starts=5920*8, complete_method_track_coverage=len(forecast)==8,
            **{name+'_days':sum(r[name+'_seconds'] for r in forecast)/86400 for name in ('low_observed','median_observed','high_observed')},
            limitations='Calculated extrapolation, not measured full benchmark; frozen two-task rates transferred to other tasks, budgets and multipass/HPO settings. No bootstrap/reference/tuning-management allowance. Range is not a confidence interval.'))
    atomic_json(output/'analysis.json', summary)
    if publish_ledger:
        publish_rows=[]
        for row in completed:
            notes=dict(run_dir=str(output.relative_to(ROOT) if output.is_absolute() else output),
                task=row['task'],ordering=row['ordering'],budget_bytes=row['budget_bytes'],
                algorithm_bytes=row['peak_model_state_bytes'],total_persistent_bytes=row['peak_total_persistent_bytes'],
                training_seconds=row['training_seconds'],passes=row['data_passes'],
                interpretation='measured classical, validation only, untuned, one seed; no k-star or quantum claim')
            for metric in ('accuracy','balanced_accuracy','macro_f1'):
                publish_rows.append(dict(experiment_id=f"streaming-{manifest['run_id']}-{row['id']}-{metric}",
                    dataset='CFPB frozen binary primary validation', date=manifest['started_utc'][:10],
                    git_commit=manifest['root_commit'],preprocessing=row['representation'],
                    n_samples=row['examples_processed'],n_features=50000 if row['representation']=='A_frozen' else '',
                    max_sparsity='',sketch_method=row['method'],sketch_dimension='',seed=42,
                    model='unweighted no-intercept half-squared-loss streaming learner',
                    metric='validation_'+metric,result=row['primary_'+metric],notes=json.dumps(notes,sort_keys=True)))
        count=append_exactly_once(ROOT/'results/experiment_log.csv',publish_rows,
                                  manifest['prior_ledger_bytes'],manifest['prior_ledger_sha256'])
        atomic_json(output/'ledger_append.json',dict(run_id=manifest['run_id'],expected_rows=len(publish_rows),
            historical_prefix_bytes=manifest['prior_ledger_bytes'],historical_prefix_sha256=manifest['prior_ledger_sha256'],
            experiment_ids=[r['experiment_id'] for r in publish_rows],ledger_sha256=sha256(ROOT/'results/experiment_log.csv')))
        print(f'Appended {count} new ledger rows')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output');parser.add_argument('--append-ledger',action='store_true')
    args=parser.parse_args();summarize(args.output,args.append_ledger)
