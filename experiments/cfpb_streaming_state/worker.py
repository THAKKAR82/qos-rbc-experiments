"""One bounded learner process, with resumable orderly-stop checkpoints."""
import argparse
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
from src.cfpb_baselines import measured
from src.cfpb_prepare import check_resources
from .data import ROOT, TASKS, Encoder, Stream, atomic_json, sha256
from .methods import Learner, budgeted
from .prepare import score
from .state import deep_bytes, totals


class Deadline(Exception):
    def __init__(self, message, presentations=0):
        super().__init__(message)
        self.presentations = presentations


def evaluate(model, encoder, spool, n, deadline, stop):
    groups = {'primary': [[], []], 'natural_stream': [[], []]}
    stream = Stream(spool, n, role='validation')
    max_workspace = 0
    try:
        for i, row in enumerate(stream):
            if i % 64 == 0:
                check_resources()
                if stop[0] or time.monotonic() >= deadline:
                    raise Deadline('Validation interrupted before completion', stream.cursor)
            keys, x = encoder.encode(row['text'])
            pred = 1 if model.predict(keys, x) >= 0 else -1
            for population in ('primary', 'natural_stream'):
                if row[population]:
                    groups[population][0].append(row['y']); groups[population][1].append(pred)
            if row['natural_stream']:
                name = 'natural_stratum_' + row['stratum']
                groups.setdefault(name, [[], []])
                groups[name][0].append(row['y']); groups[name][1].append(pred)
            max_workspace = max(max_workspace, deep_bytes(row) + keys.nbytes + x.nbytes)
        return {name: score(np.asarray(y), np.asarray(p)) for name, (y, p) in groups.items() if y}, dict(
            validation_examples=n, validation_workspace_inventory_lower_bound=max_workspace + deep_bytes(groups),
            validation_stream_state_bytes=stream.state_bytes())
    finally:
        stream.close()


def checkpoint(output, trial, model, record):
    """State generation first, then atomic pointer; never append duplicate rows."""
    meta, arrays = model.snapshot()
    state = output / 'data_cache' / f'{trial}_state_{model.t}.npz'
    tmp = state.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, metadata=json.dumps(meta, sort_keys=True), **arrays)
    tmp.replace(state)
    record['examples_processed'] = model.t
    record['data_passes'] = model.t / TASKS[record['task']]['N']
    record['state_path'] = str(state.relative_to(output))
    record['state_sha256'] = sha256(state)
    record['state_inventory'] = model.state_inventory()
    record['model_allocated_bytes'] = model.state_bytes()
    record['model_state_registers64'] = math.ceil(record['model_allocated_bytes'] / 8)
    record['harness_record_bytes'] = deep_bytes(record)
    atomic_json(output / 'trials' / f'{trial}.json', record)
    for old in (output / 'data_cache').glob(f'{trial}_state_*.npz'):
        if old != state:
            old.unlink()


def run(output, config, seconds):
    output = Path(output); trial = config['id']; path = output / 'trials' / f'{trial}.json'
    prep = json.loads((output / 'preparation.json').read_text())
    spec = TASKS[config['task']]; spool = prep['spools'][config['task']]
    for info in spool.values():
        if sha256(output / info['path']) != info['sha256']:
            raise ValueError('Input spool was altered')
        if sha256((output / info['path']).with_suffix('.offsets.npy')) != info['offsets_sha256']:
            raise ValueError('Input offset index was altered')
    deadline = time.monotonic() + max(0., seconds - 5.)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    timings = []
    if path.exists():
        record = json.loads(path.read_text())
        if record['status'] == 'completed':
            return record
        if record['status'] != 'paused':
            raise ValueError('Hard failures are not automatically retried')
        if record['config'] != config:
            raise ValueError('Restart configuration mismatch')
        state_path = output / record['state_path']
        if sha256(state_path) != record['state_sha256']:
            raise ValueError('Checkpoint integrity failure')
        with np.load(state_path, allow_pickle=False) as saved:
            model = Learner.restore(json.loads(str(saved['metadata'])), saved)
        if model.t != record['examples_processed']:
            raise ValueError('Checkpoint cursor mismatch')
    else:
        model = budgeted(config['method'], config['budget_bytes'], spec['N'])
        record = dict(**{k: v for k, v in config.items() if k != 'id'}, id=trial, config=config,
                      status='running', checkpoints=[], training_seconds=0., update_seconds=0.,
                      encoding_seconds=0., input_seconds=0., evaluation_seconds=0.,
                      validation_presentations=0, input_coordinate_visits=0,
                      peak_model_state_bytes=model.state_bytes(), peak_stream_state_bytes=0,
                      peak_total_persistent_bytes=0, training_workspace_inventory_lower_bound=0,
                      attempts=0, fit_intercept=False, class_weight=None,
                      reference=prep['references'][config['task']]['selected'],
                      historical_preprocessing_documents=13459 if config['representation'] == 'A_frozen' else 0,
                      preprocessing_passes_before_stream=1 if config['representation'] == 'A_frozen' else 0,
                      external_training_spool_bytes=spool['train']['bytes'],
                      external_training_index_bytes=spool['train']['offsets_bytes'])
    record['attempts'] += 1
    stream = None
    try:
        with measured('trajectory', timings):
            encoder_start = time.perf_counter()
            encoder = Encoder(config['representation'])
            record['preprocessing_initialization_seconds'] = time.perf_counter() - encoder_start
            record['preprocessing_inventory'] = encoder.state_inventory()
            pre_bytes = totals(record['preprocessing_inventory'])['bytes']
            record['preprocessing_state_bytes'] = pre_bytes
            stream = Stream(output / spool['train']['path'], spec['N'], role='train',
                            order=config['ordering'], cursor=model.t)
            record['random_order_setup_extra_array_payload_bytes'] = spec['N'] * 8 if config['ordering'] == 'random' else 0
            record['random_order_setup_note'] = 'Extra temporary int64 permutation payload, beyond maintained int32 order and int64 offsets; RNG/native setup temporaries captured only by sampled RSS.'
            points = sorted(set(math.ceil(spec['N'] * f) for f in (.125, .25, .5, 1.)))
            while True:
                # Pending checkpoint evaluation is performed before new training
                # when resuming an orderly stop in validation.
                if model.t in points and not any(c['examples_processed'] == model.t for c in record['checkpoints']):
                    previous_t = model.t
                    evaluation_start = time.perf_counter()
                    try:
                        metrics, eval_resources = evaluate(model, encoder, output / spool['validation']['path'],
                                                            spool['validation']['N'], deadline, stopped)
                    except Deadline as exc:
                        record['evaluation_seconds'] += time.perf_counter() - evaluation_start
                        record['validation_presentations'] += exc.presentations
                        raise
                    record['evaluation_seconds'] += time.perf_counter() - evaluation_start
                    record['validation_presentations'] += spool['validation']['N']
                    if model.t != previous_t:
                        raise AssertionError('Validation changed training counter')
                    record['checkpoints'].append(dict(examples_processed=model.t, passes=model.t/spec['N'],
                        model_state_bytes=model.state_bytes(), metrics=metrics, **eval_resources))
                    record['status'] = 'completed' if model.t == spec['N'] else 'running'
                    checkpoint(output, trial, model, record)
                if model.t == spec['N']:
                    break
                if stopped[0] or time.monotonic() >= deadline:
                    raise Deadline('Orderly stop at a training-example boundary')
                if model.t % 64 == 0:
                    check_resources()
                started = time.perf_counter()
                t0 = time.perf_counter(); row = next(stream); record['input_seconds'] += time.perf_counter() - t0
                t0 = time.perf_counter(); keys, x = encoder.encode(row['text']); record['encoding_seconds'] += time.perf_counter() - t0
                t0 = time.perf_counter(); model.update(keys, x, row['y']); record['update_seconds'] += time.perf_counter() - t0
                record['input_coordinate_visits'] += len(x)
                size = model.state_bytes()
                if size > config['budget_bytes']:
                    raise MemoryError('Maintained algorithm state exceeded its byte ceiling')
                record['peak_model_state_bytes'] = max(record['peak_model_state_bytes'], size)
                access = stream.state_bytes()
                record['peak_stream_state_bytes'] = max(record['peak_stream_state_bytes'], access)
                record['peak_total_persistent_bytes'] = max(record['peak_total_persistent_bytes'], size+access+pre_bytes)
                record['training_workspace_inventory_lower_bound'] = max(record['training_workspace_inventory_lower_bound'],
                                                                          deep_bytes(row)+keys.nbytes+x.nbytes)
                record['training_seconds'] += time.perf_counter() - started
            record['status'] = 'completed'
            record['failure_reason'] = None
            record['final_primary'] = record['checkpoints'][-1]['metrics']['primary']
            record['final_natural_stream'] = record['checkpoints'][-1]['metrics']['natural_stream']
            if encoder.state_inventory() != record['preprocessing_inventory']:
                raise AssertionError('Preprocessing state changed during stream')
    except Deadline as exc:
        record['status'], record['failure_reason'] = 'paused', str(exc)
    except BaseException as exc:
        record['status'], record['failure_reason'] = 'failed', type(exc).__name__ + ': ' + str(exc).replace(str(ROOT), '<repo>')
    finally:
        if stream:
            stream.close()
        record['worker_timings'] = record.get('worker_timings', []) + timings
        record['peak_rss_bytes'] = max([r['peak_rss_bytes'] for r in record['worker_timings']] or [0])
        record['worker_seconds'] = sum(r['seconds'] for r in record['worker_timings'])
        if any(r['memory_violation'] for r in timings):
            record['status'], record['failure_reason'] = 'failed', 'Process memory/swap safeguard triggered'
        record['model_configuration'] = model.snapshot()[0]
        checkpoint(output, trial, model, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    protocol = json.loads((Path(args.output) / 'run_plan.json').read_text())
    config = next(c for c in protocol['trajectories'] if c['id'] == args.trial)
    result = run(args.output, config, args.seconds)
    print(json.dumps({k: result.get(k) for k in ('id', 'status', 'examples_processed', 'worker_seconds', 'failure_reason')}), flush=True)
    if result['status'] == 'failed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
