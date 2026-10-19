import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state import worker
from experiments.cfpb_streaming_state.data import TASKS, atomic_json, write_spool, sha256
from experiments.cfpb_streaming_state.run import plan


def test_grid_is_exactly_the_approved_96_and_calibration_is_not_extra():
    p = plan()
    assert len(p['trajectories']) == 96
    assert len({t['id'] for t in p['trajectories']}) == 96
    assert {t['budget_bytes'] for t in p['trajectories']} == {16384, 131072, 524288}
    assert p['calibration_trajectory'] == p['trajectories'][0]['id']
    assert p['max_passes'] == 1 and p['benchmark_wall_cap_seconds'] == 3600


def test_orderly_worker_restart_never_reprocesses_training_or_duplicates_checkpoints(tmp_path, monkeypatch):
    (tmp_path/'data_cache').mkdir(); (tmp_path/'trials').mkdir()
    monkeypatch.setitem(TASKS, 'credit_vs_checking', dict(TASKS['credit_vs_checking'], N=16))
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: None)
    info = {}
    for role, n, date in [('train', 16, '2026-01-01'), ('validation', 4, '2026-02-01')]:
        f = tmp_path/'data_cache'/f'{role}.jsonl'
        rows = [dict(id=str(i), date=date, split=role, y=1 if i%2 else -1,
                     text='card credit' if i%2 else 'checking deposit', primary=True,
                     natural_stream=True, stratum='new_in_validation') for i in range(n)]
        write_spool(f, rows)
        info[role] = dict(path=str(f.relative_to(tmp_path)), N=n, bytes=f.stat().st_size,
                          offsets_bytes=f.with_suffix('.offsets.npy').stat().st_size,
                          offsets_sha256=sha256(f.with_suffix('.offsets.npy')), sha256=sha256(f))
    atomic_json(tmp_path/'preparation.json', dict(spools={'credit_vs_checking':info},
        references={'credit_vs_checking':dict(selected={'primary':{'accuracy':1.}})}))
    cfg = dict(id='toy', task='credit_vs_checking', method='mission', representation='B_causal',
               ordering='random', budget_bytes=16384, seed=42)
    paused = worker.run(tmp_path, cfg, 0)
    assert paused['status'] == 'paused' and paused['examples_processed'] == 0
    final = worker.run(tmp_path, cfg, 60)
    assert final['status'] == 'completed', final.get('failure_reason')
    assert final['examples_processed'] == 16 and final['data_passes'] == 1
    assert [c['examples_processed'] for c in final['checkpoints']] == [2,4,8,16]
    assert final['validation_presentations'] == 16
    assert final['peak_model_state_bytes'] <= 16384
    assert final['peak_total_persistent_bytes'] > final['peak_model_state_bytes']
    before = (tmp_path/'trials/toy.json').read_bytes()
    again = worker.run(tmp_path, cfg, 60)
    assert again['examples_processed'] == 16
    assert before == (tmp_path/'trials/toy.json').read_bytes()
