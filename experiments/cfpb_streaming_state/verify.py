"""Post-run integrity checks; no model updates, predictions, or test scoring."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np
from .data import ROOT, TASKS, atomic_json, sha256
from .methods import Learner
from .prepare import SOURCE, SOURCE_SHA256, verify_frozen
from .state import totals


def verify(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('This final completeness check requires a completed pilot')
    prep=json.loads((output/'preparation.json').read_text())
    for group in ('protected_sha256','source_sha256'):
        for path,digest in manifest[group].items():
            assert sha256(ROOT/path)==digest,(group,path)
    assert sha256(SOURCE)==SOURCE_SHA256
    ledger=(ROOT/'results/experiment_log.csv').read_bytes()
    assert hashlib.sha256(ledger[:manifest['prior_ledger_bytes']]).hexdigest()==manifest['prior_ledger_sha256']
    checked=[]
    for trial,status in manifest['trajectory_status'].items():
        assert status=='completed'
        record=json.loads((output/'trials'/f'{trial}.json').read_text())
        state=output/record['state_path']
        assert sha256(state)==record['state_sha256']
        with np.load(state,allow_pickle=False) as saved:
            for name in saved.files:
                if name!='metadata':
                    assert np.isfinite(saved[name]).all(),(trial,name)
            meta=json.loads(str(saved['metadata']))
            model=Learner.restore(meta,saved)
        n=TASKS[record['task']]['N']
        assert model.t==record['examples_processed']==n
        assert record['data_passes']==1 and record['attempts']==1
        assert [p['examples_processed'] for p in record['checkpoints']]==[math.ceil(n*f) for f in (.125,.25,.5,1.)]
        assert record['validation_presentations']==4*prep['spools'][record['task']]['validation']['N']
        assert model.state_bytes()<=record['budget_bytes']
        assert record['peak_model_state_bytes']<=record['budget_bytes']
        assert not any(t['memory_violation'] for t in record['worker_timings'])
        # Python key-sharing instance dictionaries have process-history-dependent
        # allocation sizes. The worker's measured inventory is authoritative;
        # cross-process equality with shared preparation is not a valid invariant.
        assert record['preprocessing_state_bytes']==totals(record['preprocessing_inventory'])['bytes']
        checked.append(trial)
    pins={}
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            package,version=line.split('==')
            observed=importlib.metadata.version(package)
            assert observed==version,(package,observed,version)
            pins[package]=observed
    fixture_spec=ast.parse((ROOT/'tests/test_regression.py').read_text())
    fixture_hashes=next(ast.literal_eval(n.value) for n in fixture_spec.body
                       if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ARTIFACT_HASHES' for t in n.targets))
    for path,digest in fixture_hashes.items():
        assert sha256(ROOT/path)==digest,('Bank regression fixture',path)
    result=dict(status='passed',frozen_cfpb_artifacts=verify_frozen(),
        protected_tracked_files=len(manifest['protected_sha256']),
        execution_source_fingerprints=len(manifest['source_sha256']),
        completed_models_finite_hash_valid_and_restorable=len(checked),
        examples_processed=sum(TASKS[json.loads((output/'trials'/f'{t}.json').read_text())['task']]['N'] for t in checked),
        all_original_ledger_prefix_bytes_preserved=True,raw_source_hash_preserved=True,
        preprocessing_accounting_note='All A workers measured 8219927 bytes versus 8219895 in preparation. A separate initialization-only check reproduced 296/288/280-byte vectorizer and transformer dictionaries for successive instances (32-byte first/third difference). Worker totals already include the larger actual allocation; no scientific outputs were changed. The initial post-run cross-process-equality assertion was replaced by exact equality to each worker inventory.',
        bank_regression_artifacts_verified=len(fixture_hashes),
        all_pins_match=pins,verification_source_sha256=sha256(Path(__file__)),
        test_evidence=json.loads((output/'test_results.json').read_text()))
    atomic_json(output/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output')
    verify(parser.parse_args().output)
