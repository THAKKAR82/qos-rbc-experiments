import csv
import hashlib
import io
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.cfpb_streaming_state.analyze import append_exactly_once


def test_ledger_append_is_idempotent_and_preserves_original_bytes(tmp_path):
    ledger=tmp_path/'ledger.csv'
    original=b'experiment_id,result,notes\r\nold,0.3,"unchanged, historical"\r\n'
    ledger.write_bytes(original)
    digest=hashlib.sha256(original).hexdigest()
    rows=[dict(experiment_id='new',result=.9,notes='validation only')]
    assert append_exactly_once(ledger,rows,len(original),digest)==1
    after=ledger.read_bytes()
    assert after.startswith(original)
    assert append_exactly_once(ledger,rows,len(original),digest)==0
    assert ledger.read_bytes()==after
    assert len(list(csv.DictReader(io.StringIO(after.decode()))))==2
    with pytest.raises(ValueError,match='Conflicting'):
        append_exactly_once(ledger,[dict(rows[0],result=.1)],len(original),digest)
    ledger.write_bytes(after.replace(b'0.3',b'0.2'))
    with pytest.raises(ValueError,match='prefix'):
        append_exactly_once(ledger,rows,len(original),digest)
