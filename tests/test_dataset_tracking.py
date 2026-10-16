"""Metadata-only checks: no local datasets or scientific computation required."""
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dataset_checksum', ROOT / 'scripts/dataset_checksum.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_checksum_known_digest_and_cli(tmp_path):
    path = tmp_path / 'sample with spaces.txt'
    path.write_bytes(b'abc')
    expected = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert helper.checksum(path) == dict(path=str(path), size_bytes=3, sha256=expected)
    empty = tmp_path / 'empty'
    empty.touch()
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/dataset_checksum.py'), str(path), str(empty)], check=True, text=True, capture_output=True)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    assert len(rows) == 2
    assert rows[0]['sha256'] == expected
    assert rows[1]['sha256'] == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def test_manifest_schema_ids_and_parents():
    with (ROOT / 'data/manifest.csv').open(newline='') as source:
        reader = csv.DictReader(source)
        required = set('dataset_id dataset_name object_type parent_dataset_id source source_url version_or_date license_or_terms redistribution_allowed raw_filename local_path sha256 size_bytes n_rows n_columns encoded_dimension_D nnz mean_row_nnz max_row_nnz mean_col_nnz max_col_nnz target temporal_field split_definition used_in status notes'.split())
        assert required <= set(reader.fieldnames)
        rows = list(reader)
    ids = [r['dataset_id'] for r in rows]
    assert rows and all(ids) and len(ids) == len(set(ids))
    assert all(not r['parent_dataset_id'] or r['parent_dataset_id'] in ids for r in rows)
    assert all(r['parent_dataset_id'] != r['dataset_id'] for r in rows)
