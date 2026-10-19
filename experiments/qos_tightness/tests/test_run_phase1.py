import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def test_calibrate_one_cli_prints_json():
    out = subprocess.run([sys.executable, "-m", "qostight.run_phase1", "calibrate-one",
                          "--N", "8", "--s", "2", "--M", "1000"], cwd=HERE,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["status"] == "ok" and r["N"] == 8 and r["s"] == 2


def test_ledger_row_label_and_columns():
    from qostight.run_phase1 import ledger_row, LEDGER_COLUMNS, INTERPRETATION
    assert INTERPRETATION == "simulation (expected-unitary QOS, reference convention)"
    row = ledger_row("rid", "x", metric="m", result=1.0, N=8, s=2, notes={"a": 1}, commit="abc")
    assert list(row) == LEDGER_COLUMNS
    assert json.loads(row["notes"])["interpretation"] == INTERPRETATION
