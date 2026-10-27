"""Controls run: immutable outputs, summary semantics, exactly-once ledger publish."""
import csv
import json

import pandas as pd
import pytest

from experiments.sparsity_screen import controls

SMALL = dict(grid=(500, 1_000, 2_000), replicates=2, audit_n=1_000, n_boot=50)


HEADER = ("experiment_id,dataset,date,git_commit,preprocessing,n_samples,n_features,max_sparsity,"
          "sketch_method,sketch_dimension,seed,model,metric,result,notes\n")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("c")
    ledger = base / "experiment_log.csv"
    ledger.write_text(HEADER + "old-1,d,2026-01-01,,,1,1,1,none,,42,m,x,1,{}\n")
    out, run_id = controls.run_controls(base / "controls", ledger=ledger, **SMALL)
    return out, run_id, ledger


def test_run_writes_all_artifacts(run):
    out, run_id, _ = run
    for name in ("manifest.json", "scaling.csv", "slopes.csv", "controls_summary.csv", "ledger_rows.csv",
                 "audit/bounded.json", "audit/dense_col.json", "audit/bounded_margin.json"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["run_id"] == run_id and manifest["status"] == "completed"
    assert set(manifest["artifacts_sha256"]) >= {"scaling.csv", "controls_summary.csv"}
    assert len(pd.read_csv(out / "scaling.csv")) == 3 * 3 * 2


def test_summary_verdicts_and_labels(run):
    s = pd.read_csv(run[0] / "controls_summary.csv").set_index("case")
    assert s.loc["ctrl_bounded", "sparsity_verdict"] == "PASS"
    assert s.loc["ctrl_dense_col", "sparsity_verdict"] == "FAIL"
    assert s.loc["ctrl_bounded_margin", "sparsity_verdict"] == "PASS"
    assert s.loc["ctrl_bounded_margin", "label"] == "synthetic best case"
    assert s.loc["ctrl_bounded", "label"] == "synthetic control"
    best, base = s.loc["ctrl_bounded_margin"], s.loc["ctrl_bounded"]
    assert best.val_accuracy >= .99 and best.gamma_p10 > 10 * base.gamma_p10
    assert (s.gamma_p10 <= s.gamma_p10_bound + 1e-12).all()
    assert (s.log10_F61_emp_lo < s.log10_F61_proof).all() and (s.log10_F61_emp_hi > s.log10_F61_emp_lo).all()
    # Cold start is reported before kappa / gamma / F61.
    cols = list(s.reset_index().columns)
    assert cols.index("cold_start_mean_at_N_max") < cols.index("kappa_reg") < cols.index("log10_F61_proof")


def test_outputs_are_immutable(run):
    with pytest.raises(FileExistsError):
        controls.run_controls(run[0], **SMALL)


def test_publish_appends_exactly_once(run):
    out, _, ledger = run
    n_rows = len(pd.read_csv(out / "ledger_rows.csv"))
    assert controls.publish(out, ledger) == n_rows
    assert controls.publish(out, ledger) == 0
    with ledger.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["experiment_id"] == "old-1" and len(rows) == 1 + n_rows
    assert all("interpretation" in json.loads(r["notes"]) for r in rows[1:])


def test_publish_rejects_history_changed_since_the_run(tmp_path):
    ledger = tmp_path / "experiment_log.csv"
    ledger.write_text(HEADER + "old-1,d,2026-01-01,,,1,1,1,none,,42,m,x,1,{}\n")
    out, _ = controls.run_controls(tmp_path / "controls", ledger=ledger, **SMALL)
    ledger.write_text(HEADER + "old-1,EDITED,2026-01-01,,,1,1,1,none,,42,m,x,1,{}\n")   # History edited after the run.
    with pytest.raises(ValueError, match="Historical ledger prefix changed"):
        controls.publish(out, ledger)


def test_publish_requires_a_recorded_ledger_prefix(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "x"}))
    pd.DataFrame([{"experiment_id": "a"}]).to_csv(tmp_path / "ledger_rows.csv", index=False)
    with pytest.raises(KeyError):
        controls.publish(tmp_path, tmp_path / "experiment_log.csv")
