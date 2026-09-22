import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from modelgate.dataset import prepare_dataset
from modelgate.results import CheckResult
from modelgate.training import train_baselines

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_drift_checks.py"


def module():
    spec = importlib.util.spec_from_file_location("demo_drift_checks", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_demo_provenance_unlabeled_no_claims_and_repeat(tmp_path, mock_download):
    prepare_dataset(tmp_path / "data")
    prepared, models = tmp_path / "data/prepared", tmp_path / "models"
    training = train_baselines(prepared, models)
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--prepared-dir",
        str(prepared),
        "--model-dir",
        str(models),
        "--output",
        str(output),
    ]
    untrusted = subprocess.run(command, capture_output=True, text=True, check=False)
    assert untrusted.returncode == 2 and "trusted" in untrusted.stderr
    assert not output.exists()
    command.append("--trust-local-models")
    subprocess.run(command, capture_output=True, text=True, check=True)
    original = output.read_bytes()
    document = json.loads(original)
    assert document["data"] == training["data"]
    assert document["report_kind"] == "phase5_self_comparison_controls"
    assert document["reference_current_overlap_fraction"] == 1
    assert "NOT new" in document["current_source"]
    assert "overall_status" not in document
    for record in document["models"].values():
        labeled, unlabeled = (
            record["controls"]["labeled"],
            record["controls"]["unlabeled"],
        )
        assert unlabeled["performance"]["checks"] == []
        assert unlabeled["performance"]["current"] is None
        assert labeled["drift_checks"] == unlabeled["drift_checks"]
        assert len(labeled["drift_checks"]) == 20
        # The invented reference has 8 rows: demonstrate honest support warnings.
        assert all(not c["evaluated"] for c in labeled["drift_checks"])
        for c in labeled["drift_checks"] + labeled["performance"]["checks"]:
            CheckResult.model_validate(c)
    assert module().build_demo(prepared, models) == document
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode == 2 and "already exists" in repeated.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize("defect", ["manifest", "artifact"])
def test_all_provenance_checked_before_loading(
    tmp_path, mock_download, monkeypatch, defect
):
    prepare_dataset(tmp_path / "data")
    prepared, models = tmp_path / "data/prepared", tmp_path / "models"
    train_baselines(prepared, models)
    if defect == "manifest":
        (models / "baseline_metrics.json").write_text('{"data": {}}')
    else:
        path = models / "random_forest.joblib"
        path.write_bytes(path.read_bytes() + b"changed")
    demo = module()

    def never_load(*args, **kwargs):
        pytest.fail("Deserialization must wait for all provenance checks.")

    monkeypatch.setattr(demo.joblib, "load", never_load)
    with pytest.raises(ValueError, match="manifest|fingerprint"):
        demo.build_demo(prepared, models)
