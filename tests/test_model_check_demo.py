import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from modelgate.dataset import prepare_dataset
from modelgate.results import CheckResult
from modelgate.training import train_baselines

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_model_checks.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_model_checks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_with_real_pipelines_repeat_and_protections(tmp_path, mock_download):
    prepare_dataset(tmp_path / "data")
    prepared = tmp_path / "data/prepared"
    model_dir = tmp_path / "models"
    baseline = train_baselines(prepared, model_dir)
    output = tmp_path / "demo.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--prepared-dir",
        str(prepared),
        "--model-dir",
        str(model_dir),
        "--output",
        str(output),
    ]
    untrusted = subprocess.run(command, capture_output=True, text=True, check=False)
    assert untrusted.returncode == 2
    assert "trusted" in untrusted.stderr
    assert not output.exists()
    command.append("--trust-local-models")
    subprocess.run(command, capture_output=True, text=True, check=True)
    original = output.read_bytes()
    document = json.loads(original)
    assert document["report_kind"] == "phase3_model_checks_demo"
    assert "overall_status" not in document
    assert document["seed"] == 42
    assert document["data"] == baseline["data"]
    assert document["evaluation_environment"]["packages"]["scikit-learn"]
    for name, record in document["models"].items():
        validation = record["validation"]
        assert len(validation["checks"]) == 6
        assert len(validation["threshold_analysis"]) == 5
        assert len(validation["probability_metrics"]["bins"]) == 10
        assert all(CheckResult.model_validate(row) for row in validation["checks"])
        old = baseline["models"][name]["reference_metrics"]
        for metric in ("recall", "f1", "pr_auc", "average_precision"):
            assert validation["metrics"][metric] == pytest.approx(old[metric])
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert output.read_bytes() == original
    module = load_demo_module()
    assert module.build_demo(prepared, model_dir) == document
    # Hash checks must stop execution before unsafe deserialization.
    artifact = model_dir / "random_forest.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        module.build_demo(prepared, model_dir)


def test_manifest_mismatch_prevents_model_loading(tmp_path, mock_download, monkeypatch):
    prepare_dataset(tmp_path / "data")
    prepared = tmp_path / "data/prepared"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "baseline_metrics.json").write_text('{"data": {}}')
    module = load_demo_module()

    def unexpected_load(*args, **kwargs):
        pytest.fail("Model must not be loaded before provenance validation.")

    monkeypatch.setattr(module.joblib, "load", unexpected_load)
    with pytest.raises(ValueError, match="manifest"):
        module.build_demo(prepared, model_dir)
