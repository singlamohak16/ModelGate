import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from modelgate.dataset import prepare_dataset
from modelgate.results import CheckResult
from modelgate.training import train_baselines

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_segment_checks.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("demo_segment_checks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_pipelines_demo_repeats_and_requires_trust(tmp_path, mock_download):
    prepare_dataset(tmp_path / "data")
    prepared = tmp_path / "data/prepared"
    models = tmp_path / "models"
    baseline = train_baselines(prepared, models)
    output = tmp_path / "demo.json"
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
    assert untrusted.returncode == 2
    assert "trusted" in untrusted.stderr
    assert not output.exists()
    command.append("--trust-local-models")
    subprocess.run(command, capture_output=True, text=True, check=True)
    original = output.read_bytes()
    report = json.loads(original)
    assert report["report_kind"] == "phase4_segment_checks_demo"
    assert report["data"] == baseline["data"]
    assert "overall_status" not in report
    assert report["policy"]["min_segment_size"] == 50
    for name, record in report["models"].items():
        result = record["validation"]
        assert result["coverage"]["missing_segment_rows"] == 0
        assert result["coverage"]["total_rows"] == 8
        assert result["coverage"]["eligible_rows"] == 0
        assert (
            result["overall"]["recall"]
            == baseline["models"][name]["reference_metrics"]["recall"]
        )
        assert all(CheckResult.model_validate(row) for row in result["checks"])
        assert all(group["metrics"] is None for group in result["segments"])
    assert load_demo().build_demo(prepared, models) == report
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize("defect", ["manifest", "artifact"])
def test_provenance_verified_before_any_deserialization(
    tmp_path, mock_download, monkeypatch, defect
):
    prepare_dataset(tmp_path / "data")
    prepared = tmp_path / "data/prepared"
    models = tmp_path / "models"
    train_baselines(prepared, models)
    if defect == "manifest":
        (models / "baseline_metrics.json").write_text('{"data": {}}')
    else:
        # Second artifact: must reject before loading even the first model.
        artifact = models / "random_forest.joblib"
        artifact.write_bytes(artifact.read_bytes() + b"changed")
    module = load_demo()

    def unexpected_load(*args, **kwargs):
        pytest.fail("Deserialization must not occur before all provenance checks.")

    monkeypatch.setattr(module.joblib, "load", unexpected_load)
    with pytest.raises(ValueError, match="manifest|fingerprint"):
        module.build_demo(prepared, models)
