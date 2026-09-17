import json
import subprocess
import sys
from pathlib import Path

from modelgate.dataset import prepare_dataset
from modelgate.fingerprint import sha256_file
from modelgate.results import CheckResult


def test_demo_writes_results_and_protects_existing_output(tmp_path, mock_download):
    prepare_dataset(tmp_path / "data")
    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_data_checks.py"
    output = tmp_path / "demo.json"
    command = [
        sys.executable,
        str(script),
        "--prepared-dir",
        str(tmp_path / "data/prepared"),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "Individual check counts" in completed.stdout
    original = output.read_bytes()
    document = json.loads(original)
    assert document["report_kind"] == "phase2_data_checks_demo"
    assert document["source_sha256"]["train"] == sha256_file(
        tmp_path / "data/prepared/train.csv"
    )
    assert len(document["results"]) == 136
    assert all(CheckResult.model_validate(result) for result in document["results"])
    assert "overall_status" not in document
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr
    assert output.read_bytes() == original
