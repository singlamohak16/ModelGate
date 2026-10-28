"""Local checks for workflow boundaries and the verification helpers themselves."""

import importlib.util
import json
import re
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_matrix_and_least_privilege():
    # BaseLoader preserves the GitHub 'on' key rather than YAML 1.1 boolean coercion.
    workflow = yaml.load(
        (ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"push", "pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    matrix = workflow["jobs"]["tests"]["strategy"]["matrix"]["include"]
    assert {(x["os"], x["python"]) for x in matrix} == {
        ("ubuntu-24.04", "3.11"),
        ("ubuntu-24.04", "3.12"),
        ("windows-2025", "3.12"),
    }
    assert sum(x["coverage"] == "true" for x in matrix) == 1
    for job in workflow["jobs"].values():
        assert 0 < int(job["timeout-minutes"]) <= 20
        for step in job["steps"]:
            assert "continue-on-error" not in step
            if "uses" in step:
                assert re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", step["uses"])
                if "checkout@" in step["uses"]:
                    assert step["with"]["persist-credentials"] == "false"
                if "upload-artifact@" in step["uses"]:
                    assert step["with"]["retention-days"] == "7"
                    paths = step["with"]["path"].splitlines()
                    assert all(p.startswith("artifacts/ci/") for p in paths)
                    assert all(p.endswith((".json", ".xml")) for p in paths)
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "secrets." not in text
    assert "pull_request_target" not in text
    assert "--require-wheel" in text
    assert "coverage combine" in text
    assert "check_coverage.py" in text


@pytest.mark.parametrize("damage", ["low", "empty", "missing_child", "uncovered_child"])
def test_coverage_gate_rejects_incomplete_evidence(tmp_path, damage):
    data = {
        "totals": {"num_statements": 100, "covered_lines": 95},
        "files": {"src/modelgate/__main__.py": {"summary": {"missing_lines": 0}}},
    }
    if damage == "low":
        data["totals"]["covered_lines"] = 89
    elif damage == "empty":
        data["totals"]["num_statements"] = 0
    elif damage == "missing_child":
        data["files"] = {}
    else:
        data["files"]["src/modelgate/__main__.py"]["summary"]["missing_lines"] = 1
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_script("check_coverage").check_coverage(path)


def test_coverage_gate_accepts_boundary_and_windows_paths(tmp_path):
    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps(
            {
                "totals": {"num_statements": 100, "covered_lines": 90},
                "files": {
                    "src\\modelgate\\__main__.py": {"summary": {"missing_lines": 0}}
                },
            }
        )
    )
    assert load_script("check_coverage").check_coverage(path) == 90


MODULES = ["__init__", "__main__", "cli", "config", "reporting", "experiments"]


@pytest.mark.parametrize(
    "bad",
    [
        "../secret",
        "/absolute",
        "C:/private.csv",
        "fixtures/customers.csv",
        "src\\bad.py",
        "data/raw.csv",
        "models/model.joblib",
        ".env",
        ".env.production",
        "configs/local-private.yaml",
        "artifacts/report.json",
        "reports/generated/report.json",
        "mlflow.db",
        "a/__pycache__/a.pyc",
    ],
)
def test_distribution_checker_rejects_sensitive_or_unsafe_paths(bad):
    names = [f"modelgate/{name}.py" for name in MODULES]
    with pytest.raises(ValueError):
        load_script("check_distributions").validate_names([*names, bad], wheel=True)


def test_distribution_checker_requires_modules():
    with pytest.raises(ValueError, match="Missing"):
        load_script("check_distributions").validate_names(
            ["modelgate/__init__.py"], wheel=True
        )


@pytest.mark.parametrize(
    "entry",
    [
        "modelgate = modelgate.cli:main",
        "modelgate = missing:main",
        "modelgate = modelgate.cli:main_extra",
    ],
)
def test_distribution_archives_and_entrypoint(tmp_path, entry):
    wheel = tmp_path / "modelgate-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in MODULES:
            archive.writestr(f"modelgate/{name}.py", "")
        archive.writestr(
            "modelgate-0.1.0.dist-info/entry_points.txt",
            f"[console_scripts]\n{entry}\n",
        )
    with tarfile.open(tmp_path / "modelgate-0.1.0.tar.gz", "w:gz") as archive:
        for name in MODULES:
            archive.addfile(tarfile.TarInfo(f"modelgate-0.1.0/src/modelgate/{name}.py"))
    checker = load_script("check_distributions").check_distributions
    if entry != "modelgate = modelgate.cli:main":
        with pytest.raises(ValueError, match="entry point"):
            checker(tmp_path)
    else:
        assert checker(tmp_path) == wheel


def test_distribution_checker_rejects_missing_archives(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        load_script("check_distributions").check_distributions(tmp_path)


def test_wheel_guard_rejects_editable_checkout(monkeypatch, tmp_path):
    script = load_script("ci_smoke")
    monkeypatch.setattr(
        script,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda p: tmp_path / p),
    )
    with pytest.raises(ValueError, match="installed wheel"):
        script.check_installed_package()


@pytest.mark.parametrize("layout", ["standalone", "virtualenv"])
@pytest.mark.parametrize("installed", [True, False])
def test_console_location_uses_installation_scheme(
    tmp_path, monkeypatch, layout, installed
):
    script = load_script("ci_smoke")
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python_dir = tmp_path if layout == "standalone" else scripts
    monkeypatch.setattr(script.sys, "executable", str(python_dir / "python.exe"))

    def get_path(name):
        assert name == "scripts"
        return str(scripts)

    monkeypatch.setattr(script.sysconfig, "get_path", get_path)
    name = "modelgate.exe" if script.os.name == "nt" else "modelgate"
    expected = scripts / name
    # A sibling executable in standalone Python's root must not mask a missing
    # command in its configured Scripts directory.
    if layout == "standalone":
        (python_dir / name).touch()
    if installed:
        expected.touch()
        assert script.console_executable() == expected
    else:
        with pytest.raises(FileNotFoundError, match="entry point missing"):
            script.console_executable()


def test_console_smoke_pass_fail_error_and_output_protection(tmp_path):
    script = load_script("ci_smoke")
    output = tmp_path / "smoke"
    summary = script.run_smoke(output)
    assert [c["actual_exit"] for c in summary["cases"]] == [0, 1, 2]
    assert [c["status"] for c in summary["cases"]] == ["PASS", "FAIL", None]
    assert not summary["wheel_verified"]
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        script.run_smoke(output)
    assert (output / "summary.json").read_bytes() == before


def test_smoke_wrong_process_exit_fails(tmp_path, monkeypatch):
    script = load_script("ci_smoke")
    monkeypatch.setattr(
        script.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=2, stderr="injected failure"),
    )
    assert script.main(["--output-dir", str(tmp_path / "bad")]) == 1
