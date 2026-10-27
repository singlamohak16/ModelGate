"""Known-defect controls through the real CLI, plus failure and isolation tests."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

from modelgate import experiments as exp
from modelgate.checks.leakage import check_leakage
from modelgate.checks.policy import DataCheckPolicy
from modelgate.config import load_run_config
from modelgate.fingerprint import sha256_file
from modelgate.reporting import AuditReport
from modelgate.results import Status
from modelgate.schema import FEATURE_COLUMNS, ID_COLUMN, SEGMENT_COLUMN, TARGET_COLUMN


@pytest.fixture(scope="module")
def suites(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    first, repeat = root / "first", root / "repeat"
    summaries = [exp.run_experiments(path) for path in (first, repeat)]
    return first, repeat, summaries


def read_report(root, model, scenario):
    return AuditReport.model_validate_json(
        (root / "reports" / f"{model}--{scenario}.json").read_text(encoding="utf-8")
    )


def replay_payload(source_root, config_path):
    """Reuse valid measurements while explicitly updating this test run's paths."""
    config_path = Path(config_path)
    config = load_run_config(config_path)
    payload = json.loads((source_root / "reports" / config.output.name).read_text())
    files = {
        "configuration": config_path,
        "train": config.inputs.train,
        "reference": config.inputs.reference,
        "model": config.artifact.path,
    }
    if config.inputs.current is not None:
        files["current"] = config.inputs.current
    for role, file in files.items():
        payload["fingerprints"][role].update(path=str(file), sha256=sha256_file(file))
    payload["effective_config"] = config.model_dump(mode="json")
    return config, payload


def test_generation_reproducible_balanced_and_disjoint():
    first = exp.generate_fixtures()
    repeat = exp.generate_fixtures()
    changed = exp.generate_fixtures(7)
    for frame, again, other in zip(first, repeat, changed, strict=True):
        pd.testing.assert_frame_equal(frame, again)
        assert not frame.equals(other)
        assert len(frame) == 1000
        assert frame[ID_COLUMN].is_unique
        assert not frame.isna().any().any()
        counts = frame.groupby([SEGMENT_COLUMN, TARGET_COLUMN]).size().to_dict()
        assert counts == {
            (exp.LARGE_SEGMENT, 0): 450,
            (exp.LARGE_SEGMENT, 1): 450,
            (exp.SMALL_SEGMENT, 0): 50,
            (exp.SMALL_SEGMENT, 1): 50,
        }
        assert ((frame.tenure > 30).astype(int) == frame[TARGET_COLUMN]).all()
        assert ((frame.tenure - 30).abs() >= 10).all()
    assert not set(first[0][ID_COLUMN]) & set(first[1][ID_COLUMN])
    assert not set(first[0].tenure) & set(first[1].tenure)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "42", None])
def test_invalid_seed_does_not_create_output(tmp_path, seed):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="seed"):
        exp.run_experiments(destination, seed=seed)
    assert not destination.exists()


def test_mutations_are_exact_and_do_not_modify_sources():
    train, reference = exp.generate_fixtures()
    before_train, before_reference = train.copy(deep=True), reference.copy(deep=True)
    scenarios = exp.build_scenarios(train, reference)
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(reference, before_reference)
    pd.testing.assert_frame_equal(scenarios["clean"].reference, reference)
    pd.testing.assert_frame_equal(scenarios["clean"].current, reference)
    contaminated = scenarios["contamination"].reference
    pd.testing.assert_frame_equal(contaminated.iloc[:20], train.iloc[:20])
    pd.testing.assert_frame_equal(contaminated.iloc[20:], reference.iloc[20:])
    assert contaminated[ID_COLUMN].is_unique
    assert len(set(contaminated[ID_COLUMN]) & set(train[ID_COLUMN])) == 20
    leaked = scenarios["target_leakage"].reference
    pd.testing.assert_frame_equal(leaked.drop(columns=exp.LEAK_COLUMN), reference)
    assert leaked[exp.LEAK_COLUMN].equals(reference[TARGET_COLUMN])
    drift = scenarios["feature_drift"]
    pd.testing.assert_frame_equal(drift.reference, reference)
    pd.testing.assert_frame_equal(
        drift.current.drop(columns=exp.DRIFT_COLUMN),
        reference.drop(columns=[TARGET_COLUMN, exp.DRIFT_COLUMN]),
    )
    np.testing.assert_allclose(drift.current.tenure - reference.tenure, 100)
    assert not drift.current_labeled
    degraded = scenarios["segment_degradation"].reference
    selected = reference[SEGMENT_COLUMN] == exp.SMALL_SEGMENT
    pd.testing.assert_frame_equal(degraded.loc[~selected], reference.loc[~selected])
    pd.testing.assert_frame_equal(
        degraded.drop(columns="tenure"), reference.drop(columns="tenure")
    )
    np.testing.assert_allclose(
        degraded.loc[selected, "tenure"], 60 - reference.loc[selected, "tenure"]
    )
    assert (
        scenarios["segment_degradation"].mutation["affected_positions"]
        == np.flatnonzero(selected).tolist()
    )
    # No aliasing among scenarios or back to source frames.
    scenarios["clean"].current.iloc[0, 0] = "changed locally"
    pd.testing.assert_frame_equal(scenarios["clean"].reference, reference)
    pd.testing.assert_frame_equal(reference, before_reference)


def test_policy_is_fixed_across_scenarios():
    policies = [
        exp.scenario_config(s, "logistic_regression", "a" * 64, 42)
        for s in exp.SCENARIOS
    ]
    comparable = [
        {k: v for k, v in c.items() if k not in ("inputs", "output")} for c in policies
    ]
    assert all(c == comparable[0] for c in comparable)
    assert policies[0]["model"]["min_f1"] == 0.85
    assert policies[0]["segment"]["min_recall"] == 0.8
    assert policies[0]["drift"]["max_numerical_ks"] == 0.2


@pytest.mark.parametrize("model", exp.MODEL_NAMES)
@pytest.mark.parametrize("scenario", exp.SCENARIOS)
def test_full_cli_control(suites, model, scenario):
    first, _, summaries = suites
    row = next(
        r
        for r in summaries[0]["rows"]
        if r["model"] == model and r["scenario"] == scenario
    )
    assert row["matched"], row
    assert row["error"] is None
    assert row["expectations"] and all(e["matched"] for e in row["expectations"])
    report = read_report(first, model, scenario)
    assert report.execution_state == "completed"
    config = load_run_config(first / row["config"])
    assert config.artifact.expected_sha256 == report.fingerprints["model"]["sha256"]
    assert list(config.predictors) == list(FEATURE_COLUMNS)
    if scenario == "segment_degradation":
        assert report.results["model.reference"]["metrics"]["confusion_matrix"] == [
            [450, 50],
            [50, 450],
        ]
        groups = {
            g["segment"]: g for g in report.results["segment.reference"]["segments"]
        }
        assert groups[exp.SMALL_SEGMENT]["record_count"] == 100
        assert groups[exp.SMALL_SEGMENT]["metrics"]["confusion_matrix"] == [
            [0, 50],
            [50, 0],
        ]
        assert (
            groups[exp.SMALL_SEGMENT]["differences"]["recall"]["from_overall"] == -0.9
        )
        assert groups[exp.SMALL_SEGMENT]["differences"]["recall"]["from_best"] == -1
    if scenario == "feature_drift":
        assert report.effective_config["inputs"]["current_labeled"] is False
        assert not any(c.check_id.startswith("performance.") for c in report.checks)


def test_repeat_inputs_and_measurements_match(suites):
    first, repeat, summaries = suites
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (first, repeat)]
    assert manifests[0]["datasets"] == manifests[1]["datasets"]
    assert manifests[0]["mutations"] == manifests[1]["mutations"]
    assert all("\\" not in key for key in manifests[0]["datasets"])
    assert all("\\" not in key for key in manifests[0]["configurations"])
    for summary in summaries:
        assert summary["all_matched"]
        assert summary["runner_exit_code"] == 0
        assert summary["unexpected_input_changes"] == []
        assert len(summary["rows"]) == 10
    for model in exp.MODEL_NAMES:
        hashes = set()
        for scenario in exp.SCENARIOS:
            a, b = [read_report(p, model, scenario) for p in (first, repeat)]
            assert a.checks == b.checks
            assert a.results == b.results
            assert a.coverage == b.coverage
            assert a.run_id != b.run_id
            hashes.add(a.fingerprints["model"]["sha256"])
        assert len(hashes) == 1  # Same fitted artifact across every scenario.
    for path, digest in manifests[0]["datasets"].items():
        assert sha256_file(first / path) == digest


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "wrong_measurement", "blocked", "wrong_exit"]
)
def test_assessor_cannot_pass_false_evidence(suites, damage):
    report = read_report(suites[0], "logistic_regression", "contamination")
    target = next(c for c in report.checks if c.check_id == "overlap.identifiers")
    code = 1
    if damage == "missing":
        report.checks.remove(target)
    elif damage == "duplicate":
        report.checks.append(target)
    elif damage == "wrong_measurement":
        target.measurement["overlapping_reference_rows"] = 19
    elif damage == "blocked":
        report.checks = [
            c
            if c is not target
            else c.model_copy(update={"evaluated": False, "status": Status.WARNING})
            for c in report.checks
        ]
    else:
        code = 2
    assert not all(
        e["matched"] for e in exp.assess_report("contamination", report, code)
    )


def test_prohibited_leakage_rule_separate_from_heuristic():
    train, reference = exp.generate_fixtures()
    frame = exp.build_scenarios(train, reference)["target_leakage"].reference
    policy = DataCheckPolicy(prohibited_columns=(exp.LEAK_COLUMN,))
    checks = check_leakage(frame, policy, "reference")
    prohibited = next(c for c in checks if c.check_id == "leakage.prohibited_columns")
    heuristic = next(
        c for c in checks if c.check_id == f"leakage.target_like.{exp.LEAK_COLUMN}"
    )
    assert prohibited.status == "FAIL"
    assert prohibited.measurement["prohibited_column_count"] == 1
    assert prohibited.evidence["present_columns"] == [exp.LEAK_COLUMN]
    assert heuristic.status == "WARNING"
    assert heuristic.evidence["relationship"] == "exact_copy"


def test_existing_output_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "important.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        exp.run_experiments(output)
    assert marker.read_text() == "keep"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("failure", ["missing_report", "invalid_report", "timeout"])
def test_runner_execution_failure_is_not_expected_success(
    tmp_path, monkeypatch, failure
):
    def fake(command, **kwargs):
        config = load_run_config(command[-1])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 120)
        if failure == "invalid_report":
            config.output.write_text("not json")
        return SimpleNamespace(returncode=2, stdout="", stderr="injected failure")

    monkeypatch.setattr(exp.subprocess, "run", fake)
    summary = exp.run_experiments(tmp_path / "broken")
    assert summary["runner_exit_code"] == 2
    assert not summary["all_matched"]
    assert all(row["error"] and not row["matched"] for row in summary["rows"])


def test_models_fit_once_only_clean_training(tmp_path, monkeypatch):
    seen = []
    original = Pipeline.fit

    def fit(self, features, labels, **kwargs):
        if "classifier" in self.named_steps:
            seen.append((features.copy(), labels.copy()))
        return original(self, features, labels, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=2, stdout="", stderr="skip subprocess"
        ),
    )
    exp.run_experiments(tmp_path / "fit-audit")
    train, _ = exp.generate_fixtures()
    assert len(seen) == 2
    for features, labels in seen:
        pd.testing.assert_frame_equal(features, train[list(FEATURE_COLUMNS)])
        pd.testing.assert_series_equal(labels, train[TARGET_COLUMN])


def test_runner_unexpected_findings_return_one(suites, tmp_path, monkeypatch):
    # Replay real valid reports with adapted provenance, then force a mismatch.
    def replay(command, **kwargs):
        config, payload = replay_payload(suites[0], command[-1])
        config.output.write_text(json.dumps(payload))
        return SimpleNamespace(returncode=payload["exit_code"], stdout="", stderr="")

    monkeypatch.setattr(exp.subprocess, "run", replay)
    monkeypatch.setattr(exp, "assess_report", lambda *a: [{"matched": False}])
    summary = exp.run_experiments(tmp_path / "mismatch")
    assert summary["runner_exit_code"] == 1
    assert not summary["all_matched"]
    assert all(row["error"] is None for row in summary["rows"])


@pytest.mark.parametrize("tamper", ["hash", "path", "roles"])
def test_fingerprint_tamper_detected(suites, tmp_path, monkeypatch, tamper):
    def fake(command, **kwargs):
        config, payload = replay_payload(suites[0], command[-1])
        if tamper == "hash":
            payload["fingerprints"]["model"]["sha256"] = "0" * 64
        elif tamper == "path":
            payload["fingerprints"]["model"]["path"] = str(config.inputs.train)
        else:
            del payload["fingerprints"]["model"]
        config.output.write_text(json.dumps(payload))
        return SimpleNamespace(returncode=payload["exit_code"], stdout="", stderr="")

    monkeypatch.setattr(exp.subprocess, "run", fake)
    summary = exp.run_experiments(tmp_path / "tamper")
    assert summary["runner_exit_code"] == 2
    assert all("fingerprint" in row["error"] for row in summary["rows"])


def test_exit_two_with_valid_audit_cannot_pass(suites, tmp_path, monkeypatch):
    def fake(command, **kwargs):
        config = load_run_config(command[-1])
        config.output.write_bytes(
            (suites[0] / "reports" / config.output.name).read_bytes()
        )
        return SimpleNamespace(
            returncode=2, stdout="", stderr="injected process failure"
        )

    monkeypatch.setattr(exp.subprocess, "run", fake)
    summary = exp.run_experiments(tmp_path / "exit-two")
    assert summary["runner_exit_code"] == 2
    assert not summary["all_matched"]
    assert all("Audit execution failed" in row["error"] for row in summary["rows"])


def test_mutated_inputs_are_recorded_as_execution_error(tmp_path, monkeypatch):
    def fake(command, **kwargs):
        config = load_run_config(command[-1])
        config.inputs.train.write_text("unexpected mutation")
        return SimpleNamespace(returncode=2, stdout="", stderr="injected mutation")

    monkeypatch.setattr(exp.subprocess, "run", fake)
    summary = exp.run_experiments(tmp_path / "mutated")
    assert summary["unexpected_input_changes"] == ["train.csv"]
    assert summary["runner_exit_code"] == 2
    assert not summary["all_matched"]


@pytest.mark.parametrize("code", [0, 1, 2])
def test_script_propagates_runner_code(monkeypatch, tmp_path, code):
    path = Path(__file__).parents[1] / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("experiment_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "run_experiments",
        lambda *a, **k: {"rows": [], "runner_exit_code": code},
    )
    assert module.main(["--output-dir", str(tmp_path / "unused")]) == code


def test_configs_are_relative_and_generated_outputs_ignored(suites):
    root = suites[0]
    for path in (root / "configs").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        assert not Path(raw["output"]).is_absolute()
        assert not Path(raw["artifact"]["path"]).is_absolute()
        assert raw["artifact"]["trust"] is True
    ignore = (Path(__file__).parents[1] / ".gitignore").read_text()
    assert "artifacts/" in ignore.splitlines()
