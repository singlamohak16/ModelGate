"""Phase 6 integration contracts; invented CSVs, no external data/downloads."""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelgate.cli import main
from modelgate.config import ConfigError, load_run_config
from modelgate.fingerprint import sha256_file
from modelgate.reporting import AuditReport, aggregate_status, write_report
from modelgate.results import CheckResult
from modelgate.validation import prediction_features, read_audit_csv, run_validation


@pytest.fixture
def run_files(tmp_path):
    train = pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(60)],
            "x": range(60),
            "group": ["A", "B"] * 30,
            "y": [0] * 30 + [1] * 30,
        }
    )
    reference = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(60)],
            "x": range(80, 140),
            "group": ["A", "B"] * 30,
            "y": [0, 0, 1, 1] * 15,
        }
    )
    train.to_csv(tmp_path / "train.csv", index=False)
    reference.to_csv(tmp_path / "reference.csv", index=False)
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["x"]),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["group"]),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    ).fit(train[["x", "group"]], train.y)
    joblib.dump(model, tmp_path / "model.joblib")
    raw = {
        "inputs": {"train": "train.csv", "reference": "reference.csv"},
        "artifact": {"path": "model.joblib", "trust": True},
        "output": "report.json",
        "predictors": ["x", "group"],
        "data": {
            "required_columns": ["id", "x", "group", "y"],
            "column_types": {"x": "number", "group": "category"},
            "target_column": "y",
            "target_labels": [0, 1],
            "positive_class_label": 1,
            "identifier_columns": ["id"],
            "segment_column": "group",
        },
        "segment": {"min_segment_size": 2},
        "drift": {"min_sample_size": 2},
        "performance": {"min_sample_size": 2},
    }
    path = tmp_path / "run.yaml"

    def save():
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return path

    save()
    return path, raw, save


def execute(run_files):
    path, _, save = run_files
    save()
    return run_validation(load_run_config(path), path)


def test_clean_cli_and_report(run_files, capsys):
    path, _, _ = run_files
    assert main(["validate", "--config", str(path)]) == 0
    report = AuditReport.model_validate_json((path.parent / "report.json").read_text())
    assert report.overall_status == "PASS"
    assert report.execution_state == "completed"
    assert report.fingerprints["reference"]["sha256"] == sha256_file(
        path.parent / "reference.csv"
    )
    assert report.fingerprints["configuration"]["sha256"] == sha256_file(path)
    assert report.results["model.reference"]["metrics"]["tp"] == 30
    assert report.results["model.reference"]["metrics"]["fp"] == 30
    assert len(report.results["model.reference"]["threshold_analysis"]) == 5
    assert "PASS" in capsys.readouterr().out
    assert all(
        c.state == "not_applicable"
        for c in report.coverage
        if c.family in ("drift", "performance")
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("unknown", True),
        ("seed", True),
        ("decision_threshold", 1.1),
        ("predictors", ["x", "x"]),
        ("predictors", ["x"]),
        ("model", {"thresholds": [0.5, 0.5]}),
        ("model", {"min_f1": float("nan")}),
        ("model", {"enabled": "yes"}),
        ("model", {"target_labels": ["No", "Yes"]}),
        ("segment", {"min_segment_size": 0}),
        ("drift", {"smoothing": 1.0}),
        ("performance", {"max_f1_drop": -0.1}),
        ("artifact", {"path": "model.joblib", "trust": "true"}),
    ],
)
def test_invalid_run_config(run_files, field, value):
    path, raw, save = run_files
    raw[field] = value
    save()
    with pytest.raises(ConfigError):
        load_run_config(path)
    assert main(["validate", "--config", str(path)]) == 2
    assert not (path.parent / "report.json").exists()


@pytest.mark.parametrize(
    "text",
    [
        "seed: 42\nseed: 43\n",
        "inputs:\n  train: a\n  train: b\n",
        "[a, b]",
        "inputs: [broken",
    ],
)
def test_bad_yaml(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_run_config(path)


@pytest.mark.parametrize(
    "inputs", [{"current_labeled": True}, {"current": "reference.csv"}]
)
def test_impossible_or_implicit_current_mode(run_files, inputs):
    path, raw, save = run_files
    raw["inputs"].update(inputs)
    save()
    with pytest.raises(ConfigError):
        load_run_config(path)


@pytest.mark.parametrize(
    "destination", ["run.yaml", "train.csv", "reference.csv", "model.joblib"]
)
def test_input_output_collision(run_files, destination):
    path, raw, save = run_files
    raw["output"] = destination
    save()
    with pytest.raises(ConfigError, match="collide"):
        load_run_config(path)


def test_paths_independent_of_working_directory(run_files, monkeypatch, tmp_path):
    path, _, _ = run_files
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    config = load_run_config(path)
    assert config.inputs.train == path.parent / "train.csv"
    assert main(["validate", "--config", str(path)]) == 0


@pytest.mark.parametrize("change", ["trust", "hash"])
def test_trust_and_hash_precede_deserialization(run_files, monkeypatch, change):
    _, raw, _ = run_files
    if change == "trust":
        raw["artifact"]["trust"] = False
    else:
        raw["artifact"]["expected_sha256"] = "0" * 64

    def forbidden(*args, **kwargs):
        pytest.fail("Untrusted or mismatched artifact was deserialized")

    monkeypatch.setattr(joblib, "load", forbidden)
    report = execute(run_files)
    assert report.exit_code == 2
    assert report.overall_status is None
    assert report.checks


def test_valid_pinned_model_hash(run_files):
    path, raw, _ = run_files
    raw["artifact"]["expected_sha256"] = sha256_file(path.parent / "model.joblib")
    assert execute(run_files).exit_code == 0


def test_feature_copy_preserves_alignment_and_blanks(run_files):
    path, _, _ = run_files
    config = load_run_config(path)
    frame = pd.DataFrame(
        {"id": ["b", "a"], "group": ["NA", " "], "x": ["2", " "], "y": ["1", "0"]},
        index=[8, 3],
    )
    before = frame.copy(deep=True)
    features, reason = prediction_features(frame, config)
    assert reason is None
    assert list(features.columns) == ["x", "group"]
    assert list(features.index) == [8, 3]
    assert features.iloc[0].tolist() == [2.0, "NA"]
    assert features.iloc[1].isna().all()
    pd.testing.assert_frame_equal(frame, before)


def test_no_fit_and_one_prediction_per_dataset(run_files, monkeypatch):
    _, raw, _ = run_files
    raw["inputs"].update(current="reference.csv", current_labeled=True)
    original = Pipeline.predict_proba
    calls = []

    def spy(self, frame):
        calls.append(frame.copy())
        return original(self, frame)

    def forbidden(*args, **kwargs):
        pytest.fail("Validation must not fit")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "predict_proba", spy)
    report = execute(run_files)
    assert report.exit_code == 0
    assert len(calls) == 2
    assert list(calls[0].x) == list(range(80, 140))
    assert all(
        c.measurement["value"] == 0
        for c in report.checks
        if c.check_id.startswith("drift.")
    )
    assert report.results["performance.reference_vs_current"]["comparison_available"]


def test_unlabeled_current_no_performance_claims(run_files):
    path, raw, _ = run_files
    frame = pd.read_csv(path.parent / "reference.csv")
    frame["y"] = "NOT A LABEL"
    frame.to_csv(path.parent / "current.csv", index=False)
    raw["inputs"].update(current="current.csv", current_labeled=False)
    report = execute(run_files)
    assert report.exit_code == 0
    assert not any(c.check_id.startswith("performance.") for c in report.checks)
    assert "model.current" not in report.results
    assert "segment.current" not in report.results
    assert any(
        c.check_id == "drift.prediction_ks" and c.evaluated for c in report.checks
    )


@pytest.mark.parametrize("column", ["x", "y", "id"])
def test_missing_column_fails_rule_and_preserves_independent_checks(run_files, column):
    path, _, _ = run_files
    frame = pd.read_csv(path.parent / "reference.csv").drop(columns=column)
    frame.to_csv(path.parent / "reference.csv", index=False)
    report = execute(run_files)
    assert report.exit_code == 1
    assert report.execution_state == "completed"
    assert any(
        c.check_id == "schema.columns" and c.status == "FAIL" for c in report.checks
    )
    if column == "id":
        assert "model.reference" in report.results
    else:
        assert any(
            c.family == "model" and c.state == "blocked" for c in report.coverage
        )
    assert any(c.check_id == "duplicates.rows" and c.evaluated for c in report.checks)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_id",
        "quality_limit",
        "invalid_numeric",
        "invalid_label",
        "duplicate_header",
        "header_only",
    ],
)
def test_readable_defects_are_exit_one(run_files, defect):
    path, raw, save = run_files
    csv_path = path.parent / "reference.csv"
    frame = pd.read_csv(csv_path, dtype=str)
    if defect == "duplicate_id":
        frame.loc[1, "id"] = frame.loc[0, "id"]
    elif defect == "quality_limit":
        raw["model"] = {"min_f1": 0.99}
    elif defect == "invalid_numeric":
        frame.loc[0, "x"] = "broken"
    elif defect == "invalid_label":
        frame.loc[0, "y"] = "broken"
    elif defect == "duplicate_header":
        frame.columns = ["id", "x", "x", "y"]
    elif defect == "header_only":
        frame = frame.iloc[:0]
    frame.to_csv(csv_path, index=False)
    save()
    assert main(["validate", "--config", str(path)]) == 1
    report = AuditReport.model_validate_json((path.parent / "report.json").read_text())
    assert report.overall_status == "FAIL"


def test_warning_is_visible_exit_zero(run_files, capsys):
    path, raw, save = run_files
    raw["decision_threshold"] = 1.0
    save()
    assert main(["validate", "--config", str(path)]) == 0
    report = AuditReport.model_validate_json((path.parent / "report.json").read_text())
    assert report.overall_status == "WARNING"
    assert report.results["model.reference"]["metrics"]["precision"] is None
    assert report.warnings
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize(
    "defect",
    [
        "missing_file",
        "ragged_csv",
        "unclosed_quote",
        "corrupt_model",
        "prediction_failure",
        "invalid_probabilities",
    ],
)
def test_execution_errors_write_partial_report(run_files, monkeypatch, defect):
    path, raw, save = run_files
    if defect == "missing_file":
        raw["inputs"]["reference"] = "absent.csv"
    elif defect == "ragged_csv":
        (path.parent / "reference.csv").write_text("id,x\na,1,extra\n")
    elif defect == "unclosed_quote":
        (path.parent / "reference.csv").write_text('id,x\na,"unclosed\n')
    elif defect == "corrupt_model":
        (path.parent / "model.joblib").write_bytes(b"not a model")
    elif defect == "prediction_failure":

        def broken(*args, **kwargs):
            raise RuntimeError("synthetic inference error")

        monkeypatch.setattr(Pipeline, "predict_proba", broken)
    else:
        monkeypatch.setattr(
            Pipeline,
            "predict_proba",
            lambda self, frame: np.full((len(frame), 2), np.nan),
        )
    save()
    assert main(["validate", "--config", str(path)]) == 2
    report = AuditReport.model_validate_json((path.parent / "report.json").read_text())
    assert report.execution_state == "error"
    assert report.overall_status is None
    assert report.errors


def test_disabled_families_have_coverage(run_files):
    _, raw, _ = run_files
    for family in ("model", "segment", "drift", "performance"):
        raw[family] = {"enabled": False}
    report = execute(run_files)
    assert report.exit_code == 0
    assert all(c.state == "disabled" for c in report.coverage if c.family != "data")


def test_existing_output_unchanged(run_files):
    path, _, _ = run_files
    output = path.parent / "report.json"
    output.write_text("keep me")
    assert main(["validate", "--config", str(path)]) == 2
    assert output.read_text() == "keep me"


def test_write_failure_exit_two(run_files, monkeypatch):
    path, _, _ = run_files

    def fail(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr("modelgate.cli.write_report", fail)
    assert main(["validate", "--config", str(path)]) == 2


def test_report_roundtrip_and_overwrite_guard(run_files, tmp_path):
    report = execute(run_files)
    output = tmp_path / "nested" / "audit.json"
    write_report(report, output)
    assert AuditReport.model_validate_json(output.read_text()) == report
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_report(report, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "change",
    [
        {"exit_code": 1},
        {"overall_status": "FAIL"},
        {"execution_state": "error"},
        {"errors": ["oops"]},
        {"checks": []},
        {"results": {"bad": float("inf")}},
    ],
)
def test_report_rejects_inconsistency(run_files, change):
    payload = execute(run_files).model_dump(mode="python")
    payload.update(change)
    with pytest.raises(ValidationError):
        AuditReport.model_validate(payload)


def test_serialization_before_creating_destination(run_files, tmp_path):
    report = execute(run_files)
    report.results["bad"] = float("nan")
    destination = tmp_path / "not-created" / "report.json"
    with pytest.raises(ValueError):
        write_report(report, destination)
    assert not destination.parent.exists()


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["PASS"], "PASS"),
        (["PASS", "WARNING"], "WARNING"),
        (["FAIL", "WARNING"], "FAIL"),
    ],
)
def test_severity_order(statuses, expected):
    checks = [
        CheckResult(
            check_id="test",
            dataset="reference",
            status=s,
            evaluated=True,
            measurement={},
            threshold={},
            evidence={},
            message="fixture",
        )
        for s in statuses
    ]
    assert aggregate_status(checks) == expected


def test_literal_csv_and_fingerprint(tmp_path):
    path = tmp_path / "literal.csv"
    path.write_bytes(b'id,text\n001,NA\n002,"line1\nline2"\n')
    frame, digest = read_audit_csv(path)
    assert frame.iloc[0].tolist() == ["001", "NA"]
    assert frame.iloc[1, 1] == "line1\nline2"
    assert digest == sha256_file(path)


def test_module_entrypoint(run_files):
    path, _, _ = run_files
    completed = subprocess.run(
        [sys.executable, "-m", "modelgate", "validate", "--config", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=path.parent,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PASS" in completed.stdout


def test_console_script_declared():
    import tomllib

    project = Path(__file__).parents[1] / "pyproject.toml"
    assert (
        tomllib.loads(project.read_text())["project"]["scripts"]["modelgate"]
        == "modelgate.cli:main"
    )


def test_config_hash_tracks_parsed_bytes_not_later_edit(run_files):
    path, _, _ = run_files
    config = load_run_config(path)
    digest = sha256_file(path)
    path.write_text("changed: true\n")
    report = run_validation(config, path)
    assert report.fingerprints["configuration"]["sha256"] == digest
    assert digest != sha256_file(path)


def test_shared_roles_and_thresholds(run_files):
    path, raw, save = run_files
    raw["decision_threshold"] = 0.7
    raw["data"]["positive_class_label"] = 0
    save()
    policies = load_run_config(path).policies()
    for family in ("model", "segment", "performance"):
        assert policies[family].decision_threshold == 0.7
        assert policies[family].positive_class_label == 0
    assert policies["drift"].numerical_columns == ("x",)
    assert policies["drift"].categorical_columns == ("group",)


@pytest.mark.parametrize(
    "update",
    [
        {"target_column": "id"},
        {"identifier_columns": ["x"]},
        {"target_labels": [0, 0]},
        {"positive_class_label": 2},
        {"segment_column": "x"},
        {"missing_overrides": {"absent": 0.1}},
    ],
)
def test_data_role_conflicts_rejected(run_files, update):
    path, raw, save = run_files
    raw["data"].update(update)
    save()
    with pytest.raises(ConfigError):
        load_run_config(path)


def test_partial_drift_survives_blocked_predictions(run_files):
    path, raw, _ = run_files
    frame = pd.read_csv(path.parent / "reference.csv").drop(columns="x")
    frame.to_csv(path.parent / "current.csv", index=False)
    raw["inputs"].update(current="current.csv", current_labeled=True)
    report = execute(run_files)
    assert report.exit_code == 1
    drift = {c.check_id: c for c in report.checks if c.check_id.startswith("drift.")}
    assert drift["drift.categorical_distance"].measurement["value"] == 0
    assert not drift["drift.numerical_ks"].evaluated
    assert not drift["drift.prediction_ks"].evaluated
    coverage = next(c for c in report.coverage if c.family == "drift")
    assert coverage.evaluated_checks == 1
    assert coverage.blocked_checks == 2
    changes = [
        c
        for c in report.checks
        if c.check_id.startswith("performance.")
        and c.check_id != "performance.class_support"
    ]
    assert len(changes) == 5
    assert all(not c.evaluated for c in changes)


def test_missing_current_labels_fail_without_silently_changing_mode(run_files):
    path, raw, _ = run_files
    frame = pd.read_csv(path.parent / "reference.csv").drop(columns="y")
    frame.to_csv(path.parent / "current.csv", index=False)
    raw["inputs"].update(current="current.csv", current_labeled=True)
    report = execute(run_files)
    assert report.exit_code == 1
    assert any(
        c.check_id == "drift.prediction_ks" and c.evaluated for c in report.checks
    )
    assert not report.results["performance.reference_vs_current"][
        "comparison_available"
    ]


def test_unlabeled_current_not_predicted_when_no_comparison_enabled(
    run_files, monkeypatch
):
    _, raw, _ = run_files
    raw["inputs"].update(current="reference.csv", current_labeled=False)
    raw["drift"] = {"enabled": False}
    original = Pipeline.predict_proba
    calls = []

    def spy(self, frame):
        calls.append(frame)
        return original(self, frame)

    monkeypatch.setattr(Pipeline, "predict_proba", spy)
    assert execute(run_files).exit_code == 0
    assert len(calls) == 1


def test_fitted_predictor_order_mismatch_is_error(run_files):
    _, raw, _ = run_files
    raw["predictors"] = ["group", "x"]
    report = execute(run_files)
    assert report.exit_code == 2
    assert "predictor order" in report.errors[0]


def test_independent_model_checks_with_invalid_train(run_files):
    path, _, _ = run_files
    frame = pd.read_csv(path.parent / "train.csv").drop(columns="x")
    frame.to_csv(path.parent / "train.csv", index=False)
    report = execute(run_files)
    assert report.exit_code == 1
    assert "model.reference" in report.results


def test_report_os_write_failure(run_files, monkeypatch):
    path, _, _ = run_files
    output = path.parent / "report.json"
    original = Path.open

    def fail(self, *args, **kwargs):
        if self == output:
            raise PermissionError("synthetic output permission failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    assert main(["validate", "--config", str(path)]) == 2
    assert not output.exists()


def test_existing_output_race_still_refuses_overwrite(run_files, monkeypatch):
    path, _, _ = run_files
    output = path.parent / "report.json"
    original = run_validation

    def race(*args):
        report = original(*args)
        output.write_text("another writer's report")
        return report

    monkeypatch.setattr("modelgate.cli.run_validation", race)
    assert main(["validate", "--config", str(path)]) == 2
    assert output.read_text() == "another writer's report"


@pytest.mark.parametrize(
    "arguments,code", [([], 2), (["validate"], 2), (["--help"], 0)]
)
def test_cli_usage(arguments, code):
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == code
