"""Offline orchestration: raw audit first, fitted inference second, no fitting."""

import csv
import io
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from modelgate import __version__
from modelgate.checks._common import missing_mask, numerical_values, structure_problem
from modelgate.checks.drift import run_drift_checks
from modelgate.checks.model import run_model_checks
from modelgate.checks.performance import run_performance_checks
from modelgate.checks.runner import run_data_checks
from modelgate.checks.segment import run_segment_checks
from modelgate.config import RunConfig
from modelgate.fingerprint import sha256_bytes, sha256_file
from modelgate.predictions import positive_probabilities
from modelgate.reporting import AuditReport, aggregate_status
from modelgate.results import CheckResult, Status


def read_audit_csv(path: Path):
    """Hash and parse the same bytes; preserve literal cells and bad schema.

    Unlike pandas inference, this rejects ragged records rather than shifting
    columns or silently treating an extra field as an index. Duplicate/blank
    headers remain schema evidence rather than being renamed by the parser.
    """
    payload = path.read_bytes()
    rows = list(
        csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True)
    )
    if not rows:
        raise ValueError(f"CSV has no header: {path}")
    header, *records = rows
    if any(len(row) != len(header) for row in records):
        raise ValueError(f"CSV contains ragged records: {path}")
    return pd.DataFrame(records, columns=header, dtype=object), sha256_bytes(payload)


def prediction_features(frame, config):
    """Return an ordered copy or a dependency reason; never drop/reorder rows."""
    problem = structure_problem(frame)
    absent = set(config.predictors) - set(frame.columns)
    if problem or absent:
        return None, problem or f"Missing predictors: {sorted(absent)}"
    features = frame.loc[:, list(config.predictors)].copy(deep=True)
    for column in config.predictors:
        kind = config.data.column_types[column]
        if kind == "category":
            allowed = config.data.allowed_categories.get(column)
            if (
                allowed
                and (
                    ~missing_mask(features[column]) & ~features[column].isin(allowed)
                ).any()
            ):
                return None, f"Invalid configured category: {column}"
            features[column] = features[column].where(
                ~missing_mask(features[column]), np.nan
            )
        else:
            values, invalid = numerical_values(features[column], kind)
            if invalid.any():
                return None, f"Invalid numerical predictor: {column}"
            features[column] = values.astype(float)
    return features, None


def label_problem(frame, config):
    problem = structure_problem(frame)
    target = config.data.target_column
    if problem or target not in frame:
        return problem or f"Missing target: {target}"
    if not frame[target].isin([str(v) for v in config.data.target_labels]).all():
        return "Missing or invalid declared target labels."
    return None


def _parameter_value(value):
    """Retain JSON-native parameters; label complex sklearn objects as repr."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return {"representation": repr(value), "type": type(value).__name__}


def run_validation(config: RunConfig, config_path: str | Path) -> AuditReport:
    started = datetime.now(UTC).isoformat()
    checks, coverage, errors = [], [], []
    results, fingerprints, identity = {}, {}, {}
    packages = {"python": platform.python_version()}
    policies = config.policies()

    def cover(family, dataset, state, reason=None, family_checks=()):
        coverage.append(
            dict(
                family=family,
                dataset=dataset,
                state=state,
                reason=reason,
                evaluated_checks=sum(c.evaluated for c in family_checks),
                blocked_checks=sum(not c.evaluated for c in family_checks),
            )
        )

    def blocked(family, dataset, reason):
        checks.append(
            CheckResult(
                check_id=f"{family}.prerequisites",
                dataset=dataset,
                status=Status.WARNING,
                evaluated=False,
                measurement={"value": None},
                threshold={},
                evidence={"reason": reason},
                message=reason,
            )
        )
        cover(family, dataset, "blocked", reason, checks[-1:])

    def collect(family, dataset, report):
        family_checks = [CheckResult.model_validate(c) for c in report["checks"]]
        checks.extend(family_checks)
        results[f"{family}.{dataset}"] = {
            k: v for k, v in report.items() if k != "checks"
        }
        has_evaluated = any(c.evaluated for c in family_checks)
        cover(
            family,
            dataset,
            "evaluated" if has_evaluated else "blocked",
            "Some checks could not be evaluated; see check evidence."
            if any(not c.evaluated for c in family_checks)
            else None,
            family_checks,
        )

    stage = "provenance"
    try:
        for package in (
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "joblib",
            "pydantic",
            "PyYAML",
        ):
            packages[package] = version(package)
        fingerprints["configuration"] = {
            "path": str(config._source_path or Path(config_path).resolve()),
            "sha256": config._source_sha256 or sha256_file(config_path),
        }
        frames = {}
        stage = "CSV loading"
        for name in ("train", "reference", "current"):
            path = getattr(config.inputs, name)
            if path is not None:
                frame, digest = read_audit_csv(path)
                frames[name] = frame
                fingerprints[name] = {
                    "path": str(path),
                    "sha256": digest,
                    "rows": len(frame),
                    "columns": len(frame.columns),
                }
        stage = "data checks"
        checks.extend(
            run_data_checks(
                frames["train"],
                frames["reference"],
                config.data,
                current=frames.get("current"),
                current_labeled=config.inputs.current_labeled,
            )
        )
        for name in (*frames, "train/reference"):
            cover(
                "data",
                name,
                "evaluated",
                family_checks=[c for c in checks if c.dataset == name],
            )
        stage = "model loading"
        if not config.artifact.trust:
            raise ValueError(
                "Set artifact.trust: true only for a trusted local joblib artifact; "
                "loading can execute code."
            )
        payload = config.artifact.path.read_bytes()
        digest = sha256_bytes(payload)
        fingerprints["model"] = {"path": str(config.artifact.path), "sha256": digest}
        if (
            config.artifact.expected_sha256
            and digest != config.artifact.expected_sha256
        ):
            raise ValueError("Model SHA256 does not match expected_sha256.")
        model = joblib.load(io.BytesIO(payload))
        identity = {
            "type": f"{type(model).__module__}.{type(model).__qualname__}",
            "classes": _parameter_value(
                np.asarray(getattr(model, "classes_", [])).tolist()
            ),
            "parameters": {
                key: _parameter_value(value)
                for key, value in model.get_params(deep=True).items()
            },
        }
        probabilities, reasons = {}, {}
        for name in ("reference", "current"):
            if name not in frames:
                continue
            needs_inference = (
                (name == "reference" or config.inputs.current_labeled)
                and (config.model.enabled or config.segment.enabled)
            ) or (
                "current" in frames
                and (
                    config.drift.enabled
                    or (config.performance.enabled and config.inputs.current_labeled)
                )
            )
            if not needs_inference:
                cover(
                    "predictions",
                    name,
                    "disabled",
                    "No enabled check needs predictions.",
                )
                continue
            features, reason = prediction_features(frames[name], config)
            if reason:
                reasons[name] = reason
                blocked("predictions", name, reason)
                continue
            stage = f"{name} model inference"
            if hasattr(model, "feature_names_in_") and list(
                model.feature_names_in_
            ) != list(config.predictors):
                raise ValueError(
                    "Configured predictor order differs from model.feature_names_in_."
                )
            probabilities[name] = positive_probabilities(
                model,
                features,
                target_labels=config.data.target_labels,
                positive_label=config.data.positive_class_label,
            )
            cover("predictions", name, "evaluated")
        for name in ("reference", "current"):
            for family in ("model", "segment"):
                if not getattr(config, family).enabled:
                    cover(family, name, "disabled")
                    continue
                if name not in frames or (
                    name == "current" and not config.inputs.current_labeled
                ):
                    cover(
                        family,
                        name,
                        "not_applicable",
                        "Dataset absent or labels not declared available.",
                    )
                    continue
                frame = frames[name]
                reason = reasons.get(name) or label_problem(frame, config)
                if family == "segment" and config.data.segment_column not in frame:
                    reason = reason or "Missing segment column."
                if reason:
                    blocked(family, name, reason)
                    continue
                stage = f"{name} {family} checks"
                if family == "model":
                    report = run_model_checks(
                        frame[config.data.target_column],
                        probabilities[name],
                        policies[family],
                        dataset=name,
                    )
                else:
                    report = run_segment_checks(
                        frame, probabilities[name], policies[family], dataset=name
                    )
                collect(family, name, report)
        for family in ("drift", "performance"):
            dataset = "reference_vs_current"
            if not getattr(config, family).enabled:
                cover(family, dataset, "disabled")
            elif "current" not in frames or (
                family == "performance" and not config.inputs.current_labeled
            ):
                cover(
                    family,
                    dataset,
                    "not_applicable",
                    "Dataset absent or labels not declared available.",
                )
            else:
                stage = f"{family} checks"
                args = (frames["reference"], frames["current"])
                if family == "drift":
                    drift = run_drift_checks(
                        *args,
                        policies[family],
                        reference_probabilities=probabilities.get("reference"),
                        current_probabilities=probabilities.get("current"),
                    )
                    collect(
                        family,
                        dataset,
                        {"checks": [c.model_dump(mode="json") for c in drift]},
                    )
                else:
                    collect(
                        family,
                        dataset,
                        run_performance_checks(
                            *args,
                            probabilities.get("reference"),
                            probabilities.get("current"),
                            policies[family],
                            current_labeled=True,
                        ),
                    )
    except Exception as error:
        # Unexpected execution errors must never become ordinary rule results.
        errors.append(f"{stage}: {type(error).__name__}: {error}")
        cover("execution", stage, "blocked", errors[-1])
    status = None if errors else aggregate_status(checks)
    warnings = [
        {
            "check_index": i,
            "check_id": c.check_id,
            "dataset": c.dataset,
            "message": c.message,
        }
        for i, c in enumerate(checks)
        if c.status == Status.WARNING
    ]
    return AuditReport(
        run_id=str(uuid4()),
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        execution_state="error" if errors else "completed",
        seed=config.seed,
        packages=packages,
        fingerprints=fingerprints,
        effective_config=config.model_dump(mode="json"),
        model=identity,
        checks=checks,
        coverage=coverage,
        results=results,
        warnings=warnings,
        errors=errors,
        overall_status=status,
        exit_code=2 if errors else 1 if status == Status.FAIL else 0,
    )
