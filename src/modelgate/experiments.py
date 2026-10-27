"""Deterministic software controls, not realistic churn or production evidence."""

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from modelgate import __version__
from modelgate.fingerprint import sha256_file
from modelgate.preprocessing import build_baselines
from modelgate.reporting import AuditReport
from modelgate.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    SEGMENT_COLUMN,
    TARGET_COLUMN,
)

ROWS = 1000
CONTAMINATION_ROWS = 20
SMALL_SEGMENT = "One year"
LARGE_SEGMENT = "Month-to-month"
LEAK_COLUMN = "injected_target_copy"
DRIFT_COLUMN = "tenure"
DRIFT_SHIFT = 100.0
SCENARIOS = (
    "clean",
    "contamination",
    "target_leakage",
    "feature_drift",
    "segment_degradation",
)
MODEL_NAMES = ("logistic_regression", "random_forest")


@dataclass
class Scenario:
    reference: pd.DataFrame
    current: pd.DataFrame | None
    current_labeled: bool
    mutation: dict


def generate_fixtures(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balanced outcomes in 900/100-row segments with continuous separated signal.

    Telco-shaped names permit reuse of the existing pipelines, but these are NOT
    simulated customers. Fractional tenure and constant other features are
    intentional laboratory controls. Train/reference use independent RNG streams.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    frames = []
    for split, stream in zip(
        ("train", "reference"), np.random.SeedSequence(seed).spawn(2), strict=True
    ):
        rng = np.random.default_rng(stream)
        labels = np.tile([0, 1], ROWS // 2)
        frame = pd.DataFrame(
            {column: ["constant"] * ROWS for column in CATEGORICAL_COLUMNS}
        )
        frame[SEGMENT_COLUMN] = [LARGE_SEGMENT] * 900 + [SMALL_SEGMENT] * 100
        # Labels are determined by which side of 30 tenure lies on; both sides
        # have a margin >=10. Other predictors contain no outcome information.
        frame[DRIFT_COLUMN] = 30 + (2 * labels - 1) * rng.uniform(10, 20, ROWS)
        frame["MonthlyCharges"] = 50.0
        frame["TotalCharges"] = 1000.0
        frame[TARGET_COLUMN] = labels
        frame[ID_COLUMN] = [f"synthetic-{split}-{i:04d}" for i in range(ROWS)]
        frame = frame.loc[:, list(REQUIRED_COLUMNS)]
        frames.append(frame.iloc[rng.permutation(ROWS)].reset_index(drop=True))
    return tuple(frames)


def build_scenarios(
    train: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, Scenario]:
    """Copy first, then inject only the documented mutation; never edit sources."""
    scenarios = {
        "clean": Scenario(
            reference.copy(deep=True),
            reference.copy(deep=True),
            True,
            {
                "operation": "none",
                "affected_positions": [],
                "current_source": (
                    "Exact reference copy: labeled software control, not temporal data."
                ),
            },
        ),
    }
    contaminated = reference.copy(deep=True)
    contaminated.iloc[:CONTAMINATION_ROWS] = train.iloc[:CONTAMINATION_ROWS].to_numpy()
    scenarios["contamination"] = Scenario(
        contaminated,
        None,
        False,
        {
            "operation": "replace complete reference rows with training rows",
            "affected_positions": list(range(CONTAMINATION_ROWS)),
            "source_train_positions": list(range(CONTAMINATION_ROWS)),
            "injected_ids": train[ID_COLUMN].iloc[:CONTAMINATION_ROWS].tolist(),
        },
    )
    leaked = reference.copy(deep=True)
    leaked[LEAK_COLUMN] = leaked[TARGET_COLUMN]
    scenarios["target_leakage"] = Scenario(
        leaked,
        None,
        False,
        {
            "operation": "add exact target copy as audit-only extra column",
            "column": LEAK_COLUMN,
            "affected_positions": list(range(len(reference))),
            "used_by_model": False,
        },
    )
    current = reference.drop(columns=TARGET_COLUMN).copy(deep=True)
    current[DRIFT_COLUMN] += DRIFT_SHIFT
    scenarios["feature_drift"] = Scenario(
        reference.copy(deep=True),
        current,
        False,
        {
            "operation": "add fixed numerical shift to unlabeled reference copy",
            "column": DRIFT_COLUMN,
            "shift": DRIFT_SHIFT,
            "affected_positions": list(range(len(reference))),
            "current_source": "Mutated reference copy, not new/temporal data.",
            "labels_removed": True,
        },
    )
    degraded = reference.copy(deep=True)
    selected = degraded[SEGMENT_COLUMN] == SMALL_SEGMENT
    degraded.loc[selected, DRIFT_COLUMN] = 60 - degraded.loc[selected, DRIFT_COLUMN]
    scenarios["segment_degradation"] = Scenario(
        degraded,
        None,
        False,
        {
            "operation": "reflect tenure around 30 within the selected segment",
            "column": DRIFT_COLUMN,
            "segment": SMALL_SEGMENT,
            "affected_positions": np.flatnonzero(selected.to_numpy()).tolist(),
            "labels_and_ids_unchanged": True,
        },
    )
    return scenarios


def scenario_config(name: str, model: str, digest: str, seed: int) -> dict:
    """Fixed illustrative limits shared across scenarios; no result-based tuning."""
    if name not in SCENARIOS or model not in MODEL_NAMES:
        raise ValueError("Unknown scenario or model.")
    return {
        "inputs": {
            "train": "../train.csv",
            "reference": f"../{name}/reference.csv",
            "current": f"../{name}/current.csv"
            if name in ("clean", "feature_drift")
            else None,
            "current_labeled": name == "clean",
        },
        "artifact": {
            "path": f"../models/{model}.joblib",
            "trust": True,
            "expected_sha256": digest,
        },
        "output": f"../reports/{model}--{name}.json",
        "seed": seed,
        "decision_threshold": 0.5,
        "model": {"min_f1": 0.85},
        "segment": {"min_segment_size": 50, "min_recall": 0.8},
        "drift": {
            "min_sample_size": 50,
            "max_numerical_ks": 0.2,
            "max_categorical_distance": 0.2,
            "max_prediction_ks": 0.2,
        },
        "performance": {"min_sample_size": 50},
    }


def assess_report(name: str, report: AuditReport, process_code: int) -> list[dict]:
    """Compare independently declared expectations to existing check evidence.

    Do not recalculate metrics or aggregate a second validation verdict. Missing
    or duplicate targeted checks are explicit mismatches, not silent successes.
    """
    if name not in SCENARIOS:
        raise ValueError("Unknown scenario.")
    findings = []

    def expect(label, actual, wanted):
        matched = (
            math.isclose(actual, wanted, rel_tol=0, abs_tol=1e-12)
            if type(actual) in (int, float) and type(wanted) in (int, float)
            else actual == wanted
        )
        findings.append(
            {
                "expectation": label,
                "expected": wanted,
                "observed": actual,
                "matched": matched,
            }
        )

    def check(rule, dataset="reference", **evidence):
        matches = [
            c
            for c in report.checks
            if c.check_id == rule
            and c.dataset == dataset
            and all(c.evidence.get(k) == v for k, v in evidence.items())
        ]
        label = f"{dataset}/{rule}/{evidence}"
        expect(f"unique check: {label}", len(matches), 1)
        if len(matches) != 1:
            return None
        expect(f"evaluated: {label}", matches[0].evaluated, True)
        return matches[0]

    def measurement(result, key):
        return result.measurement.get(key) if result else None

    def status(result):
        return result.status.value if result else None

    expected_code = 1 if name in ("contamination", "segment_degradation") else 0
    expected_status = (
        "FAIL" if expected_code else "PASS" if name == "clean" else "WARNING"
    )
    expect("execution completed", report.execution_state, "completed")
    expect("no execution errors", report.errors, [])
    expect("process exit", process_code, expected_code)
    expect("reported exit", report.exit_code, expected_code)
    expect("overall verdict", report.overall_status, expected_status)
    expected_failures = {
        "contamination": ["overlap.identifiers"],
        "segment_degradation": ["segment.recall"],
    }.get(name, [])
    expect(
        "failed rule IDs",
        sorted(c.check_id for c in report.checks if c.status == "FAIL"),
        expected_failures,
    )
    expect("seed retained", report.seed, report.effective_config["seed"])
    overlap = check("overlap.identifiers", "train/reference")
    count = CONTAMINATION_ROWS if name == "contamination" else 0
    expect("identifier overlap status", status(overlap), "FAIL" if count else "PASS")
    expect(
        "overlapping reference rows",
        measurement(overlap, "overlapping_reference_rows"),
        count,
    )
    expect("overlap denominator", measurement(overlap, "reference_rows"), ROWS)
    expect("overlap fraction", measurement(overlap, "fraction"), count / ROWS)
    overall = check("model.f1")
    expect("overall F1 rule", status(overall), "PASS")
    expect(
        "overall F1",
        measurement(overall, "value"),
        0.9 if name == "segment_degradation" else 1.0,
    )
    for group in (SMALL_SEGMENT, LARGE_SEGMENT):
        recall = check("segment.recall", segment=group)
        bad = name == "segment_degradation" and group == SMALL_SEGMENT
        expect(f"{group} recall rule", status(recall), "FAIL" if bad else "PASS")
        expect(f"{group} recall", measurement(recall, "value"), 0.0 if bad else 1.0)
    if name == "clean":
        expect("clean has no warnings", len(report.warnings), 0)
    if name == "contamination":
        records = check("overlap.records", "train/reference")
        expect(
            "predictor overlap rows",
            measurement(records, "overlapping_reference_rows"),
            count,
        )
    if name == "target_leakage":
        leak = check(f"leakage.target_like.{LEAK_COLUMN}")
        expect("target-like warning", status(leak), "WARNING")
        expect("target-like measurement", measurement(leak, "target_like"), True)
        expect(
            "copy relationship",
            leak.evidence.get("relationship") if leak else None,
            "exact_copy",
        )
        expect("copy coverage", measurement(leak, "coverage"), 1.0)
        expect(
            "leak excluded from model",
            LEAK_COLUMN in report.effective_config["predictors"],
            False,
        )
        expect(
            "no explicit prohibition",
            status(check("leakage.prohibited_columns")),
            "PASS",
        )
    if name in ("clean", "feature_drift"):
        for column in NUMERIC_COLUMNS:
            drift = check("drift.numerical_ks", "reference_vs_current", column=column)
            changed = name == "feature_drift" and column == DRIFT_COLUMN
            expect(f"{column} KS", measurement(drift, "value"), 1.0 if changed else 0.0)
            expect(
                f"{column} drift rule", status(drift), "WARNING" if changed else "PASS"
            )
        for column in CATEGORICAL_COLUMNS:
            drift = check(
                "drift.categorical_distance", "reference_vs_current", column=column
            )
            expect(f"{column} distance", measurement(drift, "value"), 0.0)
            expect(f"{column} drift rule", status(drift), "PASS")
        prediction = check("drift.prediction_ks", "reference_vs_current")
        if name == "clean":
            expect("self prediction KS", measurement(prediction, "value"), 0.0)
            for metric in ("precision", "recall", "f1", "pr_auc", "brier_score"):
                change = check(f"performance.{metric}", "reference_vs_current")
                expect(
                    f"self {metric} change",
                    measurement(change, "current_minus_reference"),
                    0.0,
                )
        else:
            # Prediction KS is model-dependent: require measured, bounded evidence,
            # not a hard-coded distance shared by different estimators.
            value = measurement(prediction, "value")
            expect(
                "prediction KS in [0,1]", value is not None and 0 <= value <= 1, True
            )
            expect(
                "no performance checks",
                any(c.check_id.startswith("performance.") for c in report.checks),
                False,
            )
            expect(
                "no current labeled metrics",
                any(
                    k in report.results
                    for k in (
                        "model.current",
                        "segment.current",
                        "performance.reference_vs_current",
                    )
                ),
                False,
            )
    return findings


def _write_json(path: Path, document: dict) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, lineterminator="\n", float_format="%.17g")


def run_experiments(output_dir: str | Path, *, seed: int = 42) -> dict:
    """Fit two synthetic baselines once; run ten ordinary CLI subprocess audits.

    Refuse existing output, even if empty. Partial output is retained on failure.
    No existing artifact is loaded or modified by experiment generation.
    """
    train, reference = generate_fixtures(seed)
    scenarios = build_scenarios(train, reference)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("models", "configs", "reports"):
        (root / directory).mkdir()
    _write_csv(root / "train.csv", train)
    for name, scenario in scenarios.items():
        folder = root / name
        folder.mkdir()
        _write_csv(folder / "reference.csv", scenario.reference)
        if scenario.current is not None:
            _write_csv(folder / "current.csv", scenario.current)
    provenance = {
        "schema_version": 1,
        "kind": "synthetic_software_controls",
        "created_at": datetime.now(UTC).isoformat(),
        "tool_version": __version__,
        "seed": seed,
        "rows_per_split": ROWS,
        "description": (
            "Invented Telco-shaped fixtures; not IBM data or realistic customers."
        ),
        "mutations": {name: s.mutation for name, s in scenarios.items()},
        "datasets": {
            p.relative_to(root).as_posix(): sha256_file(p)
            for p in sorted(root.rglob("*.csv"))
        },
        "models": {},
    }
    # Models fit only the clean training frame and exclude IDs/target/added leaks.
    for name, model in build_baselines(seed).items():
        model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
        artifact = root / "models" / f"{name}.joblib"
        joblib.dump(model, artifact)
        provenance["models"][name] = {
            "path": artifact.relative_to(root).as_posix(),
            "sha256": sha256_file(artifact),
        }
        for scenario in SCENARIOS:
            config = scenario_config(scenario, name, sha256_file(artifact), seed)
            path = root / "configs" / f"{name}--{scenario}.yaml"
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
    provenance["configurations"] = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted((root / "configs").glob("*.yaml"))
    }
    _write_json(root / "manifest.json", provenance)
    protected = {
        **provenance["datasets"],
        **provenance["configurations"],
        **{m["path"]: m["sha256"] for m in provenance["models"].values()},
    }
    rows = []
    for model in MODEL_NAMES:
        for scenario in SCENARIOS:
            slug = f"{model}--{scenario}"
            row = {
                "model": model,
                "scenario": scenario,
                "config": f"configs/{slug}.yaml",
                "report": f"reports/{slug}.json",
                "process_exit": None,
                "audit_status": None,
                "expectations": [],
                "matched": False,
                "error": None,
            }
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "modelgate",
                        "validate",
                        "--config",
                        str(root / row["config"]),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                row.update(
                    process_exit=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
                report = AuditReport.model_validate_json(
                    (root / row["report"]).read_text(encoding="utf-8")
                )
                row["audit_status"] = report.overall_status
                if (
                    completed.returncode not in (0, 1)
                    or report.execution_state != "completed"
                ):
                    raise RuntimeError(
                        f"Audit execution failed: {completed.stderr or report.errors}"
                    )
                # Require the report to identify exactly the generated inputs/model.
                expected_files = {
                    "configuration": row["config"],
                    "train": "train.csv",
                    "reference": f"{scenario}/reference.csv",
                    "model": f"models/{model}.joblib",
                }
                if scenario in ("clean", "feature_drift"):
                    expected_files["current"] = f"{scenario}/current.csv"
                if set(report.fingerprints) != set(expected_files):
                    raise ValueError(
                        "Audit fingerprint roles differ from generated inputs."
                    )
                for role, relative in expected_files.items():
                    item = report.fingerprints[role]
                    if (
                        Path(item["path"]).resolve() != root / relative
                        or item["sha256"] != protected[relative]
                    ):
                        raise ValueError(
                            "Audit fingerprint differs from generated file."
                        )
                row["expectations"] = assess_report(
                    scenario, report, completed.returncode
                )
                row["matched"] = all(item["matched"] for item in row["expectations"])
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
    # Make accidental mutation by validation explicit, including fitted artifacts.
    changed = [
        path
        for path, digest in protected.items()
        if not (root / path).is_file() or sha256_file(root / path) != digest
    ]
    summary = {
        "schema_version": 1,
        "kind": "synthetic_expected_vs_observed",
        "seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": "manifest.json",
        "rows": rows,
        "unexpected_input_changes": changed,
        "all_matched": not changed and all(row["matched"] for row in rows),
        "runner_exit_code": 2
        if changed or any(row["error"] for row in rows)
        else 0
        if all(row["matched"] for row in rows)
        else 1,
    }
    _write_json(root / "summary.json", summary)
    return summary
