"""Exercise the installed console command on synthetic PASS/FAIL/error cases."""

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import joblib
import yaml

import modelgate
from modelgate.experiments import (
    assess_report,
    build_scenarios,
    generate_fixtures,
    scenario_config,
)
from modelgate.fingerprint import sha256_file
from modelgate.preprocessing import build_baselines
from modelgate.reporting import AuditReport
from modelgate.schema import FEATURE_COLUMNS, TARGET_COLUMN


def check_installed_package():
    """Reject editable/PYTHONPATH imports during the isolated wheel smoke test."""
    expected = Path(distribution("modelgate").locate_file("modelgate/__init__.py"))
    actual = Path(modelgate.__file__).resolve()
    if not expected.is_file() or expected.resolve() != actual:
        raise ValueError(
            "Smoke test requires the installed wheel, not an editable checkout."
        )
    return str(actual)


def run_smoke(output_dir, *, require_wheel=False):
    module_path = check_installed_package() if require_wheel else modelgate.__file__
    executable = Path(sys.executable).parent / (
        "modelgate.exe" if os.name == "nt" else "modelgate"
    )
    if not executable.is_file():
        raise FileNotFoundError(f"Installed console entry point missing: {executable}")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("models", "configs", "reports", "clean", "contamination"):
        (root / name).mkdir()
    train, reference = generate_fixtures()
    scenarios = build_scenarios(train, reference)
    train.to_csv(root / "train.csv", index=False)
    for name in ("clean", "contamination"):
        scenario = scenarios[name]
        scenario.reference.to_csv(root / name / "reference.csv", index=False)
        if scenario.current is not None:
            scenario.current.to_csv(root / name / "current.csv", index=False)
    model = build_baselines()["logistic_regression"]
    model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    artifact = root / "models" / "logistic_regression.joblib"
    joblib.dump(model, artifact)
    rows = []
    for name, expected_exit in (("clean", 0), ("contamination", 1), ("input_error", 2)):
        config = scenario_config(
            "clean" if name == "input_error" else name,
            "logistic_regression",
            sha256_file(artifact),
            42,
        )
        config["output"] = f"../reports/{name}.json"
        if name == "input_error":
            config["inputs"]["reference"] = "../intentionally-absent.csv"
        path = root / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        process = subprocess.run(
            [str(executable), "validate", "--config", str(path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if process.returncode != expected_exit:
            raise ValueError(
                f"{name}: expected exit {expected_exit}, got {process.returncode}: "
                f"{process.stderr}"
            )
        report = AuditReport.model_validate_json(
            (root / "reports" / f"{name}.json").read_text(encoding="utf-8")
        )
        if name == "input_error":
            if (
                report.execution_state != "error"
                or report.overall_status is not None
                or report.exit_code != 2
                or not report.errors
            ):
                raise ValueError(
                    "Input error did not produce an error report without verdict."
                )
        else:
            assertions = assess_report(name, report, process.returncode)
            if not all(item["matched"] for item in assertions):
                raise ValueError(f"Unexpected evidence in {name}: {assertions}")
        rows.append(
            {
                "case": name,
                "expected_exit": expected_exit,
                "actual_exit": process.returncode,
                "status": report.overall_status,
                "execution_state": report.execution_state,
                "report": f"reports/{name}.json",
            }
        )
    summary = {
        "kind": "synthetic_installed_cli_smoke",
        "wheel_verified": require_wheel,
        "module_path": str(module_path),
        "console_path": str(executable),
        "cases": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-wheel", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_smoke(args.output_dir, require_wheel=args.require_wheel)
    except Exception as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
