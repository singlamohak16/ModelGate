"""Evaluate trusted local Phase 1 artifacts; not the Phase 6 validation CLI."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib

from modelgate import __version__
from modelgate.checks.model import ModelCheckPolicy, run_model_checks
from modelgate.fingerprint import sha256_file
from modelgate.predictions import positive_probabilities
from modelgate.schema import FEATURE_COLUMNS, TARGET_COLUMN
from modelgate.training import load_prepared_data


def build_demo(prepared_dir, model_dir):
    """Only use with artifacts you trust: joblib loading can execute code."""
    model_dir = Path(model_dir)
    _, reference, manifest = load_prepared_data(prepared_dir)
    provenance_path = model_dir / "baseline_metrics.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["data"] != manifest:
        raise ValueError("Model training data manifest does not match prepared data.")
    policy = ModelCheckPolicy()
    records = {}
    # Validate every artifact hash before deserializing any model. A hash detects
    # changes, not malicious artifacts; the caller must establish trust first.
    for name in ("logistic_regression", "random_forest"):
        path = model_dir / f"{name}.joblib"
        record = provenance["models"][name]
        if record["artifact"]["file"] != path.name or (
            sha256_file(path) != record["artifact"]["sha256"]
        ):
            raise ValueError(f"Model fingerprint mismatch: {name}.")
    for name in ("logistic_regression", "random_forest"):
        model = joblib.load(model_dir / f"{name}.joblib")
        probabilities = positive_probabilities(model, reference[list(FEATURE_COLUMNS)])
        records[name] = {
            "model_provenance": provenance["models"][name],
            "validation": run_model_checks(
                reference[TARGET_COLUMN], probabilities, policy
            ),
        }
    return {
        "report_kind": "phase3_model_checks_demo",
        "tool_version": __version__,
        "evaluation_environment": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "joblib")
            },
        },
        "policy": policy.model_dump(mode="json"),
        "training_provenance_sha256": sha256_file(provenance_path),
        "data": manifest,
        "seed": provenance["seed"],
        "training_environment": provenance["environment"],
        "models": records,
        "limitations": [
            "No quality limits configured; PASS measurements are descriptive only.",
            "One reference split, not threshold optimization or a release decision.",
            "Brier score measures probability error, not calibration alone.",
            "Calibration-bin counts expose sample sizes, not uncertainty intervals.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument(
        "--trust-local-models",
        action="store_true",
        help="Confirm these joblib files came from a trusted local run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/phase3_model_checks.json"),
    )
    args = parser.parse_args()
    if not args.trust_local_models:
        parser.exit(2, "Only trusted models may be loaded; use --trust-local-models.\n")
    if args.output.exists():
        parser.exit(2, "Demo output already exists; choose a new --output path.\n")
    try:
        document = build_demo(args.prepared_dir, args.model_dir)
        content = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Model-check demo error: {error}\n")
    print(f"Demo written to {args.output}; no overall validation status assigned.")


if __name__ == "__main__":
    main()
