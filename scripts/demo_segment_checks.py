"""Evaluate trusted local baselines by Contract; not the future validation CLI."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib

from modelgate import __version__
from modelgate.checks.segment import SegmentCheckPolicy, run_segment_checks
from modelgate.fingerprint import sha256_file
from modelgate.predictions import positive_probabilities
from modelgate.schema import FEATURE_COLUMNS
from modelgate.training import load_prepared_data


def build_demo(prepared_dir, model_dir) -> dict:
    """Caller must trust the artifacts; matching hashes do not establish trust."""
    model_dir = Path(model_dir)
    _, reference, manifest = load_prepared_data(prepared_dir)
    provenance_path = model_dir / "baseline_metrics.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["data"] != manifest:
        raise ValueError("Model training data manifest does not match prepared data.")
    names = ("logistic_regression", "random_forest")
    for name in names:
        path = model_dir / f"{name}.joblib"
        artifact = provenance["models"][name]["artifact"]
        if artifact["file"] != path.name or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Model fingerprint mismatch: {name}.")
    policy = SegmentCheckPolicy()
    records = {}
    for name in names:
        model = joblib.load(model_dir / f"{name}.joblib")
        probabilities = positive_probabilities(model, reference[list(FEATURE_COLUMNS)])
        records[name] = {
            "model_provenance": provenance["models"][name],
            "validation": run_segment_checks(reference, probabilities, policy),
        }
    return {
        "report_kind": "phase4_segment_checks_demo",
        "tool_version": __version__,
        "policy": policy.model_dump(mode="json"),
        "data": manifest,
        "seed": provenance["seed"],
        "training_provenance_sha256": sha256_file(provenance_path),
        "training_environment": provenance["environment"],
        "evaluation_environment": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "joblib")
            },
        },
        "models": records,
        "limitations": [
            "No quality limits configured; metric PASS is descriptive only.",
            "Minimum 50 rows is a reporting safeguard, not a reliability guarantee.",
            "Differences are descriptive, not significance or fairness guarantees.",
            "Best means highest observed defined metric among size-eligible segments.",
            "Prevalence and class counts differ; group difficulty can differ too.",
            "One reference split; no fitting or segment-specific threshold selection.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument("--trust-local-models", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/phase4_segment_checks.json"),
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
        parser.exit(2, f"Segment-check demo error: {error}\n")
    print(f"Demo written to {args.output}; no overall validation status assigned.")


if __name__ == "__main__":
    main()
