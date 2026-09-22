"""Reference-versus-itself controls, NOT observations of real current data."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib

from modelgate import __version__
from modelgate.checks.drift import DriftCheckPolicy, run_drift_checks
from modelgate.checks.performance import PerformanceChangePolicy, run_performance_checks
from modelgate.fingerprint import sha256_file
from modelgate.predictions import positive_probabilities
from modelgate.schema import FEATURE_COLUMNS, TARGET_COLUMN
from modelgate.training import load_prepared_data


def build_demo(prepared_dir, model_dir) -> dict:
    """Caller must trust both artifacts; hashes alone do not establish trust."""
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
    drift_policy, performance_policy = DriftCheckPolicy(), PerformanceChangePolicy()
    current = reference.copy(deep=True)
    unlabeled = current.drop(columns=[TARGET_COLUMN])
    records = {}
    for name in names:
        model = joblib.load(model_dir / f"{name}.joblib")
        p_ref = positive_probabilities(model, reference[list(FEATURE_COLUMNS)])
        p_cur = positive_probabilities(model, current[list(FEATURE_COLUMNS)])
        modes = {}
        for mode, frame, labeled in (
            ("labeled", current, True),
            ("unlabeled", unlabeled, False),
        ):
            modes[mode] = {
                "drift_checks": [
                    check.model_dump(mode="json")
                    for check in run_drift_checks(
                        reference,
                        frame,
                        drift_policy,
                        reference_probabilities=p_ref,
                        current_probabilities=p_cur,
                    )
                ],
                "performance": run_performance_checks(
                    reference,
                    frame,
                    p_ref,
                    p_cur,
                    performance_policy,
                    current_labeled=labeled,
                ),
            }
        records[name] = {
            "model_provenance": provenance["models"][name],
            "controls": modes,
        }
    return {
        "report_kind": "phase5_self_comparison_controls",
        "tool_version": __version__,
        "current_source": "Copy of reference; NOT new or temporal data.",
        "reference_current_overlap_fraction": 1.0,
        "drift_policy": drift_policy.model_dump(mode="json"),
        "performance_policy": performance_policy.model_dump(mode="json"),
        "data": manifest,
        "seed": provenance["seed"],
        "training_provenance_sha256": sha256_file(provenance_path),
        "training_environment": provenance["environment"],
        "evaluation_environment": {
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
            },
        },
        "models": records,
        "limitations": [
            "Self-comparison verifies a zero-change control, not real-world stability.",
            "No significance tests, causal claims or unlabeled performance inference.",
            "No configured quality limits; descriptive PASS is not release approval.",
            "No fitting, tuning, downloading, or Phase 7 scenario framework.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument("--trust-local-models", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/phase5_controls.json")
    )
    args = parser.parse_args()
    if not args.trust_local_models:
        parser.exit(2, "Only trusted models may be loaded; use --trust-local-models.\n")
    if args.output.exists():
        parser.exit(2, "Demo output already exists; choose a new --output path.\n")
    try:
        content = (
            json.dumps(
                build_demo(args.prepared_dir, args.model_dir),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Drift-control demo error: {error}\n")
    print(f"Self-comparison controls written to {args.output}; not real current data.")


if __name__ == "__main__":
    main()
