"""Baseline fitting and descriptive measurements; validation rules arrive later."""

import json
import platform
import subprocess
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from modelgate import __version__
from modelgate.dataset import prepare_frame, read_csv_strings, write_json
from modelgate.fingerprint import sha256_file
from modelgate.preprocessing import build_baselines
from modelgate.schema import FEATURE_COLUMNS, ID_COLUMN, SEGMENT_COLUMN, TARGET_COLUMN


def baseline_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Measure class-1 performance at 0.50 and PR area from probabilities."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or probabilities.shape != y_true.shape or not len(y_true):
        raise ValueError(
            "Labels and probabilities must be equal-length nonempty vectors."
        )
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Baseline metrics require both binary classes 0 and 1.")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", pos_label=1, zero_division=0
    )
    curve_precision, curve_recall, _ = precision_recall_curve(y_true, probabilities)
    return {
        "threshold": 0.5,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_rate": float(predicted.mean()),
        "pr_auc": float(auc(curve_recall, curve_precision)),
        "pr_auc_method": "trapezoidal",
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_layout": (
            "rows=actual, columns=predicted, labels=[0,1]; [[TN,FP],[FN,TP]]"
        ),
        "warnings": (
            ["Precision is undefined without positive predictions; reported as 0."]
            if not predicted.any()
            else []
        ),
    }


def load_prepared_data(
    directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verify the prepared manifest and partitions before any model fitting."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 1:
        raise ValueError("Unsupported preparation manifest version.")
    partitions = []
    for name in ("train", "reference"):
        metadata = manifest["partitions"][name]
        path = directory / f"{name}.csv"
        if metadata["file"] != path.name or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Fingerprint mismatch for prepared {name} data.")
        partition = prepare_frame(read_csv_strings(path), encoded_target=True)
        if len(partition) != metadata["rows"]:
            raise ValueError(f"Row count mismatch for {name} data.")
        if int(partition[TARGET_COLUMN].sum()) != metadata["positive_count"]:
            raise ValueError(f"Positive count mismatch for {name} data.")
        partitions.append(partition)
    train, reference = partitions
    overlap = set(train[ID_COLUMN]) & set(reference[ID_COLUMN])
    if overlap:
        raise ValueError(f"Prepared partitions overlap on {len(overlap)} customer IDs.")
    return train, reference, manifest


def _git_state() -> dict:
    """Record local source provenance when Git is available."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def train_baselines(prepared_dir: str | Path, output_dir: str | Path) -> dict:
    """Fit both models on identical data and save pipelines plus JSON metadata."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Training output exists: {output_dir}. Choose a new --output-dir."
        )
    train, reference, manifest = load_prepared_data(prepared_dir)
    seed = manifest["seed"]
    models = build_baselines(seed=seed)
    features = list(FEATURE_COLUMNS)
    train_x, reference_x = train[features], reference[features]
    results = {
        "report_kind": "baseline_training_summary",
        "tool_version": __version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "git": _git_state(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")
            },
        },
        "data": manifest,
        "features": features,
        "target": TARGET_COLUMN,
        "positive_label": 1,
        "identifiers": [ID_COLUMN],
        "segment": SEGMENT_COLUMN,
        "models": {},
        "limitations": [
            "Descriptive baseline results, not a ModelGate validation audit.",
            "One random reference split; no tuning or independent temporal evaluation.",
            "No calibration, segment, drift, or release-threshold checks yet.",
        ],
    }
    # Do not create model outputs until input validation has succeeded.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, pipeline in models.items():
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit(train_x, train[TARGET_COLUMN])
        classes = pipeline.named_steps["classifier"].classes_
        positive_index = int(np.flatnonzero(classes == 1)[0])
        probabilities = pipeline.predict_proba(reference_x)[:, positive_index]
        metrics = baseline_metrics(reference[TARGET_COLUMN].to_numpy(), probabilities)
        path = output_dir / f"{name}.joblib"
        joblib.dump(pipeline, path)
        # This reload is safe because this process just created the artifact.
        restored = joblib.load(path)
        np.testing.assert_allclose(
            restored.predict_proba(reference_x)[:, positive_index],
            probabilities,
            rtol=0,
            atol=1e-12,
        )
        results["models"][name] = {
            "model_type": type(pipeline.named_steps["classifier"]).__name__,
            "parameters": pipeline.named_steps["classifier"].get_params(deep=False),
            "classes": classes.tolist(),
            "transformed_feature_count": len(
                pipeline.named_steps["preprocessor"].get_feature_names_out()
            ),
            "artifact": {"file": path.name, "sha256": sha256_file(path)},
            "reference_metrics": metrics,
            "training_warnings": [
                f"{item.category.__name__}: {item.message}" for item in captured
            ],
            "roundtrip_predictions_verified": True,
        }
    write_json(output_dir / "baseline_metrics.json", results)
    return results
