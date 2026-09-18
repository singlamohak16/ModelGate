"""Binary measurements with explicit undefined values and no policy decisions."""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

DEFAULT_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def label_token(value):
    """Accept string/integer labels (including CSV strings), never booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean labels are not supported.")
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError("Labels must be nonblank strings or integers.")


def validate_labels(target_labels, positive_label):
    tokens = tuple(label_token(value) for value in target_labels)
    positive = label_token(positive_label)
    if len(tokens) != 2 or len(set(tokens)) != 2 or positive not in tokens:
        raise ValueError("Require two distinct labels including the positive label.")
    return tokens, positive


def validate_fraction(value, name="threshold"):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return float(value)


def validate_inputs(labels, probabilities, target_labels=(0, 1), positive_label=1):
    """Check vectors before encoding; arrays are positional, Series must align."""
    tokens, positive = validate_labels(target_labels, positive_label)
    if (
        isinstance(labels, pd.Series)
        and isinstance(probabilities, pd.Series)
        and not labels.index.equals(probabilities.index)
    ):
        raise ValueError("Label and probability Series indexes must match.")
    raw_y = np.asarray(labels, dtype=object)
    raw_p = np.asarray(probabilities)
    if raw_y.ndim != 1 or not raw_y.size or raw_p.shape != raw_y.shape:
        raise ValueError("Require equal-length, nonempty one-dimensional vectors.")
    if raw_p.dtype.kind not in "iuf" or any(
        isinstance(value, (bool, np.bool_))
        for value in np.asarray(probabilities, dtype=object).flat
    ):
        raise ValueError("Probabilities must be numeric, not booleans or strings.")
    p = raw_p.astype(float, copy=True)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    encoded = [label_token(value) for value in raw_y]
    if not set(encoded) <= set(tokens):
        raise ValueError("Observed labels are outside the configured binary labels.")
    return np.array([value == positive for value in encoded], dtype=int), p


def _classification(y, p, threshold):
    predicted = p >= threshold
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    reasons = {}
    if tp + fp == 0:
        reasons["precision"] = "No positive predictions."
    if tp + fn == 0:
        reasons["recall"] = "No actual positive outcomes."
    if 2 * tp + fp + fn == 0:
        reasons["f1"] = "No actual or predicted positives."
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "predicted_positive_rate": (tp + fp) / len(y),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_matrix_layout": "rows=actual, columns=predicted; [[TN,FP],[FN,TP]]",
        "undefined_reasons": reasons,
    }


def classification_metrics(
    labels, probabilities, threshold=0.5, *, target_labels=(0, 1), positive_label=1
):
    y, p = validate_inputs(labels, probabilities, target_labels, positive_label)
    threshold = validate_fraction(threshold)
    result = _classification(y, p, threshold)
    result.update(
        sample_size=len(y),
        positive_count=int(y.sum()),
        prevalence=float(y.mean()),
        pr_auc=None,
        average_precision=None,
        pr_auc_method="trapezoidal",
    )
    if len(np.unique(y)) == 2:
        precision, recall, _ = precision_recall_curve(y, p)
        result["pr_auc"] = float(auc(recall, precision))
        result["average_precision"] = float(average_precision_score(y, p))
    else:
        for name in ("pr_auc", "average_precision"):
            result["undefined_reasons"][name] = "Both outcome classes are required."
    return result


def threshold_analysis(
    labels,
    probabilities,
    thresholds=DEFAULT_THRESHOLDS,
    *,
    target_labels=(0, 1),
    positive_label=1,
):
    y, p = validate_inputs(labels, probabilities, target_labels, positive_label)
    grid = tuple(validate_fraction(value) for value in thresholds)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("Threshold grid must be nonempty and unique.")
    return [_classification(y, p, value) for value in grid]


def probability_metrics(
    labels, probabilities, n_bins=10, *, target_labels=(0, 1), positive_label=1
):
    y, p = validate_inputs(labels, probabilities, target_labels, positive_label)
    if (
        isinstance(n_bins, bool)
        or not isinstance(n_bins, int)
        or not 2 <= n_bins <= 100
    ):
        raise ValueError("n_bins must be an integer between 2 and 100.")
    edges = np.linspace(0, 1, n_bins + 1)
    assignments = np.searchsorted(edges[1:-1], p, side="left")
    bins = []
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_probability": float(p[mask].mean()) if count else None,
                "positive_fraction": float(y[mask].mean()) if count else None,
            }
        )
    return {
        "brier_score": float(np.mean((p - y) ** 2)),
        "brier_definition": "mean((positive_probability - binary_outcome)^2)",
        "calibration_strategy": "uniform",
        "bin_boundaries": "First bin [lower,upper]; subsequent bins (lower,upper].",
        "bins": bins,
    }
