"""One-column, descriptive segment measurements; never train or optimize."""

import numpy as np
import pandas as pd

from modelgate.metrics import classification_metrics, validate_fraction, validate_inputs

COMPARISON_METRICS = ("precision", "recall", "f1", "pr_auc")


def _segment_values(segments, size: int) -> np.ndarray:
    values = np.asarray(segments, dtype=object)
    if values.ndim != 1 or len(values) != size:
        raise ValueError("Segments must be a one-dimensional vector matching labels.")
    normalized = []
    for value in values:
        if isinstance(value, str):
            # Do not trim nonblank names or collide with literal 'missing' groups.
            normalized.append(value if value.strip() else None)
        elif pd.api.types.is_scalar(value) and pd.isna(value):
            normalized.append(None)
        else:
            raise ValueError("Segment values must be strings or missing values.")
    return np.asarray(normalized, dtype=object)


def analyze_segments(
    labels,
    probabilities,
    segments,
    *,
    min_segment_size: int = 50,
    decision_threshold: float = 0.5,
    target_labels=(0, 1),
    positive_label=1,
) -> dict:
    """Arrays align positionally; every supplied Series must share its index."""
    indexed = [
        value
        for value in (labels, probabilities, segments)
        if isinstance(value, pd.Series)
    ]
    if any(not value.index.equals(indexed[0].index) for value in indexed[1:]):
        raise ValueError("Label, probability and segment Series indexes must match.")
    if (
        isinstance(min_segment_size, bool)
        or not isinstance(min_segment_size, int)
        or min_segment_size < 1
    ):
        raise ValueError("min_segment_size must be a positive integer.")
    threshold = validate_fraction(decision_threshold)
    y, p = validate_inputs(labels, probabilities, target_labels, positive_label)
    values = _segment_values(segments, len(y))
    overall = classification_metrics(y, p, threshold)
    groups = []
    for name in sorted({value for value in values if value is not None}):
        mask = values == name
        count = int(mask.sum())
        positive = int(y[mask].sum())
        eligible = count >= min_segment_size
        groups.append(
            {
                "segment": name,
                "record_count": count,
                "positive_count": positive,
                "negative_count": count - positive,
                "prevalence": positive / count,
                "eligible": eligible,
                "metrics": classification_metrics(y[mask], p[mask], threshold)
                if eligible
                else None,
            }
        )
    best = {}
    for metric in COMPARISON_METRICS:
        candidates = [
            group
            for group in groups
            if group["eligible"] and group["metrics"][metric] is not None
        ]
        value = max((group["metrics"][metric] for group in candidates), default=None)
        best[metric] = {
            "value": value,
            "segments": [
                group["segment"]
                for group in candidates
                if group["metrics"][metric] == value
            ],
            "candidate_count": len(candidates),
            "reason": "No eligible segment has a defined metric."
            if value is None
            else None,
        }
    for group in groups:
        comparisons = {}
        for metric in COMPARISON_METRICS:
            value = group["metrics"][metric] if group["eligible"] else None
            unavailable = (
                "Segment is below min_segment_size."
                if not group["eligible"]
                else group["metrics"]["undefined_reasons"].get(metric)
            )
            overall_reason = unavailable or overall["undefined_reasons"].get(metric)
            best_reason = unavailable or best[metric]["reason"]
            comparisons[metric] = {
                "from_overall": value - overall[metric] if not overall_reason else None,
                "from_best": value - best[metric]["value"] if not best_reason else None,
                "overall_reason": overall_reason,
                "best_reason": best_reason,
            }
        group["differences"] = comparisons
    named_count = sum(group["record_count"] for group in groups)
    eligible_count = sum(group["record_count"] for group in groups if group["eligible"])
    return {
        "decision_threshold": threshold,
        "min_segment_size": min_segment_size,
        "overall": overall,
        "segments": groups,
        "best_by_metric": best,
        "difference_definition": "segment minus comparator, in metric units",
        "coverage": {
            "total_rows": len(y),
            "named_rows": named_count,
            "missing_segment_rows": len(y) - named_count,
            "named_fraction": named_count / len(y),
            "eligible_rows": eligible_count,
            "eligible_fraction": eligible_count / len(y),
            "segment_count": len(groups),
            "eligible_segment_count": sum(group["eligible"] for group in groups),
            "overall_includes_missing_segments": True,
        },
    }
