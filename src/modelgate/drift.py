"""Distribution measurements, with no significance or performance claims."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from modelgate.checks._common import missing_mask, numerical_values
from modelgate.metrics import validate_fraction


def finite_vector(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iuf":
        raise ValueError("Require a nonempty one-dimensional numeric vector.")
    if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(values, dtype=object)):
        raise ValueError("Boolean values are not measurements.")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Measurements must be finite.")
    return result


def probability_vector(values) -> np.ndarray:
    result = finite_vector(values)
    if ((result < 0) | (result > 1)).any():
        raise ValueError("Probabilities must be in [0, 1].")
    return result


def ks_distance(reference, current) -> float:
    """Maximum empirical CDF gap; do not compute an unused inferential p-value."""
    ref = np.sort(finite_vector(reference))
    cur = np.sort(finite_vector(current))
    points = np.union1d(ref, cur)
    ref_cdf = np.searchsorted(ref, points, side="right") / len(ref)
    cur_cdf = np.searchsorted(cur, points, side="right") / len(cur)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def numerical_sample(values) -> tuple[np.ndarray, dict]:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("Feature samples must be one-dimensional.")
    if any(not pd.api.types.is_scalar(value) for value in raw):
        raise ValueError("Numerical feature observations must be scalar values.")
    series = pd.Series(raw)
    missing = missing_mask(series)
    parsed, invalid = numerical_values(series, "number")
    usable = ~missing & ~invalid
    counts = {
        "total": len(series),
        "missing": int(missing.sum()),
        "invalid": int(invalid.sum()),
        "usable": int(usable.sum()),
        "usable_fraction": float(usable.mean()) if len(series) else None,
    }
    return parsed[usable].to_numpy(dtype=float), counts


def categorical_sample(values) -> list[tuple[str, str]]:
    """Typed keys prevent real category names colliding with missing values."""
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise ValueError(
            "Categorical samples must be nonempty one-dimensional vectors."
        )
    keys = []
    for value in raw:
        if isinstance(value, str):
            keys.append(("value", value) if value.strip() else ("missing", ""))
        elif pd.api.types.is_scalar(value) and pd.isna(value):
            keys.append(("missing", ""))
        else:
            raise ValueError("Categories must be strings or missing scalar values.")
    return keys


def categorical_distance(
    reference, current, *, smoothing=1e-6, max_examples=10
) -> dict:
    from collections import Counter

    epsilon = validate_fraction(smoothing, "smoothing")
    if epsilon == 1:
        raise ValueError("Smoothing must be less than one.")
    if (
        isinstance(max_examples, bool)
        or not isinstance(max_examples, int)
        or not 0 <= max_examples <= 20
    ):
        raise ValueError("max_examples must be an integer in [0, 20].")
    ref, cur = categorical_sample(reference), categorical_sample(current)
    ref_counts, cur_counts = Counter(ref), Counter(cur)
    support = sorted(ref_counts.keys() | cur_counts.keys())
    p = np.array([ref_counts[key] / len(ref) for key in support])
    q = np.array([cur_counts[key] / len(cur) for key in support])
    smoothed_p = (1 - epsilon) * p + epsilon / len(support)
    smoothed_q = (1 - epsilon) * q + epsilon / len(support)
    # Equality shortcut avoids floating-point roundoff for identical proportions.
    distance = (
        0.0
        if np.array_equal(p, q)
        else float(jensenshannon(smoothed_p, smoothed_q, base=2))
    )
    if not np.isfinite(distance):
        raise ValueError("Categorical distance could not be computed finitely.")
    examples = []
    for index in sorted(
        range(len(support)), key=lambda i: (-abs(q[i] - p[i]), support[i])
    )[:max_examples]:
        key = support[index]
        examples.append(
            {
                "category": key[1] if key[0] == "value" else None,
                "is_missing": key[0] == "missing",
                "reference_count": ref_counts[key],
                "current_count": cur_counts[key],
                "reference_fraction": float(p[index]),
                "current_fraction": float(q[index]),
                "reference_smoothed": float(smoothed_p[index]),
                "current_smoothed": float(smoothed_q[index]),
            }
        )
    return {
        "value": distance,
        "reference_count": len(ref),
        "current_count": len(cur),
        "category_count": len(support),
        "smoothing": epsilon,
        "smoothing_definition": "(1-epsilon)*observed_fraction + epsilon/K",
        "log_base": 2,
        "reference_missing": ref_counts[("missing", "")],
        "current_missing": cur_counts[("missing", "")],
        "new_category_count": len(cur_counts.keys() - ref_counts.keys()),
        "removed_category_count": len(ref_counts.keys() - cur_counts.keys()),
        "examples": examples,
        "examples_truncated": len(support) > len(examples),
    }
