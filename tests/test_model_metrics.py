import json

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from modelgate.metrics import (
    classification_metrics,
    probability_metrics,
    threshold_analysis,
)
from modelgate.training import baseline_metrics


def test_hand_calculations_and_phase1_compatibility():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    result = classification_metrics(y, p)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["precision"] == 1
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["pr_auc"] == pytest.approx(19 / 24)
    assert result["average_precision"] == pytest.approx(5 / 6)
    assert result["predicted_positive_rate"] == 0.25
    old = baseline_metrics(y, p)
    for name in ("precision", "recall", "f1", "pr_auc", "average_precision"):
        assert result[name] == pytest.approx(old[name])
    assert probability_metrics(y, p)["brier_score"] == pytest.approx(0.158125)


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_trusted_sklearn_metrics_and_thresholds(seed):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], 50)
    p = rng.uniform(size=len(y))
    first = threshold_analysis(y, p)
    assert first == threshold_analysis(y, p)
    for row in first:
        predicted = p >= row["threshold"]
        assert row["precision"] == precision_score(y, predicted)
        assert row["recall"] == recall_score(y, predicted)
        assert row["f1"] == f1_score(y, predicted)
        assert row["confusion_matrix"] == confusion_matrix(y, predicted).tolist()
        assert sum(row[key] for key in ("tp", "fp", "tn", "fn")) == len(y)
    result = classification_metrics(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    assert result["pr_auc"] == auc(recall, precision)
    assert result["average_precision"] == average_precision_score(y, p)
    assert probability_metrics(y, p)["brier_score"] == brier_score_loss(y, p)


@pytest.mark.parametrize("p,expected", [([0, 1], 0), ([1, 0], 1), ([0.5, 0.5], 0.25)])
def test_brier_extremes(p, expected):
    assert probability_metrics([0, 1], p)["brier_score"] == expected


def test_calibration_boundaries_empty_bins_and_sklearn():
    p = np.array([0, 0.1, 0.2, 0.5, 0.5, 1])
    y = np.array([0, 1, 0, 1, 0, 1])
    result = probability_metrics(y, p)
    bins = result["bins"]
    assert [row["count"] for row in bins] == [2, 1, 0, 0, 2, 0, 0, 0, 0, 1]
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
    nonempty = [row for row in bins if row["count"]]
    np.testing.assert_allclose([row["positive_fraction"] for row in nonempty], observed)
    np.testing.assert_allclose([row["mean_probability"] for row in nonempty], predicted)
    assert bins[2]["mean_probability"] is None
    assert bins[2]["positive_fraction"] is None
    assert sum(row["count"] for row in bins) == len(y)
    json.dumps(result, allow_nan=False)


def test_exact_thresholds_and_no_input_mutation():
    y = pd.Series([0, 1, 1, 0, 1])
    p = pd.Series([0.3, 0.4, 0.5, 0.6, 0.7])
    original = p.copy()
    rows = threshold_analysis(y, p)
    assert [r["tp"] + r["fp"] for r in rows] == [5, 4, 3, 2, 1]
    pd.testing.assert_series_equal(p, original)
    assert classification_metrics([0, 1], [0, 1], 1)["tp"] == 1


def test_undefined_is_not_zero():
    result = classification_metrics([0, 1], [0.1, 0.2])
    assert result["precision"] is None
    assert result["recall"] == result["f1"] == 0
    result = classification_metrics([0, 0], [0.1, 0.2])
    for name in ("precision", "recall", "f1", "pr_auc", "average_precision"):
        assert result[name] is None
        assert result["undefined_reasons"][name]
    result = classification_metrics([1, 1], [0.9, 0.8])
    assert result["recall"] == 1
    assert result["pr_auc"] is None


def test_custom_labels_and_positive_direction():
    result = classification_metrics(
        ["No", "Yes"], [0.8, 0.2], target_labels=("No", "Yes"), positive_label="No"
    )
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]
    assert classification_metrics(["0", "1"], [0.1, 0.9])["f1"] == 1


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([0, 1], [0.1]),
        ([[0, 1]], [[0.1, 0.2]]),
        ([0, 1], [0.1, np.nan]),
        ([0, 1], [0.1, np.inf]),
        ([0, 1], [-0.1, 1]),
        ([0, 1], [0, 1.01]),
        ([0, 2], [0.1, 0.9]),
        ([False, True], [0.1, 0.9]),
        ([0, None], [0.1, 0.9]),
        ([0, " "], [0.1, 0.9]),
        ([0, 1], [False, True]),
        ([0, 1], [False, 0.5]),
        ([0, 1], ["0.1", "0.9"]),
    ],
)
def test_invalid_inputs(y, p):
    for fn in (classification_metrics, probability_metrics, threshold_analysis):
        with pytest.raises(ValueError):
            fn(y, p)


def test_series_alignment():
    with pytest.raises(ValueError, match="indexes"):
        classification_metrics(pd.Series([0, 1]), pd.Series([0.1, 0.9], index=[1, 0]))


@pytest.mark.parametrize("value", [True, -0.1, 1.1, np.nan, "0.5"])
def test_invalid_threshold(value):
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.1, 0.9], value)


@pytest.mark.parametrize("grid", [(), (0.5, 0.5), (True,), (1.1,)])
def test_invalid_grid(grid):
    with pytest.raises(ValueError):
        threshold_analysis([0, 1], [0.1, 0.9], grid)


@pytest.mark.parametrize("bins", [True, 1, 101, 2.5])
def test_invalid_bins(bins):
    with pytest.raises(ValueError):
        probability_metrics([0, 1], [0.1, 0.9], bins)
