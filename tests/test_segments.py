import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve

from modelgate.metrics import classification_metrics
from modelgate.segments import COMPARISON_METRICS, analyze_segments


def test_hand_counts_overall_and_best_differences():
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    p = [0.1, 0.4, 0.35, 0.8, 0.1, 0.9, 0.6, 0.7]
    result = analyze_segments(y, p, ["A"] * 4 + ["B"] * 4, min_segment_size=4)
    a, b = result["segments"]
    assert a["record_count"] == 4
    assert a["positive_count"] == a["negative_count"] == 2
    assert a["prevalence"] == 0.5
    assert a["metrics"]["confusion_matrix"] == [[2, 0], [1, 1]]
    assert b["metrics"]["confusion_matrix"] == [[1, 1], [0, 2]]
    assert a["metrics"]["precision"] == 1
    assert b["metrics"]["recall"] == 1
    assert result["overall"]["confusion_matrix"] == [[3, 1], [1, 3]]
    assert result["overall"]["f1"] == 0.75
    assert result["overall"]["f1"] != (a["metrics"]["f1"] + b["metrics"]["f1"]) / 2
    assert a["differences"]["recall"]["from_overall"] == -0.25
    assert a["differences"]["recall"]["from_best"] == -0.5
    assert result["best_by_metric"]["precision"]["segments"] == ["A"]
    assert result["best_by_metric"]["recall"]["segments"] == ["B"]
    assert result["coverage"]["eligible_rows"] == 8
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_group_metrics_match_sklearn_and_phase3(seed):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], 60)
    p = rng.uniform(size=len(y))
    segments = np.repeat(["C", "A", "B"], 40)
    result = analyze_segments(y, p, segments, min_segment_size=40)
    assert result == analyze_segments(y, p, segments, min_segment_size=40)
    assert [g["segment"] for g in result["segments"]] == ["A", "B", "C"]
    for group in result["segments"]:
        mask = segments == group["segment"]
        assert group["metrics"] == classification_metrics(y[mask], p[mask])
        assert (
            group["metrics"]["confusion_matrix"]
            == confusion_matrix(y[mask], p[mask] >= 0.5, labels=[0, 1]).tolist()
        )
        precision, recall, _ = precision_recall_curve(y[mask], p[mask])
        assert group["metrics"]["pr_auc"] == auc(recall, precision)
        for metric in COMPARISON_METRICS:
            assert group["differences"][metric]["from_overall"] == pytest.approx(
                group["metrics"][metric] - result["overall"][metric]
            )


def test_49_50_boundary_and_small_segment_excluded_from_best():
    y = [0, 1] * 24 + [1] + [0, 1] * 25
    p = y[:49] + [0.5] * 50
    result = analyze_segments(y, p, ["small"] * 49 + ["eligible"] * 50)
    eligible, small = result["segments"]
    assert not small["eligible"]
    assert small["metrics"] is None
    assert small["record_count"] == 49
    assert small["prevalence"] == 25 / 49
    assert eligible["eligible"]
    for metric in COMPARISON_METRICS:
        assert result["best_by_metric"][metric]["segments"] == ["eligible"]
        assert small["differences"][metric]["from_best"] is None
        assert small["differences"][metric]["best_reason"]
    assert result["coverage"]["eligible_fraction"] == 50 / 99


def test_ties_include_all_names_and_permutation_is_stable():
    y, p, groups = [0, 1, 0, 1], [0.1, 0.9, 0.1, 0.9], ["B", "B", "A", "A"]
    result = analyze_segments(y, p, groups, min_segment_size=2)
    assert result == analyze_segments(
        y[::-1], p[::-1], groups[::-1], min_segment_size=2
    )
    for metric in COMPARISON_METRICS:
        assert result["best_by_metric"][metric]["segments"] == ["A", "B"]
        assert result["best_by_metric"][metric]["candidate_count"] == 2
        for group in result["segments"]:
            assert group["differences"][metric]["from_best"] == 0


def test_missing_values_remain_in_overall_and_literal_names_do_not_collide():
    segments = [None, np.nan, pd.NA, "", " \t", "missing", "NA", " A "]
    result = analyze_segments([0, 1] * 4, [0.5] * 8, segments, min_segment_size=1)
    assert result["coverage"]["missing_segment_rows"] == 5
    assert result["coverage"]["named_fraction"] == 3 / 8
    assert result["overall"]["sample_size"] == 8
    assert [g["segment"] for g in result["segments"]] == [" A ", "NA", "missing"]


@pytest.mark.parametrize("segments", [[None, ""], ["A", "B"]])
def test_no_eligible_segments(segments):
    result = analyze_segments([0, 1], [0.1, 0.9], segments)
    for metric in COMPARISON_METRICS:
        assert result["best_by_metric"][metric]["value"] is None
        assert result["best_by_metric"][metric]["segments"] == []
        assert result["best_by_metric"][metric]["reason"]


def test_one_class_and_undefined_are_not_zeros():
    result = analyze_segments(
        [0, 0, 1, 1],
        [0.1, 0.1, 0.1, 0.9],
        ["negative"] * 2 + ["positive"] * 2,
        min_segment_size=2,
    )
    negative, positive = result["segments"]
    assert negative["metrics"]["precision"] is None
    assert negative["metrics"]["recall"] is None
    assert positive["metrics"]["recall"] == 0.5
    assert positive["metrics"]["pr_auc"] is None
    assert result["best_by_metric"]["pr_auc"]["value"] is None
    assert result["overall"]["pr_auc"] is not None
    assert negative["differences"]["precision"]["from_overall"] is None
    assert negative["differences"]["precision"]["overall_reason"]


def test_custom_labels_positive_direction_and_threshold_equality():
    result = analyze_segments(
        ["Yes", "No"],
        [0.1, 0.4],
        ["A", "A"],
        min_segment_size=2,
        decision_threshold=0.4,
        target_labels=("Yes", "No"),
        positive_label="No",
    )
    assert result["segments"][0]["metrics"]["confusion_matrix"] == [[1, 0], [0, 1]]


def test_no_mutation_and_duplicate_indexes_are_positional():
    y = pd.Series([0, 1, 0, 1], index=[3, 3, 1, 1])
    p = pd.Series([0.1, 0.9, 0.1, 0.9], index=y.index)
    groups = pd.Series(["A", "B", "A", "B"], index=y.index)
    originals = [x.copy(deep=True) for x in (y, p, groups)]
    analyze_segments(y, p, groups, min_segment_size=2)
    for actual, expected in zip((y, p, groups), originals, strict=True):
        pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_misaligned_series(which):
    inputs = [pd.Series([0, 1]), pd.Series([0.1, 0.9]), pd.Series(["A", "A"])]
    inputs[which].index = [1, 0]
    with pytest.raises(ValueError, match="indexes"):
        analyze_segments(*inputs)


@pytest.mark.parametrize(
    "segments", [["A"], [["A", "B"]], [1, "B"], [True, "B"], [[1], "B"], [np.inf, "B"]]
)
def test_invalid_segments(segments):
    with pytest.raises(ValueError):
        analyze_segments([0, 1], [0.1, 0.9], segments)


@pytest.mark.parametrize("minimum", [0, -1, True, 2.5, "50"])
def test_invalid_minimum(minimum):
    with pytest.raises(ValueError):
        analyze_segments([0, 1], [0.1, 0.9], ["A", "B"], min_segment_size=minimum)


@pytest.mark.parametrize(
    "labels,probabilities",
    [
        ([], []),
        ([0, 2], [0.1, 0.9]),
        ([0, 1], [np.nan, 0.9]),
        ([0, 1], [-1, 1]),
        ([0, 1], [0.1]),
    ],
)
def test_invalid_metric_inputs(labels, probabilities):
    with pytest.raises(ValueError):
        analyze_segments(labels, probabilities, ["A", "A"])
