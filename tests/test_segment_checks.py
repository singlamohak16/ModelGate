import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from modelgate.checks.segment import SegmentCheckPolicy, run_segment_checks
from modelgate.results import CheckResult


def select(report, rule, segment=None):
    return [
        row
        for row in report["checks"]
        if row["check_id"] == rule
        and (segment is None or row["evidence"].get("segment") == segment)
    ]


def test_exact_quality_boundaries_and_evidence():
    frame = pd.DataFrame({"Churn": [0, 1, 0, 1], "Contract": ["A"] * 2 + ["B"] * 2})
    policy = SegmentCheckPolicy(
        min_segment_size=2,
        min_precision=0.5,
        min_recall=1,
        min_f1=2 / 3,
        min_pr_auc=0.75,
    )
    report = run_segment_checks(frame, [0.5] * 4, policy)
    assert len(report["checks"]) == 17
    for row in report["checks"]:
        CheckResult.model_validate(row)
        assert row["status"] == "PASS"
        assert row["evaluated"] is True
    f1 = select(report, "segment.f1", "A")[0]
    assert f1["threshold"] == {"operator": ">=", "value": 2 / 3, "applied": True}
    assert f1["measurement"]["value"] == 2 / 3
    assert f1["evidence"]["confusion_matrix"] == [[0, 1], [0, 1]]
    assert f1["evidence"]["best_comparator"]["segments"] == ["A", "B"]
    json.dumps(report, allow_nan=False)


def test_overall_can_hide_segment_recall_failure():
    frame = pd.DataFrame(
        {"Churn": [0, 1] * 50, "Contract": ["large"] * 80 + ["weak"] * 20}
    )
    probabilities = [0.1, 0.9] * 40 + [0.1] * 20
    policy = SegmentCheckPolicy(min_segment_size=20, min_recall=0.75)
    report = run_segment_checks(frame, probabilities, policy)
    assert report["overall"]["recall"] == 0.8
    weak = select(report, "segment.recall", "weak")[0]
    assert weak["status"] == "FAIL"
    assert weak["measurement"]["value"] == 0
    assert weak["evidence"]["differences"]["from_overall"] == -0.8
    assert weak["evidence"]["differences"]["from_best"] == -1
    assert select(report, "segment.recall", "large")[0]["status"] == "PASS"
    precision = select(report, "segment.precision", "weak")[0]
    assert precision["status"] == "WARNING"
    assert not precision["evaluated"]


def test_small_segment_does_not_pass_or_fail_metric_policy():
    frame = pd.DataFrame({"Churn": [0, 1], "Contract": ["A", "A"]})
    report = run_segment_checks(frame, [0, 1], SegmentCheckPolicy(min_recall=1))
    size = select(report, "segment.sample_size")[0]
    assert size["status"] == "WARNING"
    assert size["evaluated"] is True
    assert size["measurement"]["value"] == 2
    assert size["threshold"]["value"] == 50
    for metric in ("precision", "recall", "f1", "pr_auc"):
        row = select(report, f"segment.{metric}")[0]
        assert row["status"] == "WARNING"
        assert row["measurement"]["value"] is None
        assert row["evaluated"] is False
        assert "min_segment_size" in row["message"]


def test_missing_segments_fail_and_do_not_change_overall_denominator():
    frame = pd.DataFrame({"Churn": [0, 1, 0, 1], "Contract": [None, " ", "A", "A"]})
    report = run_segment_checks(
        frame, [0.1, 0.9, 0.1, 0.9], SegmentCheckPolicy(min_segment_size=2)
    )
    row = select(report, "segment.completeness")[0]
    assert row["status"] == "FAIL"
    assert row["measurement"] == {"count": 2, "fraction": 0.5}
    assert row["evidence"]["total_rows"] == 4
    assert row["evidence"]["named_rows"] == 2
    assert report["overall"]["sample_size"] == 4
    assert report["coverage"]["eligible_fraction"] == 0.5


def test_all_missing_still_has_checks_and_no_best():
    frame = pd.DataFrame({"Churn": [0, 1], "Contract": [None, pd.NA]})
    report = run_segment_checks(frame, [0, 1], SegmentCheckPolicy())
    assert report["segments"] == []
    assert len(report["checks"]) == 5
    assert report["checks"][0]["status"] == "FAIL"
    assert all(row["status"] == "WARNING" for row in report["checks"][1:])
    assert report["overall"]["recall"] == 1


def test_single_eligible_segment_warns_despite_zero_self_gap():
    frame = pd.DataFrame({"Churn": [0, 1], "Contract": ["A", "A"]})
    report = run_segment_checks(frame, [0, 1], SegmentCheckPolicy(min_segment_size=2))
    for metric in ("precision", "recall", "f1", "pr_auc"):
        row = select(report, f"segment.comparison_support.{metric}")[0]
        assert row["status"] == "WARNING"
        assert row["measurement"]["candidate_count"] == 1
        assert report["segments"][0]["differences"][metric]["from_best"] == 0
        measurement = select(report, f"segment.{metric}")[0]
        assert measurement["status"] == "PASS"
        assert not measurement["threshold"]["applied"]
        assert "Descriptive" in measurement["message"]


def test_one_class_group_and_whole_sample():
    frame = pd.DataFrame({"Churn": [0, 0], "Contract": ["A", "A"]})
    report = run_segment_checks(
        frame, [0.1, 0.9], SegmentCheckPolicy(min_segment_size=2)
    )
    assert select(report, "segment.class_support")[0]["status"] == "WARNING"
    assert select(report, "segment.recall")[0]["measurement"]["value"] is None
    assert select(report, "segment.pr_auc")[0]["measurement"]["value"] is None
    assert select(report, "segment.precision")[0]["measurement"]["value"] == 0


def test_all_minimum_rules_fail_without_rounding():
    frame = pd.DataFrame({"Churn": [0, 1], "Contract": ["A", "A"]})
    policy = SegmentCheckPolicy(
        min_segment_size=2,
        min_precision=np.nextafter(0.5, 1),
        min_recall=1,
        min_f1=0.9,
        min_pr_auc=0.9,
    )
    report = run_segment_checks(frame, [0.5, 0.5], policy)
    assert select(report, "segment.precision")[0]["status"] == "FAIL"
    assert select(report, "segment.recall")[0]["status"] == "PASS"
    assert select(report, "segment.f1")[0]["status"] == "FAIL"
    assert select(report, "segment.pr_auc")[0]["status"] == "FAIL"


def test_custom_columns_labels_and_no_frame_mutation():
    frame = pd.DataFrame({"target": ["Yes", "No"], "group": ["A", "A"]}, index=[5, 5])
    original = frame.copy(deep=True)
    policy = SegmentCheckPolicy(
        target_column="target",
        segment_column="group",
        target_labels=("Yes", "No"),
        positive_class_label="No",
        min_segment_size=2,
    )
    report = run_segment_checks(frame, [0.1, 0.9], policy, dataset="test")
    assert report["overall"]["recall"] == 1
    assert all(row["dataset"] == "test" for row in report["checks"])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_segment_size": 0},
        {"min_segment_size": True},
        {"min_segment_size": 2.5},
        {"min_recall": -0.1},
        {"min_f1": 1.1},
        {"min_precision": True},
        {"min_pr_auc": float("nan")},
        {"decision_threshold": float("inf")},
        {"target_column": " "},
        {"segment_column": "Churn"},
        {"segment_column": 3},
        {"target_labels": [0, "0"]},
        {"positive_class_label": 2},
        {"unknown": 1},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValidationError):
        SegmentCheckPolicy(**kwargs)


@pytest.mark.parametrize(
    "frame",
    [
        None,
        pd.DataFrame({"Churn": [0, 1]}),
        pd.DataFrame([[0, "A"]], columns=["Churn", "Churn"]),
        pd.DataFrame(columns=["Churn", "Contract"]),
    ],
)
def test_invalid_frames(frame):
    with pytest.raises(ValueError):
        run_segment_checks(frame, [0.1, 0.9], SegmentCheckPolicy())


def test_probability_series_must_match_frame_index():
    frame = pd.DataFrame({"Churn": [0, 1], "Contract": ["A", "A"]})
    with pytest.raises(ValueError, match="indexes"):
        run_segment_checks(
            frame, pd.Series([0.1, 0.9], index=[1, 0]), SegmentCheckPolicy()
        )
