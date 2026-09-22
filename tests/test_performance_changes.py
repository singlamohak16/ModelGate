import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from modelgate.checks.performance import PerformanceChangePolicy, run_performance_checks
from modelgate.results import CheckResult


def data(labels=(0, 1, 0, 1)):
    return pd.DataFrame({"Churn": labels})


def compare(p_ref, p_cur, **limits):
    return run_performance_checks(
        data(),
        data(),
        p_ref,
        p_cur,
        PerformanceChangePolicy(min_sample_size=4, **limits),
        current_labeled=True,
    )


def test_identical_measurements_zero_change_equality():
    r = compare(
        [0.1, 0.9, 0.2, 0.8],
        [0.1, 0.9, 0.2, 0.8],
        max_precision_drop=0,
        max_recall_drop=0,
        max_f1_drop=0,
        max_pr_auc_drop=0,
        max_brier_increase=0,
    )
    assert r["comparison_available"]
    assert len(r["checks"]) == 7
    for check in r["checks"]:
        CheckResult.model_validate(check)
        assert check["status"] == "PASS"
    for check in r["checks"][:5]:
        assert check["measurement"]["current_minus_reference"] == 0
        assert check["measurement"]["deterioration"] == 0
        assert check["threshold"]["applied"]
    json.dumps(r, allow_nan=False)


def test_known_degradation_and_brier_direction():
    r = compare(
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        max_precision_drop=0.1,
        max_recall_drop=0.1,
        max_f1_drop=0.1,
        max_pr_auc_drop=0.1,
        max_brier_increase=0.1,
    )
    assert all(c["status"] == "FAIL" for c in r["checks"][:5])
    for row in r["checks"][:3]:
        assert row["measurement"]["current_minus_reference"] == -1
        assert row["measurement"]["deterioration"] == 1
    brier = r["checks"][4]
    assert brier["measurement"] == {
        "reference": 0,
        "current": 1,
        "current_minus_reference": 1,
        "deterioration": 1,
    }
    assert brier["evidence"]["current"]["confusion_matrix"] == [[0, 2], [2, 0]]


def test_improvement_not_degradation():
    r = compare(
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        max_precision_drop=0,
        max_recall_drop=0,
        max_f1_drop=0,
        max_pr_auc_drop=0,
        max_brier_increase=0,
    )
    assert all(c["status"] == "PASS" for c in r["checks"])
    assert all(c["measurement"]["deterioration"] < 0 for c in r["checks"][:5])


def test_exact_nonzero_boundary_and_strict_exceedance():
    # Recall drops from 1 to 0.5; probabilities at 0.5 count as positives.
    kwargs = ([0, 1, 0, 1], [0, 0.5, 0, 0.1])
    assert compare(*kwargs, max_recall_drop=0.5)["checks"][1]["status"] == "PASS"
    assert (
        compare(*kwargs, max_recall_drop=np.nextafter(0.5, 0))["checks"][1]["status"]
        == "FAIL"
    )


@pytest.mark.parametrize(
    "current", [data(), pd.DataFrame({"Churn": ["bad"]}), pd.DataFrame({"x": [1]})]
)
def test_unlabeled_never_reads_targets_or_claims_metrics(current):
    r = run_performance_checks(None, current, None, None, PerformanceChangePolicy())
    assert not r["comparison_available"]
    assert r["reference"] is None and r["current"] is None and r["checks"] == []
    assert "not declared" in r["reason"]


def test_current_absent_is_unavailable_even_when_labels_requested():
    r = run_performance_checks(
        data(),
        None,
        [0, 1, 0, 1],
        None,
        PerformanceChangePolicy(),
        current_labeled=True,
    )
    assert r["checks"] == [] and not r["comparison_available"]
    assert "not supplied" in r["reason"]


@pytest.mark.parametrize(
    "current,p",
    [
        (pd.DataFrame({"x": [0, 1]}), [0, 1]),
        (data([0, 2]), [0, 1]),
        (data([0, None]), [0, 1]),
        (data(), [np.nan] * 4),
        (data(), [0.1]),
        (data(), pd.Series([0.1] * 4, index=[1, 2, 3, 4])),
    ],
)
def test_declared_bad_labels_or_predictions_block_performance(current, p):
    r = run_performance_checks(
        data(),
        current,
        [0, 1, 0, 1],
        p,
        PerformanceChangePolicy(min_sample_size=1),
        current_labeled=True,
    )
    assert not r["comparison_available"]
    assert "current:" in r["reason"]
    assert all(c["status"] == "WARNING" and not c["evaluated"] for c in r["checks"][:5])
    assert all(c["measurement"]["deterioration"] is None for c in r["checks"][:5])


def test_one_class_preserves_brier_but_warns_and_blocks_pr_auc():
    r = run_performance_checks(
        data(),
        data([0, 0]),
        [0, 1, 0, 1],
        [0.1, 0.2],
        PerformanceChangePolicy(min_sample_size=2),
        current_labeled=True,
    )
    assert r["comparison_available"]
    assert r["checks"][0]["measurement"]["current"] is None
    assert not r["checks"][3]["evaluated"]
    assert r["checks"][4]["evaluated"]
    assert r["checks"][-1]["status"] == "WARNING"


def test_size_49_50():
    for n, expected in ((49, False), (50, True)):
        ref, cur = data([0, 1] * 25), data(([0, 1] * 25)[:n])
        r = run_performance_checks(
            ref,
            cur,
            [0.5] * 50,
            [0.5] * n,
            PerformanceChangePolicy(),
            current_labeled=True,
        )
        assert r["comparison_available"] is expected
        assert r["checks"][1]["evaluated"] is expected


def test_custom_labels_direction_unequal_sizes_and_no_mutation():
    ref = pd.DataFrame({"target": ["No", "Yes"]}, index=[2, 2])
    cur = pd.DataFrame({"target": ["No", "Yes", "No"]})
    original = ref.copy(deep=True)
    policy = PerformanceChangePolicy(
        target_column="target",
        target_labels=("No", "Yes"),
        positive_class_label="No",
        min_sample_size=2,
    )
    r = run_performance_checks(
        ref, cur, [0.9, 0.1], [0.9, 0.1, 0.8], policy, current_labeled=True
    )
    assert r["reference"]["recall"] == r["current"]["recall"] == 1
    assert r["checks"][1]["threshold"]["applied"] is False
    pd.testing.assert_frame_equal(ref, original)


@pytest.mark.parametrize("flag", [1, "true", None])
def test_label_flag_must_be_boolean(flag):
    with pytest.raises(ValueError):
        run_performance_checks(
            data(),
            data(),
            [0.5] * 4,
            [0.5] * 4,
            PerformanceChangePolicy(),
            current_labeled=flag,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_brier_increase": -0.1},
        {"max_f1_drop": True},
        {"max_recall_drop": np.nan},
        {"max_precision_drop": 1.1},
        {"min_sample_size": 0},
        {"min_sample_size": True},
        {"target_column": " "},
        {"positive_class_label": 2},
        {"target_labels": [0, "0"]},
        {"decision_threshold": np.inf},
        {"unexpected": 1},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValidationError):
        PerformanceChangePolicy(**kwargs)
