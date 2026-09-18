import json

import numpy as np
import pytest
from pydantic import ValidationError

from modelgate.checks.model import ModelCheckPolicy, run_model_checks
from modelgate.predictions import positive_probabilities
from modelgate.results import CheckResult


def test_policy_boundaries_and_serialization():
    report = run_model_checks(
        [0, 0, 1, 1],
        [0.5, 0.5, 0.5, 0.5],
        ModelCheckPolicy(
            min_precision=0.5,
            min_recall=1,
            min_f1=2 / 3,
            min_pr_auc=0.75,
            max_brier_score=0.25,
        ),
    )
    for result in report["checks"]:
        CheckResult.model_validate(result)
        assert result["status"] == "PASS"
        assert result["evidence"]["sample_size"] == 4
    json.dumps(report, allow_nan=False)
    failed = run_model_checks(
        [0, 1],
        [0.8, 0.2],
        ModelCheckPolicy(
            min_precision=0.1,
            min_recall=0.1,
            min_f1=0.1,
            min_pr_auc=1,
            max_brier_score=0.1,
        ),
    )
    assert [r["status"] for r in failed["checks"]] == ["FAIL"] * 5 + ["PASS"]


def test_unconfigured_and_undefined():
    report = run_model_checks([0, 1], [0.1, 0.2], ModelCheckPolicy())
    precision = report["checks"][0]
    assert precision["status"] == "WARNING"
    assert precision["evaluated"] is False
    assert precision["measurement"]["value"] is None
    for row in report["checks"][:5]:
        assert row["threshold"]["applied"] is False
    single = run_model_checks([0, 0], [0.1, 0.9], ModelCheckPolicy())
    assert single["checks"][-1]["status"] == "WARNING"
    assert single["checks"][3]["evaluated"] is False
    assert single["checks"][4]["evaluated"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_f1": True},
        {"min_recall": -0.1},
        {"max_brier_score": float("nan")},
        {"decision_threshold": 1.1},
        {"thresholds": []},
        {"thresholds": [0.5, 0.5]},
        {"calibration_bins": True},
        {"calibration_bins": 1},
        {"unexpected": 1},
        {"target_labels": [0, "0"]},
        {"positive_class_label": 2},
        {"target_labels": [False, True]},
    ],
)
def test_invalid_policy(kwargs):
    with pytest.raises(ValidationError):
        ModelCheckPolicy(**kwargs)


class FakeModel:
    classes_ = np.array([1, 0])

    def __init__(self, output=None):
        self.output = [[0.8, 0.2], [0.1, 0.9]] if output is None else output

    def predict_proba(self, features):
        return self.output


def test_reversed_classes():
    model = FakeModel()
    np.testing.assert_array_equal(positive_probabilities(model, [0, 0]), [0.8, 0.1])
    model.classes_ = np.array(["Yes", "No"])
    np.testing.assert_array_equal(
        positive_probabilities(
            model, [0, 0], target_labels=("No", "Yes"), positive_label="No"
        ),
        [0.2, 0.9],
    )


@pytest.mark.parametrize(
    "matrix",
    [
        [0.1, 0.9],
        [[0.1, 0.2]],
        [[0.1, 0.2], [0.1, 0.9]],
        [[np.nan, 0], [0, 1]],
        [[-0.1, 1.1], [0, 1]],
        [[True, False], [False, True]],
        [[False, 1.0], [0.5, 0.5]],
        [["0", "1"], ["1", "0"]],
    ],
)
def test_invalid_model_output(matrix):
    with pytest.raises(ValueError):
        positive_probabilities(FakeModel(matrix), [0, 0])


@pytest.mark.parametrize(
    "classes", [None, [0], [0, 1, 2], [0, 0], [False, True], [0, 2]]
)
def test_invalid_model_classes(classes):
    model = FakeModel()
    model.classes_ = classes
    with pytest.raises(ValueError):
        positive_probabilities(model, [0, 0])


def test_missing_predict_proba_and_empty_features():
    model = FakeModel()
    model.predict_proba = None
    with pytest.raises(ValueError, match="predict_proba"):
        positive_probabilities(model, [0, 0])
    with pytest.raises(ValueError, match="nonempty"):
        positive_probabilities(FakeModel(), [])
