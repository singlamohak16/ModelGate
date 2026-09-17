import pytest
from pydantic import ValidationError

from modelgate.checks.policy import DataCheckPolicy


def test_telco_defaults_and_raw_label_variant():
    policy = DataCheckPolicy()
    assert policy.identifier_columns == ("customerID",)
    assert policy.column_types["TotalCharges"] == "number"
    assert policy.column_types["SeniorCitizen"] == "category"
    assert policy.target_labels == (0, 1)
    assert DataCheckPolicy(target_labels=("No", "Yes"), positive_class_label="Yes")


@pytest.mark.parametrize(
    "change",
    [
        {"max_missing_fraction": -0.1},
        {"max_missing_fraction": 1.1},
        {"max_missing_fraction": True},
        {"max_missing_fraction": float("nan")},
        {"max_duplicate_id_fraction": float("inf")},
        {"max_identifier_overlap_fraction": -1},
        {"max_examples": 21},
        {"max_examples": True},
        {"min_leakage_rows": 1},
        {"target_labels": (0, "0")},
        {"target_labels": (False, True)},
        {"positive_class_label": 3},
        {"segment_column": "Churn"},
        {"identifier_columns": ()},
        {"identifier_columns": ("customerID", "customerID")},
        {"missing_exclusions": ("typo",)},
        {"missing_overrides": {"typo": 0.1}},
        {"missing_exclusions": ("tenure",), "missing_overrides": {"tenure": 0.1}},
        {"allowed_categories": {"tenure": ("1",)}},
        {"allowed_categories": {"Contract": ()}},
        {"prohibited_columns": ("Churn",)},
        {"required_columns": ("Churn",)},
        {"column_types": {"Churn": "category", "Contract": "category"}},
        {"unexpected_policy_field": True},
    ],
)
def test_invalid_policy_fails_early(change):
    with pytest.raises(ValidationError):
        DataCheckPolicy(**change)
