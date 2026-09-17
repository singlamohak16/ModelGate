import pytest

from modelgate.checks.leakage import check_leakage
from modelgate.checks.policy import DataCheckPolicy


def by_id(results):
    return {result.check_id: result for result in results}


@pytest.mark.parametrize(
    "encoding,relationship",
    [
        ((0, 1), "exact_copy"),
        ((1, 0), "inverted_labels"),
        ((7, 99), "two_value_recoding"),
        (("stay", "leave"), "two_value_recoding"),
        ((False, True), "two_value_recoding"),
    ],
)
def test_copies_inversions_and_recodings_warn(
    check_frame, check_policy, encoding, relationship
):
    frame = check_frame.assign(
        suspicious=check_frame.target.map(dict(enumerate(encoding)))
    )
    result = by_id(check_leakage(frame, check_policy, "train"))[
        "leakage.target_like.suspicious"
    ]
    assert result.status == "WARNING"
    assert result.evaluated is True
    assert result.measurement["target_like"] is True
    assert result.measurement["comparable_rows"] == 6
    assert result.measurement["coverage"] == 1
    assert result.evidence["relationship"] == relationship
    assert len(result.evidence["mapping_examples"]) == 2


def test_prohibition_fails_independently_of_target_validity(check_frame, check_policy):
    policy = DataCheckPolicy(
        **{**check_policy.model_dump(), "prohibited_columns": ("post_churn",)}
    )
    frame = check_frame.drop(columns="target").assign(post_churn="available")
    results = by_id(check_leakage(frame, policy, "train"))
    prohibited = results["leakage.prohibited_columns"]
    assert prohibited.status == "FAIL"
    assert prohibited.measurement["prohibited_column_count"] == 1
    assert prohibited.evidence["present_columns"] == ["post_churn"]
    assert results["leakage.target_like"].evaluated is False


def test_ordinary_features_and_unique_values_are_not_target_recodings(
    check_frame, check_policy
):
    results = check_leakage(check_frame, check_policy, "train")
    assert all(result.status == "PASS" for result in results)
    assert not any(
        result.check_id.endswith(".id") or result.check_id.endswith(".target")
        for result in results
    )
    frame = check_frame.assign(unique_feature=[f"unique-{i}" for i in range(6)])
    result = by_id(check_leakage(frame, check_policy, "train"))[
        "leakage.target_like.unique_feature"
    ]
    assert result.measurement["distinct_feature_values"] == 6
    assert result.status == "PASS"


def test_partial_coverage_is_reported_and_requires_enough_evidence(
    check_frame, check_policy
):
    frame = check_frame.assign(suspicious=check_frame.target.astype(object))
    frame.loc[0, "suspicious"] = None
    result = by_id(check_leakage(frame, check_policy, "train"))[
        "leakage.target_like.suspicious"
    ]
    assert result.evaluated is True
    assert result.status == "WARNING"
    assert result.measurement["comparable_rows"] == 5
    assert result.measurement["coverage"] == 5 / 6
    frame.loc[1, "suspicious"] = " "
    result = by_id(check_leakage(frame, check_policy, "train"))[
        "leakage.target_like.suspicious"
    ]
    assert result.evaluated is False
    assert result.measurement["coverage"] == 4 / 6


def test_minimum_sample_count_and_class_presence(check_frame, check_policy):
    small = check_frame.iloc[:2].assign(copy=check_frame.target.iloc[:2])
    assert (
        by_id(check_leakage(small, check_policy, "train"))[
            "leakage.target_like.copy"
        ].evaluated
        is False
    )
    one_class = check_frame.assign(target=0)
    result = by_id(check_leakage(one_class, check_policy, "reference"))[
        "leakage.target_like"
    ]
    assert result.status == "WARNING"
    assert "Both target classes" in result.evidence["reason"]


def test_small_comparable_subset_with_one_class_is_blocked(check_frame, check_policy):
    frame = check_frame.assign(copy=check_frame.target.astype(object))
    frame.loc[frame.target == 0, "copy"] = None
    policy = DataCheckPolicy(
        **{
            **check_policy.model_dump(),
            "min_leakage_rows": 2,
            "min_leakage_coverage": 0.5,
        }
    )
    result = by_id(check_leakage(frame, policy, "reference"))[
        "leakage.target_like.copy"
    ]
    assert result.evaluated is False
    assert "lack both" in result.evidence["reason"]


def test_mapping_examples_can_be_suppressed(check_frame, check_policy):
    policy = DataCheckPolicy(**{**check_policy.model_dump(), "max_examples": 0})
    result = by_id(
        check_leakage(check_frame.assign(copy=check_frame.target), policy, "train")
    )["leakage.target_like.copy"]
    assert result.measurement["target_like"] is True
    assert result.evidence["mapping_examples"] == []
