import json

import numpy as np
import pandas as pd
import pytest

from modelgate.checks import DataCheckPolicy, run_data_checks
from modelgate.checks.data import check_duplicates, check_missingness, check_schema


def by_id(results):
    return {result.check_id: result for result in results}


def changed(policy, **updates):
    return DataCheckPolicy(**{**policy.model_dump(), **updates})


def test_schema_checks_numeric_csv_text_and_allowed_categories(
    check_frame, check_policy
):
    check_frame["amount"] = check_frame.amount.astype(str)
    check_frame["visits"] = check_frame.visits.astype(str)
    results = check_schema(check_frame, check_policy, "train")
    assert all(result.status == "PASS" for result in results)
    check_frame.loc[0, "amount"] = "inf"
    check_frame.loc[1, "visits"] = "1.5"
    check_frame.loc[2, "flag"] = "C"
    results = by_id(check_schema(check_frame, check_policy, "train"))
    for column, row in (("amount", 0), ("visits", 1), ("flag", 2)):
        result = results[f"schema.type.{column}"]
        assert result.status == "FAIL"
        assert result.measurement["invalid_count"] == 1
        assert result.evidence["examples"][0]["row_position"] == row


def test_numeric_boolean_and_nan_text_are_invalid(check_frame, check_policy):
    check_frame["amount"] = check_frame.amount.astype(object)
    check_frame.loc[0, "amount"] = True
    check_frame.loc[1, "amount"] = "NaN"
    result = by_id(check_schema(check_frame, check_policy, "train"))[
        "schema.type.amount"
    ]
    assert result.measurement["invalid_count"] == 2
    assert result.status == "FAIL"


def test_missing_extra_columns_and_independent_checks(check_frame, check_policy):
    damaged = check_frame.drop(columns="amount").assign(surprise="x")
    results = by_id(check_schema(damaged, check_policy, "train"))
    assert results["schema.columns"].status == "FAIL"
    assert results["schema.columns"].evidence == {
        "missing_columns": ["amount"],
        "unexpected_columns": ["surprise"],
    }
    assert results["schema.type.amount"].evaluated is False
    assert results["schema.type.visits"].status == "PASS"
    extra = check_frame.assign(surprise="x")
    assert (
        by_id(check_schema(extra, check_policy, "train"))["schema.columns"].status
        == "WARNING"
    )
    strict = changed(check_policy, strict_extra_columns=True)
    assert (
        by_id(check_schema(extra, strict, "train"))["schema.columns"].status == "FAIL"
    )


def test_target_missing_invalid_and_explicit_encoding(check_frame, check_policy):
    check_frame["target"] = ["No", "Yes"] * 3
    assert (
        by_id(check_schema(check_frame, check_policy, "train"))["schema.target"].status
        == "FAIL"
    )
    raw_policy = changed(
        check_policy, target_labels=("No", "Yes"), positive_class_label="Yes"
    )
    assert (
        by_id(check_schema(check_frame, raw_policy, "train"))["schema.target"].status
        == "PASS"
    )
    check_frame.loc[0, "target"] = " "
    check_frame.loc[1, "target"] = "maybe"
    result = by_id(check_schema(check_frame, raw_policy, "train"))["schema.target"]
    assert (
        result.measurement["missing_count"] == result.measurement["invalid_count"] == 1
    )
    assert result.status == "FAIL"


def test_encoded_csv_labels_and_booleans(check_frame, check_policy):
    check_frame["target"] = check_frame.target.astype(str)
    assert (
        by_id(check_schema(check_frame, check_policy, "train"))["schema.target"].status
        == "PASS"
    )
    check_frame["target"] = [False, True] * 3
    assert (
        by_id(check_schema(check_frame, check_policy, "train"))[
            "schema.target"
        ].measurement["invalid_count"]
        == 6
    )


def test_single_class_train_fails_reference_warns(check_frame, check_policy):
    frame = check_frame.assign(target=0)
    train = by_id(check_schema(frame, check_policy, "train"))["schema.target"]
    reference = by_id(check_schema(frame, check_policy, "reference"))["schema.target"]
    assert train.status == "FAIL"
    assert reference.status == "WARNING"
    assert reference.measurement["class_counts"] == {"0": 6, "1": 0}


def test_missingness_normalization_boundaries_and_overrides(check_frame, check_policy):
    check_frame["amount"] = [None, np.nan, " ", "", "NA", "2"]
    policy = changed(check_policy, missing_overrides={"amount": 4 / 6})
    result = by_id(check_missingness(check_frame, policy, "train"))[
        "missingness.amount"
    ]
    assert result.status == "PASS"
    assert result.measurement == {
        "count": 4,
        "rows": 6,
        "fraction": 4 / 6,
        "percentage": pytest.approx(400 / 6),
    }
    policy = changed(check_policy, missing_overrides={"amount": 0.5})
    assert (
        by_id(check_missingness(check_frame, policy, "train"))[
            "missingness.amount"
        ].status
        == "FAIL"
    )


def test_exclusion_does_not_disable_required_target(check_frame, check_policy):
    check_frame["target"] = None
    policy = changed(check_policy, missing_exclusions=("target",))
    result = by_id(check_missingness(check_frame, policy, "train"))[
        "missingness.target"
    ]
    assert result.status == "PASS"
    assert result.measurement["count"] == 6
    assert result.threshold["excluded"] is True
    assert (
        by_id(check_schema(check_frame, policy, "train"))["schema.target"].status
        == "FAIL"
    )


def test_ids_and_segment_missing_measurements(check_frame, check_policy):
    check_frame.loc[0, "id"] = " "
    check_frame.loc[1, "group"] = None
    results = by_id(check_schema(check_frame, check_policy, "train"))
    assert results["schema.identifiers"].measurement["count"] == 1
    assert results["schema.segment"].evidence["row_positions"] == [1]
    assert results["schema.segment"].status == "FAIL"


def test_duplicates_count_occurrences_after_first(check_frame, check_policy):
    frame = pd.concat(
        [check_frame, check_frame.iloc[[0]], check_frame.iloc[[0]]], ignore_index=True
    )
    results = by_id(check_duplicates(frame, check_policy, "train"))
    for name in ("duplicates.rows", "duplicates.identifiers"):
        assert results[name].measurement["count"] == 2
        assert results[name].measurement["fraction"] == 0.25
        assert results[name].status == "FAIL"
    assert results["duplicates.identifiers"].evidence["example_identifiers"] == [
        {"id": "T0"}
    ]
    policy = changed(
        check_policy, max_duplicate_row_fraction=0.25, max_duplicate_id_fraction=0.25
    )
    assert all(
        result.status == "PASS" for result in check_duplicates(frame, policy, "train")
    )


def test_incomplete_ids_do_not_match_each_other(check_frame, check_policy):
    check_frame.loc[[0, 1], "id"] = None
    result = by_id(check_duplicates(check_frame, check_policy, "train"))[
        "duplicates.identifiers"
    ]
    assert result.measurement["count"] == 0
    assert result.evidence["excluded_missing_id_rows"] == 2
    assert result.status == "WARNING"


@pytest.mark.parametrize(
    "damage", ["empty", "duplicate_columns", "nonstring_column", "no_columns"]
)
def test_unusable_structure_is_failure_not_a_false_pass(
    check_frame, check_policy, damage
):
    if damage == "empty":
        frame = check_frame.iloc[:0]
    elif damage == "no_columns":
        frame = check_frame.iloc[:, :0]
    elif damage == "duplicate_columns":
        frame = pd.concat([check_frame, check_frame[["amount"]]], axis=1)
    else:
        frame = check_frame.rename(columns={"amount": 123})
    results = run_data_checks(frame, check_frame, check_policy)
    structure = next(
        result
        for result in results
        if result.dataset == "train" and result.check_id == "schema.structure"
    )
    assert structure.status == "FAIL"
    overlap = next(
        result for result in results if result.check_id == "overlap.identifiers"
    )
    assert not overlap.evaluated
    assert overlap.measurement == {"value": None}
    json.dumps([result.model_dump(mode="json") for result in results], allow_nan=False)


def test_runner_is_deterministic_nonmutating_and_handles_unlabeled_current(
    check_frame, check_policy
):
    original = check_frame.copy(deep=True)
    reference = check_frame.assign(
        id=[f"R{i}" for i in range(6)], amount=100 + check_frame.amount
    )
    current = reference.drop(columns="target")
    results = run_data_checks(check_frame, reference, check_policy, current=current)
    assert results == run_data_checks(
        check_frame, reference, check_policy, current=current
    )
    pd.testing.assert_frame_equal(check_frame, original)
    current_results = by_id(
        [result for result in results if result.dataset == "current"]
    )
    assert current_results["schema.target"].measurement == {
        "required": False,
        "present": False,
    }
    assert "missingness.target" not in current_results
    assert current_results["leakage.target_like"].evaluated is False
    assert not any(result.status == "FAIL" for result in current_results.values())
    assert len({(result.dataset, result.check_id) for result in results}) == len(
        results
    )


def test_examples_are_bounded_and_row_positions_ignore_index_labels(
    check_frame, check_policy
):
    frame = check_frame.assign(amount="invalid")
    frame.index = [99] * len(frame)
    policy = changed(check_policy, max_examples=2)
    result = by_id(check_schema(frame, policy, "train"))["schema.type.amount"]
    assert result.measurement["invalid_count"] == 6
    assert result.evidence["examples"] == [
        {"row_position": 0, "value": "invalid"},
        {"row_position": 1, "value": "invalid"},
    ]
    policy = changed(check_policy, max_examples=0)
    assert (
        by_id(check_schema(frame, policy, "train"))["schema.type.amount"].evidence[
            "examples"
        ]
        == []
    )
