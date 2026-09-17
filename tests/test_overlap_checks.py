import pandas as pd

from modelgate.checks.data import check_duplicates, check_overlap
from modelgate.checks.policy import DataCheckPolicy


def by_id(results):
    return {result.check_id: result for result in results}


def test_identifier_overlap_uses_reference_rows_not_number_of_pairs(
    check_frame, check_policy
):
    train = pd.concat([check_frame, check_frame.iloc[[0]]], ignore_index=True)
    reference = check_frame.iloc[:3].copy()
    reference["id"] = ["T0", "T0", "NEW"]
    result = by_id(check_overlap(train, reference, check_policy))["overlap.identifiers"]
    assert result.status == "FAIL"
    assert result.measurement["overlapping_reference_rows"] == 2
    assert result.measurement["distinct_overlapping_keys"] == 1
    assert result.measurement["reference_rows"] == 3
    assert result.measurement["fraction"] == 2 / 3
    policy = DataCheckPolicy(
        **{**check_policy.model_dump(), "max_identifier_overlap_fraction": 2 / 3}
    )
    assert (
        by_id(check_overlap(train, reference, policy))["overlap.identifiers"].status
        == "PASS"
    )


def test_missing_ids_are_excluded_and_missing_columns_block(check_frame, check_policy):
    train = check_frame.copy()
    reference = check_frame.assign(id=["NEW0", "NEW1", "NEW2", "NEW3", "NEW4", "NEW5"])
    train.loc[0, "id"] = " "
    reference.loc[1, "id"] = None
    result = by_id(check_overlap(train, reference, check_policy))["overlap.identifiers"]
    assert result.status == "WARNING"
    assert result.measurement["fraction"] == 0
    assert (
        result.evidence["excluded_train_rows"]
        == result.evidence["excluded_reference_rows"]
        == 1
    )
    blocked = by_id(check_overlap(train.drop(columns="id"), reference, check_policy))
    assert blocked["overlap.identifiers"].evaluated is False
    assert blocked["overlap.records"].evaluated is True


def test_composite_keys_are_not_concatenated_and_preserve_leading_zeroes(
    check_frame, check_policy
):
    policy = DataCheckPolicy(
        **{
            **check_policy.model_dump(),
            "required_columns": (*check_policy.required_columns, "account"),
            "identifier_columns": ("id", "account"),
        }
    )
    train = check_frame.assign(account="c")
    train.loc[0, "id"] = "a|b"
    reference = check_frame.assign(id="a", account="b|c")
    assert (
        by_id(check_overlap(train, reference, policy))[
            "overlap.identifiers"
        ].measurement["fraction"]
        == 0
    )
    combined = pd.concat([train.iloc[[0]], reference.iloc[[0]]], ignore_index=True)
    assert (
        by_id(check_duplicates(combined, policy, "train"))[
            "duplicates.identifiers"
        ].measurement["count"]
        == 0
    )
    train.loc[0, "id"] = "001"
    reference.loc[0, ["id", "account"]] = ["1", "c"]
    assert (
        by_id(check_overlap(train, reference, policy))[
            "overlap.identifiers"
        ].measurement["fraction"]
        == 0
    )


def test_record_matches_exclude_ids_and_target_and_normalize_numbers(
    check_frame, check_policy
):
    reference = check_frame.assign(
        id=[f"OTHER-{i}" for i in range(6)], target=1 - check_frame.target
    )
    reference["amount"] = reference.amount.map(lambda value: f"{value:.2f}")
    reference.loc[0, "amount"] = " "
    train = check_frame.copy()
    train.loc[0, "amount"] = float("nan")
    results = by_id(check_overlap(train, reference, check_policy))
    assert results["overlap.identifiers"].status == "PASS"
    result = results["overlap.records"]
    assert result.status == "WARNING"
    assert result.measurement["overlapping_reference_rows"] == 6
    assert result.measurement["fraction"] == 1
    assert result.evidence["row_positions"] == [0, 1, 2, 3, 4]


def test_record_comparison_does_not_round_distinct_large_integers(
    check_frame, check_policy
):
    train = check_frame.iloc[[0]].assign(amount="9007199254740992")
    reference = train.assign(id="OTHER", amount="9007199254740993")
    result = by_id(check_overlap(train, reference, check_policy))["overlap.records"]
    assert result.status == "PASS"
    assert result.measurement["overlapping_reference_rows"] == 0


def test_invalid_numeric_record_values_block_only_record_comparison(
    check_frame, check_policy
):
    reference = check_frame.assign(amount="wrong")
    results = by_id(check_overlap(check_frame, reference, check_policy))
    assert results["overlap.identifiers"].status == "FAIL"
    assert results["overlap.records"].evaluated is False
    assert "amount" in results["overlap.records"].evidence["reason"]
