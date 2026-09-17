"""Non-mutating data-quality measurements with explicit thresholds and denominators."""

import numpy as np
import pandas as pd

from modelgate.checks._common import (
    blocked,
    examples,
    identifier_examples,
    identifier_keys,
    missing_mask,
    numerical_values,
    prerequisite,
    record_keys,
    structure_problem,
    target_tokens,
)
from modelgate.checks.policy import DataCheckPolicy
from modelgate.results import CheckResult, Status


def _count_result(check_id, dataset, count, total, maximum, evidence, message):
    fraction = count / total
    return CheckResult(
        check_id=check_id,
        dataset=dataset,
        status=Status.FAIL if fraction > maximum else Status.PASS,
        evaluated=True,
        measurement={
            "count": count,
            "rows": total,
            "fraction": fraction,
            "percentage": fraction * 100,
        },
        threshold={"maximum_fraction": maximum, "comparison": ">"},
        evidence=evidence,
        message=message,
    )


def check_schema(
    frame: pd.DataFrame, policy: DataCheckPolicy, dataset: str, *, labeled: bool = True
) -> list[CheckResult]:
    """Check structural and semantic schema without fitting or changing any data."""
    problem = structure_problem(frame)
    results = [
        CheckResult(
            check_id="schema.structure",
            dataset=dataset,
            status=Status.FAIL if problem else Status.PASS,
            evaluated=True,
            measurement={
                "rows": len(frame),
                "columns": len(frame.columns),
                "duplicate_column_count": int(frame.columns.duplicated().sum()),
            },
            threshold={"minimum_rows": 1, "unique_nonblank_string_columns": True},
            evidence={
                "problem": problem,
                "duplicate_columns": sorted(
                    {
                        str(column)
                        for column in frame.columns[frame.columns.duplicated()]
                    }
                ),
                "invalid_column_positions": [
                    position
                    for position, column in enumerate(frame.columns)
                    if not isinstance(column, str) or not column.strip()
                ][: policy.max_examples],
            },
            message=problem or "Structure is usable.",
        )
    ]
    if problem:
        return results
    required = set(policy.required_columns)
    if not labeled:
        required.discard(policy.target_column)
    missing = sorted(required - set(frame.columns))
    extra = sorted(set(frame.columns) - set(policy.required_columns))
    status = (
        Status.FAIL
        if missing or (extra and policy.strict_extra_columns)
        else (Status.WARNING if extra else Status.PASS)
    )
    results.append(
        CheckResult(
            check_id="schema.columns",
            dataset=dataset,
            status=status,
            evaluated=True,
            measurement={"missing_count": len(missing), "unexpected_count": len(extra)},
            threshold={
                "maximum_missing": 0,
                "strict_extra_columns": policy.strict_extra_columns,
            },
            evidence={"missing_columns": missing, "unexpected_columns": extra},
            message="Compared supplied columns with configured requirements.",
        )
    )
    for column, kind in policy.column_types.items():
        check_id = f"schema.type.{column}"
        threshold = {"expected_type": kind, "maximum_invalid": 0}
        if column not in frame:
            results.append(
                blocked(check_id, dataset, f"Missing column: {column}", threshold)
            )
            continue
        series = frame[column]
        absent = missing_mask(series)
        if kind in ("number", "integer"):
            _, invalid = numerical_values(series, kind)
        else:
            # CSV-like scalar categories are represented by their literal string values.
            allowed = policy.allowed_categories.get(column)
            invalid = (
                ~absent & ~series.map(str).isin(allowed)
                if allowed
                else pd.Series(False, index=frame.index)
            )
            threshold["allowed_categories"] = list(allowed) if allowed else None
        results.append(
            CheckResult(
                check_id=check_id,
                dataset=dataset,
                status=Status.FAIL if invalid.any() else Status.PASS,
                evaluated=True,
                measurement={
                    "invalid_count": int(invalid.sum()),
                    "nonmissing_count": int((~absent).sum()),
                    "rows": len(frame),
                },
                threshold=threshold,
                evidence={
                    "column": column,
                    "examples": examples(series, invalid, policy.max_examples),
                },
                message="Missing values are handled separately from type validity.",
            )
        )
    target_threshold = {
        "labels": list(policy.target_labels),
        "positive_class_label": policy.positive_class_label,
        "required": labeled,
        "both_classes_required": dataset == "train",
    }
    target_problem = prerequisite(frame, [policy.target_column]) if labeled else None
    if not labeled:
        results.append(
            CheckResult(
                check_id="schema.target",
                dataset=dataset,
                status=Status.PASS,
                evaluated=True,
                measurement={
                    "required": False,
                    "present": policy.target_column in frame,
                },
                threshold=target_threshold,
                evidence={"content_validated": False},
                message="Declared unlabeled; target content is not assessed.",
            )
        )
    elif target_problem:
        results.append(
            blocked("schema.target", dataset, target_problem, target_threshold)
        )
    else:
        target = frame[policy.target_column]
        absent = missing_mask(target)
        tokens = target_tokens(frame, policy)
        labels = [str(label) for label in policy.target_labels]
        invalid = ~absent & ~tokens.isin(labels)
        counts = {label: int((tokens == label).sum()) for label in labels}
        both = all(counts.values())
        status = (
            Status.FAIL
            if absent.any() or invalid.any() or (dataset == "train" and not both)
            else Status.PASS
            if both
            else Status.WARNING
        )
        results.append(
            CheckResult(
                check_id="schema.target",
                dataset=dataset,
                status=status,
                evaluated=True,
                measurement={
                    "missing_count": int(absent.sum()),
                    "invalid_count": int(invalid.sum()),
                    "class_counts": counts,
                },
                threshold=target_threshold,
                evidence={
                    "both_classes_present": both,
                    "examples": examples(target, absent | invalid, policy.max_examples),
                },
                message="Checked binary labels; one-class data has metric limitations.",
            )
        )
    for check_id, columns in (
        ("schema.identifiers", policy.identifier_columns),
        ("schema.segment", (policy.segment_column,)),
    ):
        reason = prerequisite(frame, columns)
        if reason:
            results.append(blocked(check_id, dataset, reason, {"maximum_missing": 0}))
            continue
        masks = {column: missing_mask(frame[column]) for column in columns}
        missing_rows = pd.concat(list(masks.values()), axis=1).any(axis=1)
        results.append(
            _count_result(
                check_id,
                dataset,
                int(missing_rows.sum()),
                len(frame),
                0.0,
                {
                    "columns": list(columns),
                    "missing_by_column": {
                        name: int(mask.sum()) for name, mask in masks.items()
                    },
                    "row_positions": np.flatnonzero(missing_rows.to_numpy())[
                        : policy.max_examples
                    ].tolist(),
                },
                "Identifiers and the segment must be present on every row.",
            )
        )
    return results


def check_missingness(
    frame: pd.DataFrame, policy: DataCheckPolicy, dataset: str, *, labeled: bool = True
) -> list[CheckResult]:
    problem = structure_problem(frame)
    if problem:
        return [
            blocked(
                "missingness",
                dataset,
                problem,
                {"maximum_fraction": policy.max_missing_fraction},
            )
        ]
    columns = set(frame.columns) | set(policy.required_columns)
    if not labeled:
        columns.discard(policy.target_column)
    results = []
    for column in sorted(columns):
        maximum = policy.missing_overrides.get(column, policy.max_missing_fraction)
        if column not in frame:
            results.append(
                blocked(
                    f"missingness.{column}",
                    dataset,
                    f"Missing column: {column}",
                    {"maximum_fraction": maximum},
                )
            )
            continue
        count = int(missing_mask(frame[column]).sum())
        excluded = column in policy.missing_exclusions
        result = _count_result(
            f"missingness.{column}",
            dataset,
            count,
            len(frame),
            maximum,
            {"column": column, "excluded": excluded},
            "Excluded from missingness policy; measurement retained."
            if excluded
            else "Missing fraction compared with configured maximum.",
        )
        if excluded:
            result = CheckResult(
                **{
                    **result.model_dump(),
                    "status": Status.PASS,
                    "threshold": {"excluded": True, "maximum_fraction": None},
                }
            )
        results.append(result)
    return results


def check_duplicates(
    frame: pd.DataFrame, policy: DataCheckPolicy, dataset: str
) -> list[CheckResult]:
    results = []
    row_threshold = {
        "maximum_fraction": policy.max_duplicate_row_fraction,
        "comparison": ">",
    }
    problem = structure_problem(frame)
    if problem:
        results.append(blocked("duplicates.rows", dataset, problem, row_threshold))
    else:
        duplicate = frame.duplicated(keep="first")
        results.append(
            _count_result(
                "duplicates.rows",
                dataset,
                int(duplicate.sum()),
                len(frame),
                policy.max_duplicate_row_fraction,
                {
                    "columns": list(frame.columns),
                    "definition": "Occurrences after the first; supplied values",
                    "row_positions": np.flatnonzero(duplicate.to_numpy())[
                        : policy.max_examples
                    ].tolist(),
                },
                "Exact duplicates include all columns, including target and IDs.",
            )
        )
    reason = prerequisite(frame, policy.identifier_columns)
    threshold = {
        "maximum_fraction": policy.max_duplicate_id_fraction,
        "comparison": ">",
    }
    if reason:
        results.append(blocked("duplicates.identifiers", dataset, reason, threshold))
    else:
        keys = identifier_keys(frame, policy.identifier_columns)
        seen, repeated = set(), []
        for key in keys:
            if key is not None:
                if key in seen:
                    repeated.append(key)
                seen.add(key)
        excluded = keys.count(None)
        result = _count_result(
            "duplicates.identifiers",
            dataset,
            len(repeated),
            len(frame),
            policy.max_duplicate_id_fraction,
            {
                "identifier_columns": list(policy.identifier_columns),
                "excluded_missing_id_rows": excluded,
                "example_identifiers": identifier_examples(
                    repeated, policy.identifier_columns, policy.max_examples
                ),
            },
            "Duplicate complete identifier tuples, counted after the first occurrence.",
        )
        if excluded and result.status == Status.PASS:
            result = CheckResult(
                **{
                    **result.model_dump(),
                    "status": Status.WARNING,
                    "message": "Incomplete IDs excluded; measured only complete keys.",
                }
            )
        results.append(result)
    return results


def check_overlap(
    train: pd.DataFrame, reference: pd.DataFrame, policy: DataCheckPolicy
) -> list[CheckResult]:
    dataset = "train/reference"
    threshold = {
        "maximum_fraction": policy.max_identifier_overlap_fraction,
        "comparison": ">",
    }
    reason = prerequisite(train, policy.identifier_columns) or prerequisite(
        reference, policy.identifier_columns
    )
    results = []
    if reason:
        results.append(blocked("overlap.identifiers", dataset, reason, threshold))
    else:
        train_keys = identifier_keys(train, policy.identifier_columns)
        reference_keys = identifier_keys(reference, policy.identifier_columns)
        train_set = set(train_keys) - {None}
        matches = [
            key for key in reference_keys if key is not None and key in train_set
        ]
        excluded_train, excluded_reference = (
            train_keys.count(None),
            reference_keys.count(None),
        )
        fraction = len(matches) / len(reference)
        status = (
            Status.FAIL
            if fraction > policy.max_identifier_overlap_fraction
            else (
                Status.WARNING if excluded_train or excluded_reference else Status.PASS
            )
        )
        results.append(
            CheckResult(
                check_id="overlap.identifiers",
                dataset=dataset,
                status=status,
                evaluated=True,
                measurement={
                    "overlapping_reference_rows": len(matches),
                    "reference_rows": len(reference),
                    "fraction": fraction,
                    "percentage": fraction * 100,
                    "distinct_overlapping_keys": len(set(matches)),
                },
                threshold=threshold,
                evidence={
                    "identifier_columns": list(policy.identifier_columns),
                    "excluded_train_rows": excluded_train,
                    "excluded_reference_rows": excluded_reference,
                    "example_identifiers": identifier_examples(
                        matches, policy.identifier_columns, policy.max_examples
                    ),
                },
                message="Each reference row counts once, even with many train matches.",
            )
        )
    columns = list(policy.column_types)
    threshold = {"maximum_fraction": 0.0, "on_exceed": "WARNING"}
    reason = prerequisite(train, columns) or prerequisite(reference, columns)
    if not reason:
        for frame in (train, reference):
            for column, kind in policy.column_types.items():
                if (
                    kind != "category"
                    and numerical_values(frame[column], kind)[1].any()
                ):
                    reason = (
                        f"Invalid numeric values prevent record comparison: {column}"
                    )
                    break
            if reason:
                break
    if reason:
        results.append(blocked("overlap.records", dataset, reason, threshold))
    else:
        training_records = set(record_keys(train, policy.column_types))
        reference_records = record_keys(reference, policy.column_types)
        positions = [
            index
            for index, key in enumerate(reference_records)
            if key in training_records
        ]
        fraction = len(positions) / len(reference)
        results.append(
            CheckResult(
                check_id="overlap.records",
                dataset=dataset,
                status=Status.WARNING if positions else Status.PASS,
                evaluated=True,
                measurement={
                    "overlapping_reference_rows": len(positions),
                    "reference_rows": len(reference),
                    "fraction": fraction,
                    "percentage": fraction * 100,
                },
                threshold=threshold,
                evidence={
                    "compared_columns": columns,
                    "excluded_columns": [
                        policy.target_column,
                        *policy.identifier_columns,
                    ],
                    "row_positions": positions[: policy.max_examples],
                    "excluded_unmatchable_rows": 0,
                    "interpretation": (
                        "Identical predictors can belong to different customers; "
                        "not proof of contamination."
                    ),
                },
                message="Compared predictor tuples with numeric and null rules.",
            )
        )
    return results
