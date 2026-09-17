"""Limited leakage heuristics: configured prohibitions and exact two-value recodings."""

import numpy as np
import pandas as pd

from modelgate.checks._common import (
    blocked,
    label_token,
    missing_mask,
    prerequisite,
    structure_problem,
    target_tokens,
)
from modelgate.checks.policy import DataCheckPolicy
from modelgate.results import CheckResult, Status


def check_leakage(
    frame: pd.DataFrame, policy: DataCheckPolicy, dataset: str, *, labeled: bool = True
) -> list[CheckResult]:
    prohibited = sorted(set(frame.columns) & set(policy.prohibited_columns))
    results = [
        CheckResult(
            check_id="leakage.prohibited_columns",
            dataset=dataset,
            status=Status.FAIL if prohibited else Status.PASS,
            evaluated=True,
            measurement={"prohibited_column_count": len(prohibited)},
            threshold={
                "maximum_count": 0,
                "prohibited_columns": list(policy.prohibited_columns),
            },
            evidence={"present_columns": prohibited},
            message="Prohibited feature violates policy; this does not prove leakage.",
        )
    ]
    threshold = {
        "minimum_comparable_rows": policy.min_leakage_rows,
        "minimum_coverage": policy.min_leakage_coverage,
        "relationship": "exact bijection between two feature values and two labels",
    }
    reason = structure_problem(frame) or prerequisite(frame, [policy.target_column])
    if not labeled:
        reason = "Dataset explicitly declared unlabeled."
    if not reason:
        tokens = target_tokens(frame, policy)
        labels = {str(label) for label in policy.target_labels}
        if (
            missing_mask(frame[policy.target_column]).any()
            or not tokens.isin(labels).all()
        ):
            reason = "Target contains missing or invalid labels."
        elif tokens.nunique() != 2:
            reason = "Both target classes are needed for the recoding heuristic."
    if reason:
        results.append(blocked("leakage.target_like", dataset, reason, threshold))
        return results
    ignored = {policy.target_column, *policy.identifier_columns}
    for column in sorted(set(frame.columns) - ignored):
        check_id = f"leakage.target_like.{column}"
        comparable = ~missing_mask(frame[column])
        count = int(comparable.sum())
        coverage = count / len(frame)
        measurement = {
            "comparable_rows": count,
            "rows": len(frame),
            "coverage": coverage,
        }
        if count < policy.min_leakage_rows or coverage < policy.min_leakage_coverage:
            results.append(
                blocked(
                    check_id,
                    dataset,
                    "Insufficient comparable rows or coverage.",
                    threshold,
                    measurement,
                )
            )
            continue
        feature = frame.loc[comparable, column].map(
            lambda value: (
                str(value)
                if isinstance(value, (bool, np.bool_))
                else label_token(value)
            )
        )
        target = tokens.loc[comparable]
        if target.nunique() != 2:
            results.append(
                blocked(
                    check_id,
                    dataset,
                    "Comparable rows lack both target classes.",
                    threshold,
                    measurement,
                )
            )
            continue
        pairs = set(zip(feature, target, strict=True))
        distinct = feature.nunique(dropna=False)
        suspect = distinct == 2 and len(pairs) == 2 and feature.notna().all()
        relationship = None
        if suspect:
            relationship = (
                "exact_copy"
                if all(left == right for left, right in pairs)
                else (
                    "inverted_labels"
                    if set(feature) == labels
                    else "two_value_recoding"
                )
            )
        results.append(
            CheckResult(
                check_id=check_id,
                dataset=dataset,
                status=Status.WARNING if suspect else Status.PASS,
                evaluated=True,
                measurement={
                    **measurement,
                    "distinct_feature_values": int(distinct),
                    "target_like": bool(suspect),
                },
                threshold=threshold,
                evidence={
                    "column": column,
                    "relationship": relationship,
                    "mapping_examples": [
                        {"feature_value": value, "target_label": label}
                        for value, label in sorted(pairs)[: policy.max_examples]
                    ]
                    if suspect
                    else [],
                    "scope": "Comparable rows only; cannot prove or rule out leakage.",
                },
                message="Suspicious target relationship; investigate its provenance."
                if suspect
                else "This heuristic found no exact two-value target relationship.",
            )
        )
    return results
