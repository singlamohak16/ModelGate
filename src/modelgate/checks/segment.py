"""Common segment limits and explicit small-sample/coverage evidence."""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from modelgate.checks.policy import Fraction, Label
from modelgate.metrics import validate_labels
from modelgate.results import CheckResult, Status
from modelgate.schema import SEGMENT_COLUMN, TARGET_COLUMN
from modelgate.segments import COMPARISON_METRICS, analyze_segments


class SegmentCheckPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_column: StrictStr = TARGET_COLUMN
    segment_column: StrictStr = SEGMENT_COLUMN
    target_labels: tuple[Label, Label] = (0, 1)
    positive_class_label: Label = 1
    decision_threshold: Fraction = 0.5
    min_segment_size: int = Field(default=50, ge=1, strict=True)
    min_precision: Fraction | None = None
    min_recall: Fraction | None = None
    min_f1: Fraction | None = None
    min_pr_auc: Fraction | None = None

    @model_validator(mode="after")
    def consistent_policy(self):
        validate_labels(self.target_labels, self.positive_class_label)
        if (
            not self.target_column.strip()
            or not self.segment_column.strip()
            or self.target_column == self.segment_column
        ):
            raise ValueError("Target and segment names must be nonblank and distinct.")
        return self


def run_segment_checks(
    frame: pd.DataFrame,
    probabilities,
    policy: SegmentCheckPolicy,
    *,
    dataset="reference",
) -> dict:
    """Evaluate one labeled frame; invalid inputs raise, missing segment values fail."""
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Require a DataFrame with unique column names.")
    if not {policy.target_column, policy.segment_column} <= set(frame.columns):
        raise ValueError("Configured target and segment columns must be present.")
    report = analyze_segments(
        frame[policy.target_column],
        probabilities,
        frame[policy.segment_column],
        min_segment_size=policy.min_segment_size,
        decision_threshold=policy.decision_threshold,
        target_labels=policy.target_labels,
        positive_label=policy.positive_class_label,
    )
    checks = []

    def add(check_id, status, evaluated, measurement, threshold, evidence, message):
        checks.append(
            CheckResult(
                check_id=check_id,
                dataset=dataset,
                status=status,
                evaluated=evaluated,
                measurement=measurement,
                threshold=threshold,
                evidence=evidence,
                message=message,
            ).model_dump(mode="json")
        )

    coverage = report["coverage"]
    missing = coverage["missing_segment_rows"]
    add(
        "segment.completeness",
        Status.FAIL if missing else Status.PASS,
        True,
        {"count": missing, "fraction": missing / coverage["total_rows"]},
        {"operator": "<=", "value": 0},
        dict(coverage),
        "Missing segment rows remain in overall metrics but not named comparisons."
        if missing
        else "Every row has a named segment.",
    )
    for metric in COMPARISON_METRICS:
        benchmark = report["best_by_metric"][metric]
        multiple = benchmark["candidate_count"] >= 2
        add(
            f"segment.comparison_support.{metric}",
            Status.PASS if multiple else Status.WARNING,
            True,
            {"candidate_count": benchmark["candidate_count"]},
            {"operator": ">=", "value": 2},
            dict(benchmark),
            "At least two eligible segments have defined measurements."
            if multiple
            else "Fewer than two comparable segments; a self-gap is not peer evidence.",
        )
    for group in report["segments"]:
        evidence = {
            key: group[key]
            for key in (
                "segment",
                "record_count",
                "positive_count",
                "negative_count",
                "prevalence",
            )
        }
        evidence["segment_column"] = policy.segment_column
        evidence["decision_threshold"] = policy.decision_threshold
        # IDs describe the rule, while evidence.segment identifies its group.
        add(
            "segment.sample_size",
            Status.PASS if group["eligible"] else Status.WARNING,
            True,
            {"value": group["record_count"]},
            {"operator": ">=", "value": policy.min_segment_size},
            dict(evidence),
            "Minimum size met; this is not a statistical reliability guarantee."
            if group["eligible"]
            else "Too few rows; performance checks withheld.",
        )
        both = group["positive_count"] > 0 and group["negative_count"] > 0
        add(
            "segment.class_support",
            Status.PASS if both else Status.WARNING,
            True,
            {"both_classes_present": both},
            {"expected_classes": 2},
            dict(evidence),
            "Both outcome classes present."
            if both
            else "One outcome class is absent; performance evidence is limited.",
        )
        for metric in COMPARISON_METRICS:
            value = group["metrics"][metric] if group["eligible"] else None
            limit = getattr(policy, f"min_{metric}")
            status = Status.PASS
            message = "Descriptive measurement only; no quality limit configured."
            if value is None:
                status = Status.WARNING
                message = (
                    "Segment is below min_segment_size."
                    if not group["eligible"]
                    else group["metrics"]["undefined_reasons"][metric]
                )
            elif limit is not None:
                status = Status.FAIL if value < limit else Status.PASS
                message = (
                    "Configured limit violated."
                    if value < limit
                    else "Configured limit met."
                )
            metric_evidence = dict(evidence)
            metric_evidence["differences"] = group["differences"][metric]
            metric_evidence["best_comparator"] = report["best_by_metric"][metric]
            if group["eligible"]:
                metric_evidence["confusion_matrix"] = group["metrics"][
                    "confusion_matrix"
                ]
            add(
                f"segment.{metric}",
                status,
                value is not None,
                {"value": value},
                {"operator": ">=", "value": limit, "applied": limit is not None},
                metric_evidence,
                message,
            )
    report.update(segment_column=policy.segment_column, checks=checks)
    return report
