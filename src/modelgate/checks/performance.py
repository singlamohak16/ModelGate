"""Observed labeled performance differences, never inferred from unlabeled drift."""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from modelgate.checks.policy import Fraction, Label
from modelgate.metrics import (
    classification_metrics,
    probability_metrics,
    validate_labels,
)
from modelgate.results import CheckResult, Status
from modelgate.schema import TARGET_COLUMN

METRICS = ("precision", "recall", "f1", "pr_auc", "brier_score")


class PerformanceChangePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_column: StrictStr = TARGET_COLUMN
    target_labels: tuple[Label, Label] = (0, 1)
    positive_class_label: Label = 1
    decision_threshold: Fraction = 0.5
    min_sample_size: int = Field(default=50, ge=1, strict=True)
    max_precision_drop: Fraction | None = None
    max_recall_drop: Fraction | None = None
    max_f1_drop: Fraction | None = None
    max_pr_auc_drop: Fraction | None = None
    max_brier_increase: Fraction | None = None

    @model_validator(mode="after")
    def consistent_policy(self):
        validate_labels(self.target_labels, self.positive_class_label)
        if not self.target_column.strip():
            raise ValueError("Target name must be nonblank.")
        return self


def run_performance_checks(
    reference,
    current,
    reference_probabilities,
    current_probabilities,
    policy: PerformanceChangePolicy,
    *,
    current_labeled=False,
) -> dict:
    """Do not inspect current targets or return metric claims in unlabeled mode."""
    if not isinstance(current_labeled, bool):
        raise ValueError("current_labeled must be an explicit boolean.")
    if current is None or not current_labeled:
        return {
            "comparison_available": False,
            "reason": "Current dataset not supplied."
            if current is None
            else "Current labels not declared available; no performance comparison.",
            "reference": None,
            "current": None,
            "checks": [],
        }
    measurements, errors = {}, {}
    for name, frame, probabilities in (
        ("reference", reference, reference_probabilities),
        ("current", current, current_probabilities),
    ):
        try:
            if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
                raise ValueError("Require a DataFrame with unique columns.")
            if policy.target_column not in frame.columns:
                raise ValueError("Declared target column is missing.")
            options = {
                "target_labels": policy.target_labels,
                "positive_label": policy.positive_class_label,
            }
            labels = frame[policy.target_column]
            metrics = classification_metrics(
                labels, probabilities, policy.decision_threshold, **options
            )
            metrics["brier_score"] = probability_metrics(
                labels, probabilities, **options
            )["brier_score"]
            measurements[name] = metrics
            if len(frame) < policy.min_sample_size:
                errors[name] = (
                    "Too few labeled observations for performance comparison."
                )
        except ValueError as error:
            errors[name] = str(error)
    checks = []
    common_reason = (
        "; ".join(f"{key}: {value}" for key, value in errors.items()) or None
    )
    for metric in METRICS:
        limit = getattr(
            policy,
            "max_brier_increase" if metric == "brier_score" else f"max_{metric}_drop",
        )
        ref = measurements.get("reference", {}).get(metric)
        cur = measurements.get("current", {}).get(metric)
        reason = common_reason
        if not reason and (ref is None or cur is None):
            reason = "; ".join(
                f"{name}: {values['undefined_reasons'][metric]}"
                for name, values in measurements.items()
                if values[metric] is None
            )
        change = None if reason else cur - ref
        deterioration = (
            None if reason else change if metric == "brier_score" else -change
        )
        status = (
            Status.WARNING
            if reason
            else Status.FAIL
            if limit is not None and deterioration > limit
            else Status.PASS
        )
        evidence = {
            "min_sample_size": policy.min_sample_size,
            "decision_threshold": policy.decision_threshold,
            "direction": "current minus reference"
            if metric == "brier_score"
            else "reference minus current",
            "interpretation": "Observed difference, not significance or causation.",
        }
        for name, values in measurements.items():
            evidence[name] = {
                key: values[key]
                for key in (
                    "sample_size",
                    "positive_count",
                    "prevalence",
                    "confusion_matrix",
                )
            }
        checks.append(
            CheckResult(
                check_id=f"performance.{metric}",
                dataset="reference_vs_current",
                status=status,
                evaluated=reason is None,
                measurement={
                    "reference": ref,
                    "current": cur,
                    "current_minus_reference": change,
                    "deterioration": deterioration,
                },
                threshold={
                    "operator": "<=",
                    "value": limit,
                    "applied": limit is not None,
                },
                evidence=evidence,
                message=reason
                or (
                    "Configured deterioration limit exceeded."
                    if status == Status.FAIL
                    else "Configured limit met."
                    if limit is not None
                    else "Descriptive change only; no deterioration limit configured."
                ),
            ).model_dump(mode="json")
        )
    # One-class samples retain valid measurements but require a visible warning.
    for name, values in measurements.items():
        both = 0 < values["positive_count"] < values["sample_size"]
        checks.append(
            CheckResult(
                check_id="performance.class_support",
                dataset=name,
                status=Status.PASS if both else Status.WARNING,
                evaluated=True,
                measurement={"both_classes_present": both},
                threshold={"expected_classes": 2},
                evidence={
                    "sample_size": values["sample_size"],
                    "positive_count": values["positive_count"],
                },
                message="Both classes present."
                if both
                else "One outcome class absent; evidence is limited.",
            ).model_dump(mode="json")
        )
    return {
        "comparison_available": common_reason is None,
        "reason": common_reason,
        "reference": measurements.get("reference"),
        "current": measurements.get("current"),
        "checks": checks,
    }
