"""Model measurements and optional rule decisions, not a run-level report."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modelgate.checks.policy import Fraction, Label
from modelgate.metrics import (
    DEFAULT_THRESHOLDS,
    classification_metrics,
    probability_metrics,
    threshold_analysis,
    validate_labels,
)
from modelgate.results import CheckResult, Status


class ModelCheckPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target_labels: tuple[Label, Label] = (0, 1)
    positive_class_label: Label = 1
    decision_threshold: Fraction = 0.5
    thresholds: tuple[Fraction, ...] = DEFAULT_THRESHOLDS
    calibration_bins: int = Field(default=10, ge=2, le=100, strict=True)
    min_precision: Fraction | None = None
    min_recall: Fraction | None = None
    min_f1: Fraction | None = None
    min_pr_auc: Fraction | None = None
    max_brier_score: Fraction | None = None

    @model_validator(mode="after")
    def consistent_policy(self):
        validate_labels(self.target_labels, self.positive_class_label)
        if not self.thresholds or len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("Threshold grid must be nonempty and unique.")
        return self


def run_model_checks(
    labels, probabilities, policy: ModelCheckPolicy, *, dataset="reference"
):
    """Invalid inputs raise ValueError; valid but undefined metrics warn."""
    options = {
        "target_labels": policy.target_labels,
        "positive_label": policy.positive_class_label,
    }
    metrics = classification_metrics(
        labels, probabilities, policy.decision_threshold, **options
    )
    calibration = probability_metrics(
        labels, probabilities, policy.calibration_bins, **options
    )
    grid = threshold_analysis(labels, probabilities, policy.thresholds, **options)
    evidence = {
        key: metrics[key]
        for key in (
            "sample_size",
            "positive_count",
            "prevalence",
            "tp",
            "fp",
            "tn",
            "fn",
        )
    }
    evidence["decision_threshold"] = policy.decision_threshold
    results = []
    for name in ("precision", "recall", "f1", "pr_auc", "brier_score"):
        value = calibration[name] if name == "brier_score" else metrics[name]
        maximum = name == "brier_score"
        limit = getattr(policy, f"{'max' if maximum else 'min'}_{name}")
        threshold = {
            "value": limit,
            "operator": "<=" if maximum else ">=",
            "applied": limit is not None,
        }
        status = Status.PASS
        message = "Descriptive measurement only; no quality limit configured."
        if value is None:
            status = Status.WARNING
            message = metrics["undefined_reasons"][name]
        elif limit is not None:
            failed = value > limit if maximum else value < limit
            status = Status.FAIL if failed else Status.PASS
            message = (
                "Configured limit violated." if failed else "Configured limit met."
            )
        metric_evidence = dict(evidence)
        if name == "pr_auc":
            metric_evidence["method"] = "trapezoidal"
        results.append(
            CheckResult(
                check_id=f"model.{name}",
                dataset=dataset,
                status=status,
                evaluated=value is not None,
                measurement={"value": value},
                threshold=threshold,
                evidence=metric_evidence,
                message=message,
            )
        )
    both_classes = 0 < metrics["positive_count"] < metrics["sample_size"]
    results.append(
        CheckResult(
            check_id="model.class_support",
            dataset=dataset,
            status=Status.PASS if both_classes else Status.WARNING,
            evaluated=True,
            measurement={"both_classes_present": both_classes},
            threshold={"expected_classes": 2},
            evidence=evidence,
            message="Both classes present."
            if both_classes
            else "One outcome class is absent; performance evidence is limited.",
        )
    )
    return {
        "metrics": metrics,
        "probability_metrics": calibration,
        "threshold_analysis": grid,
        "checks": [result.model_dump(mode="json") for result in results],
    }
