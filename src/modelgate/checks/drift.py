"""Independent feature and probability comparisons; no overall decision."""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from modelgate.checks._common import structure_problem
from modelgate.checks.policy import Fraction
from modelgate.drift import (
    categorical_distance,
    ks_distance,
    numerical_sample,
    probability_vector,
)
from modelgate.results import CheckResult, Status
from modelgate.schema import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


class DriftCheckPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    numerical_columns: tuple[StrictStr, ...] = NUMERIC_COLUMNS
    categorical_columns: tuple[StrictStr, ...] = CATEGORICAL_COLUMNS
    target_column: StrictStr = TARGET_COLUMN
    identifier_columns: tuple[StrictStr, ...] = (ID_COLUMN,)
    min_sample_size: int = Field(default=50, ge=1, strict=True)
    smoothing: float = Field(default=1e-6, ge=0, lt=1, allow_inf_nan=False, strict=True)
    max_examples: int = Field(default=10, ge=0, le=20, strict=True)
    max_numerical_ks: Fraction | None = None
    max_categorical_distance: Fraction | None = None
    max_prediction_ks: Fraction | None = None

    @model_validator(mode="after")
    def consistent_roles(self):
        features = (*self.numerical_columns, *self.categorical_columns)
        all_names = (*features, self.target_column, *self.identifier_columns)
        if any(not name.strip() for name in all_names) or len(set(all_names)) != len(
            all_names
        ):
            raise ValueError(
                "Feature, target and ID roles must be nonblank and disjoint."
            )
        return self


def drift_result(check_id, value, limit, evidence, reason=None) -> CheckResult:
    status = (
        Status.WARNING
        if reason or (limit is not None and value > limit)
        else Status.PASS
    )
    message = reason or (
        "Configured drift limit exceeded; this does not establish performance loss."
        if status == Status.WARNING
        else "Configured limit met."
        if limit is not None
        else "Descriptive distance only; no drift limit configured."
    )
    return CheckResult(
        check_id=check_id,
        dataset="reference_vs_current",
        status=status,
        evaluated=reason is None,
        measurement={"value": value},
        threshold={"operator": "<=", "value": limit, "applied": limit is not None},
        evidence=evidence,
        message=message,
    )


def run_drift_checks(
    reference,
    current,
    policy: DriftCheckPolicy,
    *,
    reference_probabilities=None,
    current_probabilities=None,
) -> list[CheckResult]:
    """Report blocked checks separately; bad features never suppress other features."""
    problems = []
    for name, frame in (("reference", reference), ("current", current)):
        problem = (
            "Dataset not supplied."
            if frame is None
            else (
                "Require a pandas DataFrame."
                if not isinstance(frame, pd.DataFrame)
                else structure_problem(frame)
            )
        )
        if problem:
            problems.append(f"{name}: {problem}")
    base_reason = " ".join(problems) or None
    results = []
    for method, columns, limit in (
        ("numerical_ks", policy.numerical_columns, policy.max_numerical_ks),
        (
            "categorical_distance",
            policy.categorical_columns,
            policy.max_categorical_distance,
        ),
    ):
        for column in columns:
            reason = base_reason
            evidence = {
                "column": column,
                "method": method,
                "min_sample_size": policy.min_sample_size,
            }
            value = None
            if not reason:
                absent = [
                    name
                    for name, frame in (("reference", reference), ("current", current))
                    if column not in frame.columns
                ]
                if absent:
                    reason = f"Column absent from: {', '.join(absent)}."
            if not reason:
                try:
                    if method == "numerical_ks":
                        ref, ref_stats = numerical_sample(reference[column])
                        cur, cur_stats = numerical_sample(current[column])
                        evidence.update(reference=ref_stats, current=cur_stats)
                        if ref_stats["invalid"] or cur_stats["invalid"]:
                            reason = (
                                "Invalid numeric observations block this comparison."
                            )
                        elif min(len(ref), len(cur)) < policy.min_sample_size:
                            reason = "Too few usable numerical observations."
                        else:
                            value = ks_distance(ref, cur)
                    else:
                        measurement = categorical_distance(
                            reference[column],
                            current[column],
                            smoothing=policy.smoothing,
                            max_examples=policy.max_examples,
                        )
                        evidence.update(
                            {k: v for k, v in measurement.items() if k != "value"}
                        )
                        if min(len(reference), len(current)) < policy.min_sample_size:
                            reason = "Too few categorical observations."
                        else:
                            value = measurement["value"]
                except ValueError as error:
                    reason = str(error)
            results.append(
                drift_result(f"drift.{method}", value, limit, evidence, reason)
            )
    value = None
    reason = base_reason
    evidence = {
        "method": "prediction_ks",
        "min_sample_size": policy.min_sample_size,
        "same_model_required": True,
    }
    if not reason:
        try:
            vectors = []
            for name, frame, probabilities in (
                ("reference", reference, reference_probabilities),
                ("current", current, current_probabilities),
            ):
                if probabilities is None:
                    raise ValueError(f"{name} probabilities not supplied.")
                if isinstance(
                    probabilities, pd.Series
                ) and not probabilities.index.equals(frame.index):
                    raise ValueError(
                        f"{name} probability index does not match its frame."
                    )
                vector = probability_vector(probabilities)
                if len(vector) != len(frame):
                    raise ValueError(
                        f"{name} probability count does not match its frame."
                    )
                vectors.append(vector)
                evidence[f"{name}_count"] = len(vector)
            if min(map(len, vectors)) < policy.min_sample_size:
                reason = "Too few predicted probabilities."
            else:
                value = ks_distance(*vectors)
        except ValueError as error:
            reason = str(error)
    results.append(
        drift_result(
            "drift.prediction_ks", value, policy.max_prediction_ks, evidence, reason
        )
    )
    return results
