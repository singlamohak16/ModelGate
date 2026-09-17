"""Shared scalar, prerequisite, and evidence rules for CSV-like DataFrames."""

import math
from decimal import Decimal

import numpy as np
import pandas as pd

from modelgate.checks.policy import DataCheckPolicy
from modelgate.results import CheckResult, Status


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.map(
        lambda value: isinstance(value, str) and not value.strip()
    )


def json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def label_token(value) -> str | None:
    """Permit configured integer labels as integers or CSV text; never bool labels."""
    if isinstance(value, (bool, np.bool_)) or pd.isna(value):
        return None
    if (
        isinstance(value, (float, np.floating))
        and math.isfinite(value)
        and value.is_integer()
    ):
        return str(int(value))
    return str(value)


def target_tokens(frame: pd.DataFrame, policy: DataCheckPolicy) -> pd.Series:
    return frame[policy.target_column].map(label_token)


def structure_problem(frame: pd.DataFrame) -> str | None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Checks require a pandas DataFrame.")
    if not frame.columns.is_unique:
        return "Duplicate column names prevent unambiguous selection."
    if any(
        not isinstance(column, str) or not column.strip() for column in frame.columns
    ):
        return "Column names must be nonblank strings."
    if len(frame) == 0:
        return "Dataset contains no rows."
    if len(frame.columns) == 0:
        return "Dataset contains no columns."
    return None


def prerequisite(frame: pd.DataFrame, columns=()) -> str | None:
    problem = structure_problem(frame)
    if problem:
        return problem
    absent = sorted(set(columns) - set(frame.columns))
    return f"Required columns are absent: {absent}" if absent else None


def blocked(
    check_id: str,
    dataset: str,
    reason: str,
    threshold: dict,
    measurement: dict | None = None,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        dataset=dataset,
        status=Status.WARNING,
        evaluated=False,
        measurement=measurement or {"value": None},
        threshold=threshold,
        evidence={"reason": reason},
        message=f"Check not evaluated: {reason}",
    )


def examples(series: pd.Series, mask: pd.Series, limit: int) -> list[dict]:
    # Positional evidence works even when DataFrame index labels are duplicated.
    return [
        {"row_position": int(position), "value": json_scalar(series.iloc[position])}
        for position in np.flatnonzero(mask.to_numpy())[:limit]
    ]


def numerical_values(series: pd.Series, kind: str) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce")
    invalid = ~missing_mask(series) & (
        ~np.isfinite(values)
        | series.map(lambda value: isinstance(value, (bool, np.bool_)))
    )
    if kind == "integer":
        invalid |= ~missing_mask(series) & (values % 1 != 0)
    return values, invalid


def identifier_keys(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> list[tuple | None]:
    valid = ~pd.concat([missing_mask(frame[column]) for column in columns], axis=1).any(
        axis=1
    )
    return [
        tuple(str(value) for value in row) if valid.iloc[position] else None
        for position, row in enumerate(
            frame[list(columns)].itertuples(index=False, name=None)
        )
    ]


def identifier_examples(
    keys: list[tuple | None], columns: tuple[str, ...], limit: int
) -> list[dict]:
    unique = sorted({key for key in keys if key is not None})
    return [dict(zip(columns, key, strict=True)) for key in unique[:limit]]


def record_keys(frame: pd.DataFrame, types: dict[str, str]) -> list[tuple]:
    """Structured exact keys; Decimal avoids rounding distinct numeric CSV values."""
    keys = []
    for row in frame[list(types)].itertuples(index=False, name=None):
        key = []
        for value, kind in zip(row, types.values(), strict=True):
            if pd.isna(value) or isinstance(value, str) and not value.strip():
                key.append(("missing",))
            elif kind in ("number", "integer"):
                key.append(("number", Decimal(str(value))))
            else:
                key.append(("category", str(value)))
        keys.append(tuple(key))
    return keys
