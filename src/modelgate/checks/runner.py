"""Compose data checks without modifying inputs or aggregating run status."""

import pandas as pd

from modelgate.checks.data import (
    check_duplicates,
    check_missingness,
    check_overlap,
    check_schema,
)
from modelgate.checks.leakage import check_leakage
from modelgate.checks.policy import DataCheckPolicy
from modelgate.results import CheckResult


def run_data_checks(
    train: pd.DataFrame,
    reference: pd.DataFrame,
    policy: DataCheckPolicy,
    *,
    current: pd.DataFrame | None = None,
    current_labeled: bool = False,
) -> list[CheckResult]:
    """Validate required train/reference and optional current data independently."""
    results = []
    datasets = [("train", train, True), ("reference", reference, True)]
    if current is not None:
        datasets.append(("current", current, current_labeled))
    for name, frame, labeled in datasets:
        results.extend(check_schema(frame, policy, name, labeled=labeled))
        results.extend(check_missingness(frame, policy, name, labeled=labeled))
        results.extend(check_duplicates(frame, policy, name))
        results.extend(check_leakage(frame, policy, name, labeled=labeled))
    results.extend(check_overlap(train, reference, policy))
    return results
