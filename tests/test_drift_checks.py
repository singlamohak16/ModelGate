import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from modelgate.checks.drift import DriftCheckPolicy, run_drift_checks


def policy(**kwargs):
    return DriftCheckPolicy(
        numerical_columns=("x",),
        categorical_columns=("group",),
        target_column="y",
        identifier_columns=("id",),
        **kwargs,
    )


def frame(n=50):
    return pd.DataFrame({"x": np.zeros(n), "group": ["A"] * n})


def test_identical_and_shifted_limits():
    ref, cur = frame(), frame()
    p = policy(
        max_numerical_ks=0.5, max_categorical_distance=0.5, max_prediction_ks=0.5
    )
    results = run_drift_checks(
        ref,
        cur,
        p,
        reference_probabilities=[0.1] * 50,
        current_probabilities=[0.1] * 50,
    )
    assert all(r.status == "PASS" and r.measurement["value"] == 0 for r in results)
    cur["x"], cur["group"] = 10, "B"
    shifted = run_drift_checks(
        ref,
        cur,
        p,
        reference_probabilities=[0.1] * 50,
        current_probabilities=[0.9] * 50,
    )
    assert all(r.status == "WARNING" and r.evaluated for r in shifted)
    assert shifted[0].measurement["value"] == shifted[2].measurement["value"] == 1
    assert shifted[1].measurement["value"] > 0.99
    assert shifted[1].evidence["new_category_count"] == 1


def test_strict_boundary_and_independent_invalid_feature():
    ref, cur = frame(4), frame(4)
    cur["x"] = [0, 0, 1, 1]
    results = run_drift_checks(
        ref, cur, policy(min_sample_size=4, max_numerical_ks=0.5)
    )
    assert results[0].status == "PASS" and results[0].measurement["value"] == 0.5
    ref["x"] = ["bad", "0", "0", "0"]
    results = run_drift_checks(ref, cur, policy(min_sample_size=4))
    assert results[0].status == "WARNING" and not results[0].evaluated
    assert results[0].evidence["reference"]["invalid"] == 1
    assert results[1].evaluated and results[1].measurement["value"] == 0
    assert not results[2].evaluated


def test_49_50_and_missing_numerical_coverage():
    ref, cur = frame(), frame()
    ref.loc[0, "x"] = np.nan
    checks = run_drift_checks(ref, cur, policy())
    assert not checks[0].evaluated
    assert checks[0].evidence["reference"]["usable"] == 49
    assert checks[0].evidence["reference"]["missing"] == 1
    assert checks[1].evaluated
    checks = run_drift_checks(ref, cur, policy(min_sample_size=49))
    assert checks[0].evaluated and checks[0].measurement["value"] == 0


def test_all_missing_numerical_and_categorical():
    ref = pd.DataFrame({"x": [None] * 50, "group": [None] * 50})
    checks = run_drift_checks(ref, ref.copy(), policy())
    assert checks[0].evidence["reference"]["usable"] == 0
    assert not checks[0].evaluated
    assert checks[1].measurement["value"] == 0
    assert checks[1].evidence["reference_missing"] == 50


def test_missing_current_is_never_zero_drift():
    checks = run_drift_checks(frame(), None, policy())
    assert len(checks) == 3
    assert all(
        not c.evaluated and c.status == "WARNING" and c.measurement["value"] is None
        for c in checks
    )


def test_absent_column_and_invalid_category_do_not_block_predictions():
    ref, cur = frame(), frame().drop(columns="x")
    cur["group"] = 123
    checks = run_drift_checks(
        ref,
        cur,
        policy(),
        reference_probabilities=[0.5] * 50,
        current_probabilities=[0.5] * 50,
    )
    assert not checks[0].evaluated and not checks[1].evaluated
    assert checks[2].evaluated


@pytest.mark.parametrize(
    "probabilities", [[0.5], [1.1] * 50, pd.Series([0.5] * 50, index=range(1, 51))]
)
def test_bad_prediction_vectors(probabilities):
    checks = run_drift_checks(
        frame(),
        frame(),
        policy(),
        reference_probabilities=[0.5] * 50,
        current_probabilities=probabilities,
    )
    assert not checks[-1].evaluated
    assert all(c.evaluated for c in checks[:-1])


def test_no_mutation_and_descriptive_limits():
    ref, cur = frame(), frame()
    originals = [x.copy(deep=True) for x in (ref, cur)]
    checks = run_drift_checks(
        ref,
        cur,
        policy(),
        reference_probabilities=[0.5] * 50,
        current_probabilities=[0.5] * 50,
    )
    for a, b in zip((ref, cur), originals, strict=True):
        pd.testing.assert_frame_equal(a, b)
    assert all(
        not c.threshold["applied"] and "Descriptive" in c.message for c in checks
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_sample_size": 0},
        {"min_sample_size": True},
        {"max_numerical_ks": np.nan},
        {"max_prediction_ks": 1.1},
        {"max_categorical_distance": True},
        {"smoothing": 1},
        {"smoothing": -0.1},
        {"max_examples": 21},
        {"extra": 1},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ValidationError):
        policy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"numerical_columns": ("Churn",)},
        {"categorical_columns": ("customerID",)},
        {"numerical_columns": ("x", "x")},
        {"numerical_columns": (" ",)},
        {"numerical_columns": ("Contract",)},
        {"identifier_columns": ("Churn",)},
    ],
)
def test_role_conflicts(kwargs):
    with pytest.raises(ValidationError):
        DriftCheckPolicy(**kwargs)


@pytest.mark.parametrize(
    "bad", [None, {}, pd.DataFrame(), pd.DataFrame([[1, 2]], columns=["x", "x"])]
)
def test_bad_reference_structure_blocks_all(bad):
    checks = run_drift_checks(bad, frame(), policy())
    assert all(not row.evaluated and row.measurement["value"] is None for row in checks)


def test_categorical_and_prediction_exact_limits_and_small_current():
    ref, cur = frame(4), frame(4)
    cur["group"] = "B"
    checks = run_drift_checks(
        ref,
        cur,
        policy(
            min_sample_size=4,
            smoothing=0,
            max_categorical_distance=1,
            max_prediction_ks=0.5,
        ),
        reference_probabilities=[0, 0, 0, 0],
        current_probabilities=[0, 0, 1, 1],
    )
    assert checks[1].measurement["value"] == 1 and checks[1].status == "PASS"
    assert checks[2].measurement["value"] == 0.5 and checks[2].status == "PASS"
    checks = run_drift_checks(
        ref,
        cur.iloc[:3],
        policy(min_sample_size=4),
        reference_probabilities=[0.5] * 4,
        current_probabilities=[0.5] * 3,
    )
    assert all(not row.evaluated for row in checks)


def test_nonscalar_numeric_observation_does_not_suppress_other_checks():
    ref, cur = frame(2), frame(2)
    ref["x"] = pd.Series([{"bad": 1}, 0], dtype=object)
    checks = run_drift_checks(ref, cur, policy(min_sample_size=2))
    assert not checks[0].evaluated and "scalar" in checks[0].message
    assert checks[1].evaluated
