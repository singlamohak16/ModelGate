import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from modelgate.drift import (
    categorical_distance,
    ks_distance,
    numerical_sample,
    probability_vector,
)


@pytest.mark.parametrize(
    "ref,cur,expected",
    [
        ([1, 2, 3], [1, 2, 3], 0),
        ([0, 0], [1, 1], 1),
        ([0, 1], [0, 0, 1, 1], 0),
        ([0, 0], [0, 1], 0.5),
        ([1], [2], 1),
        ([1, 1], [1, 1, 1], 0),
    ],
)
def test_ks_hand_values(ref, cur, expected):
    assert ks_distance(ref, cur) == expected
    assert ks_distance(cur, ref) == expected
    assert ks_distance(ref, cur) == pytest.approx(ks_2samp(ref, cur).statistic)


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_ks_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    ref, cur = rng.normal(size=80), rng.normal(1, 1, size=130)
    original = ref.copy()
    assert ks_distance(ref, cur) == pytest.approx(ks_2samp(ref, cur).statistic)
    np.testing.assert_array_equal(ref, original)


@pytest.mark.parametrize(
    "bad", [[], [[1, 2]], [True, 0.1], ["1", "2"], [np.nan], [np.inf]]
)
def test_ks_rejects_bad_vectors(bad):
    with pytest.raises(ValueError):
        ks_distance(bad, [0, 1])


def test_numeric_missing_invalid_counts():
    sample, counts = numerical_sample(["1", " ", None, "oops", np.inf, True, "2"])
    np.testing.assert_array_equal(sample, [1, 2])
    assert counts == {
        "total": 7,
        "missing": 2,
        "invalid": 3,
        "usable": 2,
        "usable_fraction": 2 / 7,
    }


def test_categorical_hand_distance_and_smoothing():
    assert categorical_distance(["A"], ["A"])["value"] == 0
    assert categorical_distance(["A"], ["B"], smoothing=0)["value"] == 1
    result = categorical_distance(["A"] * 3 + ["B"], ["A"] + ["B"] * 3, smoothing=0.1)
    p = 0.9 * np.array([0.75, 0.25]) + 0.1 / 2
    q = 0.9 * np.array([0.25, 0.75]) + 0.1 / 2
    assert result["value"] == pytest.approx(jensenshannon(p, q, base=2))
    assert result["smoothing"] == 0.1
    assert result["log_base"] == 2
    assert result["category_count"] == 2
    assert result["examples"][0]["reference_smoothed"] == pytest.approx(p[0])
    assert result == categorical_distance(
        ["B", "A", "A", "A"], ["B", "B", "B", "A"], smoothing=0.1
    )


def test_equal_proportions_unequal_sizes_and_missing_sentinel():
    assert categorical_distance(["A", "B"], ["A", "A", "B", "B"])["value"] == 0
    result = categorical_distance([None, "missing", "NA"], ["", "missing", "new"])
    assert result["reference_missing"] == result["current_missing"] == 1
    assert result["category_count"] == 4
    assert result["new_category_count"] == result["removed_category_count"] == 1
    assert any(row["is_missing"] for row in result["examples"])
    assert any(
        row["category"] == "missing" and not row["is_missing"]
        for row in result["examples"]
    )
    json.dumps(result, allow_nan=False)


def test_examples_are_bounded_and_whitespace_is_missing():
    result = categorical_distance(["A", "B", " "], ["C", "D", pd.NA], max_examples=1)
    assert len(result["examples"]) == 1
    assert result["examples_truncated"]
    assert result["reference_missing"] == result["current_missing"] == 1
    assert categorical_distance(["a"], ["b"], max_examples=0)["examples"] == []
    assert categorical_distance([None], [pd.NA])["value"] == 0


@pytest.mark.parametrize("bad", [[], [["a"]], [1], [False], [np.inf], [{"a": 1}]])
def test_invalid_categories(bad):
    with pytest.raises(ValueError):
        categorical_distance(bad, ["a"])


@pytest.mark.parametrize("epsilon", [-1, 1, np.nan, True, "0.1"])
def test_invalid_smoothing(epsilon):
    with pytest.raises(ValueError):
        categorical_distance(["a"], ["b"], smoothing=epsilon)


@pytest.mark.parametrize("p", [[-0.01, 1], [0, 1.01], [np.nan], [True, 0.5]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        probability_vector(p)


def test_smoothing_reduces_distance_and_examples_do_not_change_it():
    values = [
        categorical_distance(["A"], ["B"], smoothing=e)["value"] for e in (0, 0.1, 0.9)
    ]
    assert values[0] > values[1] > values[2] > 0
    assert (
        categorical_distance(["A"], ["B"], max_examples=0)["value"]
        == categorical_distance(["A"], ["B"])["value"]
    )
