"""Regression checks for malformed input at the public loading boundaries."""

from types import SimpleNamespace

import pandas as pd
import pytest

from modelgate.config import ConfigError, load_yaml_config
from modelgate.validation import prediction_features, read_audit_csv


@pytest.mark.parametrize("case", ["absent", "invalid_utf8", "non_scalar_key"])
def test_unreadable_or_ambiguous_configuration_is_rejected(tmp_path, case):
    path = tmp_path / "config.yaml"
    if case == "invalid_utf8":
        path.write_bytes(b"name: \xff")
    elif case == "non_scalar_key":
        path.write_text("? [one, two]\n: value\n", encoding="utf-8")
    message = "invalid YAML" if case == "non_scalar_key" else "Could not read"
    with pytest.raises(ConfigError, match=message):
        load_yaml_config(path)


def test_empty_csv_has_no_invented_header(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="no header"):
        read_audit_csv(path)


def test_disallowed_predictor_category_blocks_inference_without_mutation():
    frame = pd.DataFrame({"Contract": ["Month-to-month", "unknown"]}, index=[5, 2])
    before = frame.copy(deep=True)
    config = SimpleNamespace(
        predictors=["Contract"],
        data=SimpleNamespace(
            column_types={"Contract": "category"},
            allowed_categories={"Contract": ["Month-to-month", "One year"]},
        ),
    )
    features, reason = prediction_features(frame, config)
    assert features is None
    assert reason == "Invalid configured category: Contract"
    pd.testing.assert_frame_equal(frame, before)
