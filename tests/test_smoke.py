"""Phase 0 smoke tests."""

import logging

import pytest

import modelgate
from modelgate.config import ConfigError, load_yaml_config
from modelgate.logging import configure_logging


def test_package_exposes_version() -> None:
    assert modelgate.__version__ == "0.1.0"


def test_load_yaml_config_accepts_mapping(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: ModelGate\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config.root == {"project": {"name": "ModelGate"}}


def test_load_yaml_config_rejects_invalid_yaml(tmp_path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("project: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="contains invalid YAML"):
        load_yaml_config(config_path)


def test_load_yaml_config_rejects_non_mapping_root(tmp_path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML mapping at its root"):
        load_yaml_config(config_path)


def test_configure_logging_does_not_duplicate_its_handler() -> None:
    logger_name = "modelgate.tests.smoke"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    first_logger = configure_logging(logger_name=logger_name)
    second_logger = configure_logging(logger_name=logger_name)

    assert first_logger is second_logger
    assert len(logger.handlers) == 1
    assert logger.propagate is False

    logger.handlers.clear()
