"""Configuration loading primitives for ModelGate."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import RootModel, ValidationError


class ConfigError(ValueError):
    """Raised when a configuration document cannot be loaded or validated."""


class ConfigDocument(RootModel[dict[str, Any]]):
    """Phase 0 representation of a YAML document with a mapping root."""


def load_yaml_config(path: str | Path) -> ConfigDocument:
    """Load a YAML mapping from ``path`` and return its validated document."""
    config_path = Path(path)

    try:
        raw_config = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            f"Could not read configuration '{config_path}': {error}"
        ) from error

    try:
        parsed_config = yaml.safe_load(raw_config)
    except yaml.YAMLError as error:
        raise ConfigError(
            f"Configuration '{config_path}' contains invalid YAML: {error}"
        ) from error

    try:
        return ConfigDocument.model_validate(parsed_config)
    except ValidationError as error:
        raise ConfigError(
            f"Configuration '{config_path}' must contain a YAML mapping at its root: "
            f"{error}"
        ) from error
