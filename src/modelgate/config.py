"""Configuration loading primitives for ModelGate."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from modelgate.checks.drift import DriftCheckPolicy
from modelgate.checks.model import ModelCheckPolicy
from modelgate.checks.performance import PerformanceChangePolicy
from modelgate.checks.policy import DataCheckPolicy, Fraction
from modelgate.checks.segment import SegmentCheckPolicy
from modelgate.fingerprint import sha256_bytes
from modelgate.schema import FEATURE_COLUMNS


class ConfigError(ValueError):
    """Raised when a configuration document cannot be loaded or validated."""


class ConfigDocument(RootModel[dict[str, Any]]):
    """Low-level mapping loader retained for callers; not a run configuration."""

    _source_sha256: str = PrivateAttr(default="")


class UniqueKeyLoader(yaml.SafeLoader):
    """Refuse ambiguous duplicate keys, including merged YAML mappings."""


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in result:
                raise yaml.YAMLError(f"Duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as error:
            raise yaml.YAMLError("YAML mapping keys must be scalar.") from error
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def load_yaml_config(path: str | Path) -> ConfigDocument:
    """Load a YAML mapping from ``path`` and return its validated document."""
    config_path = Path(path)

    try:
        payload = config_path.read_bytes()
        raw_config = payload.decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ConfigError(
            f"Could not read configuration '{config_path}': {error}"
        ) from error

    try:
        parsed_config = yaml.load(raw_config, Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigError(
            f"Configuration '{config_path}' contains invalid YAML: {error}"
        ) from error

    try:
        document = ConfigDocument.model_validate(parsed_config)
        document._source_sha256 = sha256_bytes(payload)
        return document
    except ValidationError as error:
        raise ConfigError(
            f"Configuration '{config_path}' must contain a YAML mapping at its root: "
            f"{error}"
        ) from error


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Inputs(Settings):
    train: Path
    reference: Path
    current: Path | None = None
    current_labeled: StrictBool = False


class Artifact(Settings):
    path: Path
    trust: StrictBool = False
    expected_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ModelSettings(Settings):
    enabled: StrictBool = True
    thresholds: tuple[Fraction, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    calibration_bins: int = Field(default=10, ge=2, le=100, strict=True)
    min_precision: Fraction | None = None
    min_recall: Fraction | None = None
    min_f1: Fraction | None = None
    min_pr_auc: Fraction | None = None
    max_brier_score: Fraction | None = None


class SegmentSettings(Settings):
    enabled: StrictBool = True
    min_segment_size: int = Field(default=50, ge=1, strict=True)
    min_precision: Fraction | None = None
    min_recall: Fraction | None = None
    min_f1: Fraction | None = None
    min_pr_auc: Fraction | None = None


class DriftSettings(Settings):
    enabled: StrictBool = True
    min_sample_size: int = Field(default=50, ge=1, strict=True)
    smoothing: float = Field(default=1e-6, ge=0, lt=1, strict=True, allow_inf_nan=False)
    max_examples: int = Field(default=10, ge=0, le=20, strict=True)
    max_numerical_ks: Fraction | None = None
    max_categorical_distance: Fraction | None = None
    max_prediction_ks: Fraction | None = None


class PerformanceSettings(Settings):
    enabled: StrictBool = True
    min_sample_size: int = Field(default=50, ge=1, strict=True)
    max_precision_drop: Fraction | None = None
    max_recall_drop: Fraction | None = None
    max_f1_drop: Fraction | None = None
    max_pr_auc_drop: Fraction | None = None
    max_brier_increase: Fraction | None = None


class RunConfig(Settings):
    _source_sha256: str | None = PrivateAttr(default=None)
    _source_path: Path | None = PrivateAttr(default=None)
    inputs: Inputs
    artifact: Artifact
    output: Path
    seed: StrictInt = 42
    # Data policy is the authoritative source of roles, labels and feature types.
    data: DataCheckPolicy = Field(default_factory=DataCheckPolicy)
    predictors: tuple[str, ...] = FEATURE_COLUMNS
    decision_threshold: Fraction = 0.5
    model: ModelSettings = Field(default_factory=ModelSettings)
    segment: SegmentSettings = Field(default_factory=SegmentSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)

    def policies(self):
        labels = dict(
            target_labels=self.data.target_labels,
            positive_class_label=self.data.positive_class_label,
            decision_threshold=self.decision_threshold,
        )
        return {
            "model": ModelCheckPolicy(
                **labels, **self.model.model_dump(exclude={"enabled"})
            ),
            "segment": SegmentCheckPolicy(
                **labels,
                target_column=self.data.target_column,
                segment_column=self.data.segment_column,
                **self.segment.model_dump(exclude={"enabled"}),
            ),
            "performance": PerformanceChangePolicy(
                **labels,
                target_column=self.data.target_column,
                **self.performance.model_dump(exclude={"enabled"}),
            ),
            "drift": DriftCheckPolicy(
                numerical_columns=tuple(
                    c
                    for c in self.predictors
                    if self.data.column_types[c] != "category"
                ),
                categorical_columns=tuple(
                    c
                    for c in self.predictors
                    if self.data.column_types[c] == "category"
                ),
                target_column=self.data.target_column,
                identifier_columns=self.data.identifier_columns,
                **self.drift.model_dump(exclude={"enabled"}),
            ),
        }

    @model_validator(mode="after")
    def consistent_run(self):
        if (
            not self.predictors
            or len(set(self.predictors)) != len(self.predictors)
            or set(self.predictors) != set(self.data.column_types)
        ):
            raise ValueError(
                "predictors must order every column_types feature exactly once."
            )
        if self.inputs.current_labeled and self.inputs.current is None:
            raise ValueError("current_labeled requires a current CSV.")
        if (
            self.inputs.current is not None
            and "current_labeled" not in self.inputs.model_fields_set
        ):
            raise ValueError("Explicitly set current_labeled when supplying current.")
        self.policies()  # Reuse the existing policies' cross-field validation.
        return self


def load_run_config(path: str | Path) -> RunConfig:
    """Validate a run and resolve every file relative to its YAML directory."""
    source = Path(path).resolve()
    document = load_yaml_config(source)
    raw = document.root
    try:
        config = RunConfig.model_validate(raw)

        def resolve(value):
            return (source.parent / value).resolve()

        inputs = config.inputs.model_dump()
        for name in ("train", "reference", "current"):
            if inputs[name] is not None:
                inputs[name] = resolve(inputs[name])
        resolved = config.model_dump()
        resolved["inputs"] = inputs
        resolved["artifact"]["path"] = resolve(config.artifact.path)
        resolved["output"] = resolve(config.output)
        config = RunConfig.model_validate(resolved)
        protected = [source, config.artifact.path, inputs["train"], inputs["reference"]]
        if inputs["current"] is not None:
            protected.append(inputs["current"])
        if any(
            config.output == item
            or (
                config.output.exists()
                and item.exists()
                and config.output.samefile(item)
            )
            for item in protected
        ):
            raise ValueError(
                "Report output must not collide with configuration or inputs."
            )
        config._source_sha256 = document._source_sha256
        config._source_path = source
        return config
    except (ValidationError, ValueError, OSError) as error:
        raise ConfigError(f"Invalid run configuration '{source}': {error}") from error
