"""Validated data-check settings; not the complete YAML run configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from modelgate.schema import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    SEGMENT_COLUMN,
    TARGET_COLUMN,
)

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
Label = StrictStr | StrictInt


class DataCheckPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS
    column_types: dict[str, Literal["number", "integer", "category"]] = Field(
        default_factory=lambda: {
            **dict.fromkeys(NUMERIC_COLUMNS, "number"),
            **dict.fromkeys(CATEGORICAL_COLUMNS, "category"),
        }
    )
    target_column: str = TARGET_COLUMN
    target_labels: tuple[Label, Label] = (0, 1)
    positive_class_label: Label = 1
    identifier_columns: tuple[str, ...] = (ID_COLUMN,)
    segment_column: str = SEGMENT_COLUMN
    allowed_categories: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    strict_extra_columns: bool = False
    max_missing_fraction: Fraction = 0.05
    missing_overrides: dict[str, Fraction] = Field(default_factory=dict)
    missing_exclusions: tuple[str, ...] = ()
    max_duplicate_row_fraction: Fraction = 0.0
    max_duplicate_id_fraction: Fraction = 0.0
    max_identifier_overlap_fraction: Fraction = 0.0
    prohibited_columns: tuple[str, ...] = ()
    max_examples: int = Field(default=5, ge=0, le=20, strict=True)
    min_leakage_rows: int = Field(default=20, ge=2, strict=True)
    min_leakage_coverage: Fraction = 0.8

    @model_validator(mode="after")
    def consistent_roles(self) -> "DataCheckPolicy":
        lists = (
            self.required_columns,
            self.identifier_columns,
            self.missing_exclusions,
            self.prohibited_columns,
        )
        for names in lists:
            if len(names) != len(set(names)) or any(not name.strip() for name in names):
                raise ValueError(
                    "Column names must be nonblank and unique in each list."
                )
        if not self.identifier_columns:
            raise ValueError("At least one identifier column is required.")
        if self.target_column == self.segment_column or (
            {self.target_column, self.segment_column} & set(self.identifier_columns)
        ):
            raise ValueError("Target, segment, and identifier roles must not conflict.")
        if self.target_column in self.column_types or (
            set(self.identifier_columns) & self.column_types.keys()
        ):
            raise ValueError(
                "column_types describes predictors only, not target or IDs."
            )
        roles = {self.target_column, self.segment_column, *self.identifier_columns}
        if not (roles | self.column_types.keys()) <= set(self.required_columns):
            raise ValueError(
                "required_columns must include every configured role and type."
            )
        if self.column_types.get(self.segment_column) != "category":
            raise ValueError("The segment must be a configured categorical predictor.")
        labels = [str(value) for value in self.target_labels]
        if len(set(labels)) != 2 or any(not value.strip() for value in labels):
            raise ValueError(
                "Exactly two distinct nonblank target labels are required."
            )
        if str(self.positive_class_label) not in labels:
            raise ValueError("positive_class_label must be one of target_labels.")
        for column, values in self.allowed_categories.items():
            if self.column_types.get(column) != "category":
                raise ValueError("Allowed categories require a categorical predictor.")
            if (
                not values
                or len(values) != len(set(values))
                or any(not value.strip() for value in values)
            ):
                raise ValueError(
                    "Allowed categories must be nonempty, unique, nonblank."
                )
        missing_names = set(self.missing_overrides) | set(self.missing_exclusions)
        if not missing_names <= set(self.required_columns):
            raise ValueError("Missingness settings name an unknown column.")
        if set(self.missing_overrides) & set(self.missing_exclusions):
            raise ValueError(
                "An excluded column cannot also have a missingness override."
            )
        if self.target_column in self.prohibited_columns or (
            set(self.identifier_columns) & set(self.prohibited_columns)
        ):
            raise ValueError("Prohibited columns describe features, not target or IDs.")
        return self
