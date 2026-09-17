"""Small check results, independent of the future run-level audit report."""

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Status(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class CheckResult(BaseModel):
    """A measurement and its policy decision; unevaluated never means passed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    check_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    status: Status
    evaluated: bool
    measurement: dict[str, JsonValue]
    threshold: dict[str, JsonValue]
    evidence: dict[str, JsonValue]
    message: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> "CheckResult":
        if not self.evaluated and self.status != Status.WARNING:
            raise ValueError("An unevaluated check must have WARNING status.")
        # JsonValue checks shape; this rejects non-standard JSON floating values.
        json.dumps(self.model_dump(mode="python"), allow_nan=False)
        return self
