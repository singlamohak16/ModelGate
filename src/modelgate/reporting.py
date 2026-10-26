"""Versioned audit envelope, severity aggregation and exclusive finite JSON output."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from modelgate.results import CheckResult, Status


def aggregate_status(checks: list[CheckResult]) -> Status:
    if any(check.status == Status.FAIL for check in checks):
        return Status.FAIL
    if any(check.status == Status.WARNING for check in checks):
        return Status.WARNING
    return Status.PASS


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: str
    dataset: str
    state: Literal["evaluated", "blocked", "disabled", "not_applicable"]
    reason: str | None = None
    evaluated_checks: int = Field(default=0, ge=0)
    blocked_checks: int = Field(default=0, ge=0)


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    run_id: str
    started_at: str
    finished_at: str
    tool_version: str
    execution_state: Literal["completed", "error"]
    seed: int
    packages: dict[str, str]
    fingerprints: dict[str, JsonValue]
    effective_config: dict[str, JsonValue]
    model: dict[str, JsonValue]
    checks: list[CheckResult]
    coverage: list[Coverage]
    results: dict[str, JsonValue]
    warnings: list[dict[str, JsonValue]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    overall_status: Status | None
    exit_code: Literal[0, 1, 2]

    @model_validator(mode="after")
    def consistent_verdict(self):
        if self.execution_state == "error":
            if (
                self.overall_status is not None
                or self.exit_code != 2
                or not self.errors
            ):
                raise ValueError(
                    "Execution errors require null verdict, code 2 and errors."
                )
        else:
            expected = aggregate_status(self.checks)
            if not self.checks or self.errors or self.overall_status != expected:
                raise ValueError(
                    "Completed reports require checks and their aggregate verdict."
                )
            if self.exit_code != (1 if expected == Status.FAIL else 0):
                raise ValueError("Exit code disagrees with completed verdict.")
        json.dumps(self.model_dump(mode="python"), allow_nan=False)
        return self


def write_report(report: AuditReport, destination: str | Path) -> None:
    """Serialize before touching the output; never overwrite an existing path.

    An OS write failure can leave a partial file. It is not a valid audit and the
    CLI returns 2; the file is retained for inspection, not silently removed.
    """
    validated = AuditReport.model_validate(report.model_dump(mode="python"))
    payload = json.dumps(validated.model_dump(mode="json"), indent=2, allow_nan=False)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload + "\n")
