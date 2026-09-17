import json

import pytest
from pydantic import ValidationError

from modelgate.results import CheckResult, Status


def result_arguments():
    return dict(
        check_id="example",
        dataset="reference",
        status=Status.PASS,
        evaluated=True,
        measurement={"fraction": 0.0},
        threshold={"maximum_fraction": 0.05},
        evidence={"examples": []},
        message="Measured.",
    )


def test_result_roundtrip_is_strict_json():
    result = CheckResult(**result_arguments())
    document = json.loads(result.model_dump_json())
    assert document["status"] == "PASS"
    assert document["measurement"]["fraction"] == 0
    assert CheckResult.model_validate(document) == result


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurements_are_rejected(value):
    arguments = result_arguments()
    arguments["measurement"] = {"nested": {"values": [value]}}
    with pytest.raises(ValidationError):
        CheckResult(**arguments)


def test_blocked_check_cannot_pass_or_fail():
    arguments = result_arguments()
    arguments["evaluated"] = False
    for status in ("PASS", "FAIL"):
        with pytest.raises(ValidationError, match="unevaluated"):
            CheckResult(**{**arguments, "status": status})
    assert CheckResult(**{**arguments, "status": "WARNING"}).evaluated is False


def test_unrecognized_result_field_is_rejected():
    with pytest.raises(ValidationError):
        CheckResult(**result_arguments(), hidden_status="PASS")
