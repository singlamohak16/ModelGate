"""Small invented records; tests never download or redistribute the IBM dataset."""

import io

import pandas as pd
import pytest

from modelgate import dataset
from modelgate.fingerprint import sha256_bytes
from modelgate.schema import CATEGORICAL_COLUMNS


@pytest.fixture
def raw_telco() -> pd.DataFrame:
    records = []
    for index in range(40):
        row = dict.fromkeys(CATEGORICAL_COLUMNS, "No")
        row.update(
            {
                "customerID": f"TEST-{index:03d}",
                "gender": "Female" if index % 2 else "Male",
                "SeniorCitizen": str(index % 2),
                "Contract": "Month-to-month" if index % 3 else "One year",
                "tenure": str(index + 1),
                "MonthlyCharges": str(20 + index % 7),
                "TotalCharges": " " if index == 0 else str(25 * (index + 1)),
                "Churn": "Yes" if index % 4 == 0 else "No",
            }
        )
        records.append(row)
    return pd.DataFrame(records)


@pytest.fixture
def mock_download(monkeypatch, raw_telco):
    content = raw_telco.to_csv(index=False, lineterminator="\n").encode("utf-8")
    monkeypatch.setattr(dataset, "SOURCE_SHA256", sha256_bytes(content))
    monkeypatch.setattr(dataset, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    return content


@pytest.fixture
def check_policy():
    from modelgate.checks.policy import DataCheckPolicy

    return DataCheckPolicy(
        required_columns=("id", "amount", "visits", "group", "flag", "target"),
        column_types={
            "amount": "number",
            "visits": "integer",
            "group": "category",
            "flag": "category",
        },
        target_column="target",
        target_labels=(0, 1),
        positive_class_label=1,
        identifier_columns=("id",),
        segment_column="group",
        allowed_categories={"flag": ("A", "B")},
        min_leakage_rows=4,
    )


@pytest.fixture
def check_frame():
    return pd.DataFrame(
        {
            "id": [f"T{i}" for i in range(6)],
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "visits": [1, 2, 3, 4, 5, 6],
            "group": ["g1", "g1", "g2", "g2", "g1", "g2"],
            "flag": ["A", "A", "B", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
