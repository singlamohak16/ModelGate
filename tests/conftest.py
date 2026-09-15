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
