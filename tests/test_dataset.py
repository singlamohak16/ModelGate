import io
import json

import pandas as pd
import pytest

from modelgate import dataset
from modelgate.dataset import (
    audit_dataset,
    download_source,
    prepare_dataset,
    prepare_frame,
    read_csv_strings,
    split_dataset,
)
from modelgate.fingerprint import sha256_file


def test_parsing_is_row_local_and_retains_missingness(raw_telco):
    original = raw_telco.copy(deep=True)
    prepared = prepare_frame(raw_telco.sample(frac=1, random_state=6))
    assert prepared.iloc[0]["customerID"] == "TEST-000"
    assert pd.isna(prepared.iloc[0]["TotalCharges"])
    assert prepared.iloc[1]["TotalCharges"] == 50.0
    assert prepared["TotalCharges"].dtype == "float64"
    assert prepared["Churn"].tolist() == [int(i % 4 == 0) for i in range(40)]
    assert prepared["SeniorCitizen"].iloc[1] == "1"
    pd.testing.assert_frame_equal(original, raw_telco)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Churn", "maybe", "Churn must contain"),
        ("Churn", " ", "Churn must contain"),
        ("customerID", "", "nonmissing and unique"),
        ("customerID", "TEST-001", "nonmissing and unique"),
        ("Contract", " ", "Contract must be nonmissing"),
        ("TotalCharges", "not-a-number", "invalid nonblank numeric"),
        ("MonthlyCharges", "inf", "invalid nonblank numeric"),
        ("tenure", "NaN", "invalid nonblank numeric"),
    ],
)
def test_invalid_cells_are_rejected(raw_telco, column, value, message):
    raw_telco.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        prepare_frame(raw_telco)


def test_schema_and_single_class_errors(raw_telco):
    with pytest.raises(ValueError, match="missing=.*Contract"):
        prepare_frame(raw_telco.drop(columns="Contract"))
    with pytest.raises(ValueError, match="unexpected=.*leaked_target"):
        prepare_frame(raw_telco.assign(leaked_target="Yes"))
    with pytest.raises(ValueError, match="both Churn classes"):
        prepare_frame(raw_telco.assign(Churn="No"))
    with pytest.raises(ValueError, match="at least one row"):
        prepare_frame(raw_telco.iloc[:0])


def test_duplicate_csv_headers_are_rejected(tmp_path):
    path = tmp_path / "duplicate_header.csv"
    path.write_text("Churn,Churn\nYes,No\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names"):
        read_csv_strings(path)


def test_audit_reports_counts_and_denominators(raw_telco):
    raw_telco.loc[1, "MonthlyCharges"] = "invalid"
    repeated = pd.concat([raw_telco, raw_telco.iloc[[0]]], ignore_index=True)
    audit = audit_dataset(repeated)
    assert audit["rows"] == 41
    assert audit["column_count"] == 21
    assert audit["missing"]["TotalCharges"]["count"] == 2
    assert audit["missing"]["TotalCharges"]["percentage"] == pytest.approx(200 / 41)
    assert audit["exact_duplicate_rows"] == audit["duplicate_ids"] == 1
    assert audit["target_counts"] == {"No": 30, "Yes": 11}
    assert audit["invalid_nonblank_numeric_counts"]["MonthlyCharges"] == 1
    json.dumps(audit, allow_nan=False)


def test_split_is_disjoint_stratified_and_order_independent(raw_telco):
    prepared = prepare_frame(raw_telco)
    train, reference = split_dataset(prepared)
    again = split_dataset(prepared.sample(frac=1, random_state=99))
    assert (len(train), len(reference)) == (32, 8)
    assert set(train.customerID).isdisjoint(reference.customerID)
    assert set(train.customerID) | set(reference.customerID) == set(prepared.customerID)
    assert train.Churn.mean() == reference.Churn.mean() == 0.25
    for first, second in zip((train, reference), again, strict=True):
        pd.testing.assert_frame_equal(first, second)
    other_train, _ = split_dataset(prepared, seed=7)
    assert set(other_train.customerID) != set(train.customerID)


def test_split_rejects_impossible_or_invalid_settings(raw_telco):
    prepared = prepare_frame(raw_telco)
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_dataset(prepared, reference_fraction=1)
    with pytest.raises(ValueError, match="stratified split"):
        split_dataset(prepared.iloc[:2])


def test_download_verifies_before_writing_and_reuses_cache(
    tmp_path, monkeypatch, mock_download
):
    path = tmp_path / "source.csv"
    assert download_source(path).read_bytes() == mock_download
    monkeypatch.setattr(
        dataset, "urlopen", lambda *args, **kwargs: pytest.fail("network used")
    )
    assert download_source(path) == path
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch for existing"):
        download_source(path)
    assert path.read_bytes() == b"corrupted"


def test_download_rejects_changed_source_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_source(tmp_path / "source.csv")
    assert not (tmp_path / "source.csv").exists()


def test_prepare_writes_reproducible_fingerprints_and_refuses_overwrite(
    tmp_path, mock_download
):
    first = prepare_dataset(tmp_path / "a")
    second = prepare_dataset(tmp_path / "b")
    assert first == second
    assert first["partitions"]["train"]["rows"] == 32
    for name in ("train", "reference"):
        path = tmp_path / "a" / "prepared" / f"{name}.csv"
        assert sha256_file(path) == first["partitions"][name]["sha256"]
        assert set(read_csv_strings(path)["Churn"]) == {"0", "1"}
    assert json.loads((tmp_path / "a/prepared/manifest.json").read_text()) == first
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_dataset(tmp_path / "a")
