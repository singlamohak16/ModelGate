"""Reproducible acquisition and preparation of the one supported Telco dataset."""

import csv
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelgate.fingerprint import sha256_bytes, sha256_file
from modelgate.schema import (
    CATEGORICAL_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAPPING,
)

SOURCE_REVISION = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{SOURCE_REVISION}/data/Telco-Customer-Churn.csv"
)
SOURCE_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024


def download_source(destination: str | Path) -> Path:
    """Download the pinned source; verify before writing or reuse verified bytes."""
    destination = Path(destination)
    if destination.exists():
        if sha256_file(destination) != SOURCE_SHA256:
            raise ValueError(f"Checksum mismatch for existing source: {destination}")
        return destination
    with urlopen(SOURCE_URL, timeout=60) as response:
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Source exceeds the 5 MiB download limit.")
    if sha256_bytes(content) != SOURCE_SHA256:
        raise ValueError("Downloaded source checksum mismatch; no dataset was written.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a file appearing during the download.
    with destination.open("xb") as stream:
        stream.write(content)
    return destination


def read_csv_strings(path: str | Path) -> pd.DataFrame:
    """Read literal text, detecting duplicate headers before pandas can rename them."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV header is empty or contains duplicate column names.")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _normalize_blanks(frame: pd.DataFrame) -> pd.DataFrame:
    # Preserve nonblank category spellings; only whitespace-only cells become missing.
    return frame.replace(r"^\s*$", np.nan, regex=True)


def audit_dataset(raw: pd.DataFrame) -> dict:
    """Describe the raw CSV; duplicate counts exclude each group's first occurrence."""
    normalized = _normalize_blanks(raw)
    count = len(raw)
    missing = normalized.isna().sum()
    invalid_numeric = {}
    for column in NUMERIC_COLUMNS:
        if column in normalized:
            numeric = pd.to_numeric(normalized[column], errors="coerce")
            invalid_numeric[column] = int(
                (normalized[column].notna() & ~np.isfinite(numeric)).sum()
            )
    return {
        "rows": count,
        "columns": list(raw.columns),
        "column_count": len(raw.columns),
        "storage_dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
        "inferred_types": {
            column: pd.api.types.infer_dtype(normalized[column], skipna=True)
            for column in raw.columns
        },
        "missing": {
            column: {
                "count": int(missing[column]),
                "percentage": float(missing[column] / count * 100) if count else 0.0,
            }
            for column in raw.columns
        },
        "target_counts": (
            {
                str(key): int(value)
                for key, value in normalized[TARGET_COLUMN]
                .value_counts(dropna=False)
                .items()
            }
            if TARGET_COLUMN in raw
            else {}
        ),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_ids": (
            int(normalized[ID_COLUMN].dropna().duplicated().sum())
            if ID_COLUMN in raw
            else None
        ),
        "duplicate_definition": (
            "Occurrences after the first; exact rows before normalization"
        ),
        "invalid_nonblank_numeric_counts": invalid_numeric,
    }


def prepare_frame(raw: pd.DataFrame, *, encoded_target: bool = False) -> pd.DataFrame:
    """Parse individual rows; learned imputation belongs in the fitted pipeline."""
    if raw.empty:
        raise ValueError("Dataset must contain at least one row.")
    if not raw.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    extra = sorted(set(raw.columns) - set(REQUIRED_COLUMNS))
    if missing or extra:
        raise ValueError(
            f"Telco schema mismatch: missing={missing}, unexpected={extra}"
        )
    frame = _normalize_blanks(raw.loc[:, list(REQUIRED_COLUMNS)].copy())
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("customerID must be nonmissing and unique.")
    if frame["Contract"].isna().any():
        raise ValueError("Contract must be nonmissing for the baseline dataset.")
    mapping = {"0": 0, "1": 1} if encoded_target else TARGET_MAPPING
    target = frame[TARGET_COLUMN]
    if target.isna().any() or not target.isin(mapping).all():
        raise ValueError(
            f"Churn must contain only {list(mapping)} with no missing values."
        )
    frame[TARGET_COLUMN] = target.map(mapping).astype("int64")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Dataset must contain both Churn classes.")
    for column in NUMERIC_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & ~np.isfinite(numeric)
        if invalid.any():
            raise ValueError(
                f"{column} has {int(invalid.sum())} invalid nonblank numeric values."
            )
        frame[column] = numeric.astype("float64")
    # Object strings plus np.nan match sklearn's missing-value handling.
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(object)
    return frame.sort_values(ID_COLUMN, kind="stable").reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame, *, seed: int = 42, reference_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return disjoint, ID-sorted stratified partitions with both target classes."""
    if not 0 < reference_fraction < 1:
        raise ValueError("reference_fraction must be strictly between 0 and 1.")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("Split requires unique nonmissing customerID values.")
    ordered = frame.sort_values(ID_COLUMN, kind="stable")
    try:
        train, reference = train_test_split(
            ordered,
            test_size=reference_fraction,
            random_state=seed,
            stratify=ordered[TARGET_COLUMN],
        )
    except ValueError as error:
        raise ValueError(f"Cannot create the stratified split: {error}") from error
    if any(part[TARGET_COLUMN].nunique() != 2 for part in (train, reference)):
        raise ValueError("Both partitions must contain both target classes.")
    return tuple(
        part.sort_values(ID_COLUMN).reset_index(drop=True)
        for part in (train, reference)
    )


def write_json(path: str | Path, document: dict) -> None:
    """Write strict JSON with deterministic key order and no NaN or Infinity."""
    Path(path).write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def prepare_dataset(output_dir: str | Path, *, seed: int = 42) -> dict:
    """Produce the pinned CSV, prepared partitions, audit, and provenance manifest."""
    output_dir = Path(output_dir)
    prepared_dir = output_dir / "prepared"
    if prepared_dir.exists():
        raise FileExistsError(
            f"Prepared directory already exists: {prepared_dir}. "
            "Choose a new --output-dir."
        )
    source = download_source(output_dir / "raw" / "Telco-Customer-Churn.csv")
    raw = read_csv_strings(source)
    audit = audit_dataset(raw)
    prepared = prepare_frame(raw)
    train, reference = split_dataset(prepared, seed=seed)
    prepared_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "manifest_version": 1,
        "source": {
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        },
        "seed": seed,
        "split": {
            "strategy": "stratified_random",
            "reference_fraction": 0.2,
            "ordering": "customerID ascending before and after split",
        },
        "target_mapping": TARGET_MAPPING,
        "partitions": {},
    }
    for name, partition in (("train", train), ("reference", reference)):
        path = prepared_dir / f"{name}.csv"
        partition.to_csv(path, index=False, lineterminator="\n")
        manifest["partitions"][name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(partition),
            "positive_count": int(partition[TARGET_COLUMN].sum()),
            "positive_prevalence": float(partition[TARGET_COLUMN].mean()),
        }
    audit["prepared_dtypes"] = {
        column: str(dtype) for column, dtype in prepared.dtypes.items()
    }
    write_json(prepared_dir / "audit.json", audit)
    write_json(prepared_dir / "manifest.json", manifest)
    return manifest
