import json

import joblib
import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from modelgate.dataset import prepare_dataset, write_json
from modelgate.fingerprint import sha256_file
from modelgate.schema import FEATURE_COLUMNS
from modelgate.training import baseline_metrics, load_prepared_data, train_baselines


def test_metrics_match_hand_counts_and_sklearn():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.35, 0.8])
    metrics = baseline_metrics(y_true, probabilities)
    predicted = probabilities >= 0.5
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["precision"] == precision_score(y_true, predicted) == 1
    assert metrics["recall"] == recall_score(y_true, predicted) == 0.5
    assert metrics["f1"] == f1_score(y_true, predicted) == pytest.approx(2 / 3)
    assert metrics["predicted_positive_rate"] == 0.25
    assert metrics["pr_auc"] == pytest.approx(19 / 24)
    assert metrics["average_precision"] == average_precision_score(
        y_true, probabilities
    )
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    json.dumps(metrics, allow_nan=False)


def test_threshold_includes_exactly_half_and_reports_undefined_precision():
    assert baseline_metrics([0, 1], [0.1, 0.5])["confusion_matrix"] == [[1, 0], [0, 1]]
    metrics = baseline_metrics([0, 1], [0.1, 0.2])
    assert metrics["precision"] == 0
    assert metrics["warnings"]


@pytest.mark.parametrize(
    "labels,probabilities",
    [([0, 1], [0.1]), ([0, 1], [0, np.nan]), ([0, 1], [0, 2]), ([0, 0], [0.2, 0.3])],
)
def test_metrics_reject_invalid_input(labels, probabilities):
    with pytest.raises(ValueError):
        baseline_metrics(labels, probabilities)


def test_training_is_reproducible_and_artifacts_roundtrip(tmp_path, mock_download):
    prepare_dataset(tmp_path / "data")
    prepared_dir = tmp_path / "data/prepared"
    first = train_baselines(prepared_dir, tmp_path / "run1")
    second = train_baselines(prepared_dir, tmp_path / "run2")
    _, reference, manifest = load_prepared_data(prepared_dir)
    assert first["data"] == manifest
    assert first["seed"] == 42
    assert first["environment"]["packages"]["scikit-learn"]
    assert set(first["models"]) == {"logistic_regression", "random_forest"}
    for name, record in first["models"].items():
        assert (
            record["reference_metrics"] == second["models"][name]["reference_metrics"]
        )
        assert record["roundtrip_predictions_verified"] is True
        assert not record["training_warnings"]
        paths = [tmp_path / run / f"{name}.joblib" for run in ("run1", "run2")]
        assert record["artifact"]["sha256"] == sha256_file(paths[0])
        probabilities = [
            joblib.load(path).predict_proba(reference[list(FEATURE_COLUMNS)])
            for path in paths
        ]
        np.testing.assert_allclose(*probabilities, rtol=0, atol=1e-12)
        assert np.isfinite(probabilities[0]).all()
        np.testing.assert_allclose(probabilities[0].sum(axis=1), 1)
    assert json.loads((tmp_path / "run1/baseline_metrics.json").read_text()) == first
    with pytest.raises(FileExistsError, match="Training output exists"):
        train_baselines(prepared_dir, tmp_path / "run1")


def test_training_rejects_changed_files_before_creating_outputs(
    tmp_path, mock_download
):
    prepare_dataset(tmp_path / "data")
    directory = tmp_path / "data/prepared"
    path = directory / "reference.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Fingerprint mismatch"):
        train_baselines(directory, tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_training_rejects_overlapping_partitions_even_with_updated_hash(
    tmp_path, mock_download
):
    prepare_dataset(tmp_path / "data")
    directory = tmp_path / "data/prepared"
    train, reference, manifest = load_prepared_data(directory)
    reference.loc[0, "customerID"] = train.iloc[0]["customerID"]
    path = directory / "reference.csv"
    reference.to_csv(path, index=False, lineterminator="\n")
    manifest["partitions"]["reference"]["sha256"] = sha256_file(path)
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="overlap on 1 customer IDs"):
        load_prepared_data(directory)
