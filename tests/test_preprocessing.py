import numpy as np
import pandas as pd

from modelgate.dataset import prepare_frame
from modelgate.preprocessing import build_baselines, build_preprocessor
from modelgate.schema import NUMERIC_COLUMNS


def test_statistics_use_training_rows_and_unknown_categories_do_not_refit(raw_telco):
    prepared = prepare_frame(raw_telco)
    train = prepared.iloc[:20].copy()
    reference = prepared.iloc[20:].copy()
    reference["TotalCharges"] = 1e9
    reference["Contract"] = "UNSEEN_REFERENCE_ONLY"
    preprocessor = build_preprocessor().fit(train)
    numeric = preprocessor.named_transformers_["numeric"]
    medians = np.nanmedian(train[list(NUMERIC_COLUMNS)].to_numpy(dtype=float), axis=0)
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, medians)
    expected_imputed = (
        train[list(NUMERIC_COLUMNS)]
        .fillna(pd.Series(medians, index=NUMERIC_COLUMNS))
        .to_numpy()
    )
    np.testing.assert_allclose(
        numeric.named_steps["scaler"].mean_, expected_imputed.mean(axis=0)
    )
    previous = numeric.named_steps["scaler"].mean_.copy()
    transformed = preprocessor.transform(reference)
    np.testing.assert_array_equal(previous, numeric.named_steps["scaler"].mean_)
    assert np.isfinite(transformed).all()
    contract_columns = [
        i
        for i, name in enumerate(preprocessor.get_feature_names_out())
        if name.startswith("categorical__Contract_")
    ]
    assert contract_columns
    assert (transformed[:, contract_columns] == 0).all()
    # Even if supplied, identifiers and target are dropped by explicit column selection.
    altered = reference.assign(customerID="DIFFERENT", Churn=99)
    np.testing.assert_array_equal(transformed, preprocessor.transform(altered))
    assert not any(
        "Churn" in name or "customerID" in name
        for name in preprocessor.get_feature_names_out()
    )


def test_models_do_not_share_preprocessing_instances():
    models = build_baselines()
    assert (
        models["logistic_regression"].named_steps["preprocessor"]
        is not models["random_forest"].named_steps["preprocessor"]
    )
