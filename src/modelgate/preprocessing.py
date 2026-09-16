"""Build independent pipelines with a shared preprocessing design."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelgate.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Learn numeric statistics and category vocabulary only when fit is called."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                    keep_empty_features=True,
                ),
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_baselines(seed: int = 42) -> dict[str, Pipeline]:
    """Create unfitted models; no transformers or estimators are shared."""
    estimators = {
        "logistic_regression": LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=2000, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
    }
    return {
        name: Pipeline(
            [("preprocessor", build_preprocessor()), ("classifier", estimator)]
        )
        for name, estimator in estimators.items()
    }
