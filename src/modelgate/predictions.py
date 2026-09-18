"""Read binary probabilities from an already fitted classifier; never fit it."""

import numpy as np

from modelgate.metrics import label_token, validate_labels


def positive_probabilities(model, features, *, target_labels=(0, 1), positive_label=1):
    tokens, positive = validate_labels(target_labels, positive_label)
    classes = np.asarray(getattr(model, "classes_", None), dtype=object)
    if classes.ndim != 1 or len(classes) != 2:
        raise ValueError("A fitted binary model with two classes_ is required.")
    class_tokens = [label_token(value) for value in classes]
    if len(set(class_tokens)) != 2 or set(class_tokens) != set(tokens):
        raise ValueError("Model classes do not match the configured labels.")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError(
            "Model must support predict_proba; scores are not probabilities."
        )
    if not len(features):
        raise ValueError("Prediction features must be nonempty.")
    output = model.predict_proba(features)
    matrix = np.asarray(output)
    if (
        matrix.shape != (len(features), 2)
        or matrix.dtype.kind not in "iuf"
        or any(
            isinstance(value, (bool, np.bool_))
            for value in np.asarray(output, dtype=object).flat
        )
    ):
        raise ValueError("predict_proba must return a numeric (n_rows, 2) matrix.")
    if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("Model probabilities must be finite and in [0, 1].")
    if not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Binary probability rows must sum to one (atol=1e-8).")
    return matrix[:, class_tokens.index(positive)].astype(float, copy=True)
