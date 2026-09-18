# Phase 3: Model validation

## Purpose and boundaries

Measure a fitted binary classifier on labeled reference data without fitting,
recalibrating, choosing a winning threshold, or issuing a release decision.
Phase 1 training output is unchanged. Segment analysis, drift, the full YAML
schema, public CLI, overall status and validation exit codes remain later work.
No dependencies were added.

## APIs and data flow

`positive_probabilities(model, features, target_labels=(0,1), positive_label=1)`
checks a fitted binary model's `classes_`, locates the positive class, calls
`predict_proba` once and validates the complete two-column matrix. Every row
must sum to one within absolute tolerance 1e-8, with no relative tolerance.
Reject nonnumeric, boolean, nonfinite and out-of-range probabilities. The
returned positive vector is copied. It never calls `fit` or `predict`.

Prepare predictors using the original fitted pipeline's input contract, exclude
target/IDs, and preserve row order. This adapter does not infer column roles or
validate arbitrary model feature schemas; the demo supplies the 19 explicit
Phase 1 predictors. Estimator execution errors propagate rather than masquerade
as a successful validation. There is no decision-function fallback.

`classification_metrics(labels, probabilities, threshold=0.5)` computes class
counts, prevalence, precision, recall, F1, predicted-positive rate, confusion
matrix, trapezoidal PR-AUC and average precision.

`probability_metrics(labels, probabilities, n_bins=10)` computes Brier score
and calibration-curve data. `threshold_analysis(labels, probabilities,
thresholds=(0.3,0.4,0.5,0.6,0.7))` computes classification measurements for each
threshold in configured order. All three accept keyword-only `target_labels`
and `positive_label` arguments. They do not alter inputs.

Inputs must be equal-length, nonempty, one-dimensional vectors. Labels are
nonblank strings or integers; booleans, floats, missing values and labels
outside the configured pair are rejected. Integer labels and their CSV string
representations are equivalent (`1` and `"1"`), so cannot form distinct classes.
Strings are not stripped or case-normalized. Probabilities must be numeric,
finite and within [0,1]; strings and booleans are rejected, never clipped.
Two pandas Series must have identical indexes. Other vectors are positional:
the caller is responsible for aligning every outcome with its prediction.

`run_model_checks(labels, probabilities, policy, dataset="reference")` returns
a JSON-ready dictionary containing `metrics`, `probability_metrics`,
`threshold_analysis` and six `checks`. Each check is validated using the existing
`CheckResult` contract before serialization. Invalid inputs raise `ValueError`;
undefined measurements on otherwise valid inputs produce explicit warnings.

## Formulas and definitions

- Positive prediction: `p >= threshold`, including exact equality.
- Precision: TP / (TP + FP).
- Recall: TP / (TP + FN).
- F1: 2 TP / (2 TP + FP + FN).
- Predicted-positive rate: (TP + FP) / N; prevalence: (TP + FN) / N.
- Confusion matrix: actual rows, predicted columns, `[[TN,FP],[FN,TP]]`.
- PR-AUC: trapezoidal area under Scikit-learn's precision-recall curve.
- Average precision: Scikit-learn's non-interpolated average precision; separately
  named because it is not the trapezoidal area.
- Binary Brier: `mean((p-y)**2)` with positive outcomes encoded as 1. Its range is
  [0,1]; perfect probabilities score 0, confidently wrong probabilities score 1.

These definitions are tested against the
[Scikit-learn metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html).
No accuracy headline, confidence intervals, significance tests or sample weights
are implemented. Comparing models on one split is descriptive, not proof that
one is generally superior.

## Undefined measurements

Precision is null with no positive predictions. Recall is null with no actual
positives. F1 is null only when `2TP+FP+FN == 0`; it can genuinely equal zero
when precision is undefined. Each null has an `undefined_reasons` entry.
The corresponding check is WARNING with `evaluated=false`, never a PASS due
to a fabricated zero. This intentionally differs from the older Phase 1
descriptive helper's documented `zero_division=0` convention; ordinary defined
measurements retain regression compatibility.

One observed outcome class produces an evaluated class-support WARNING. PR-AUC
and average precision are withheld for either one-class case. Available counts,
classification metrics, Brier and calibration bins still describe that sample,
but must not be generalized to the absent class. This is not support for a
one-class fitted model: prediction extraction requires two trained classes.

## Calibration bins

Ten equal-width bins are the default (configurable integer 2–100). First bin is
[lower,upper]; later bins are (lower,upper], using NumPy linspace edges and
searchsorted, matching Scikit-learn. Serialized edges retain their floating-point
representation. Each bin has count, mean probability and positive fraction.
Empty bins remain in the output with count 0 and null measurements. The nonempty
bin values match `sklearn.calibration.calibration_curve(strategy="uniform")`.
The JSON provides curve data, not a rendered image.

Calibration asks whether groups assigned, for example, a 70% chance experience
the outcome about 70% of the time. Sparse bins give weak evidence. A Brier score
combines calibration, discrimination and outcome uncertainty; a lower Brier
score alone does not prove better calibration. Bin choices can hide or expose
patterns. See the
[Scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
Quantile bins, uncertainty intervals and recalibration are not implemented.

## Policy semantics

`ModelCheckPolicy` rejects unknown fields, invalid label pairs, boolean or
nonfinite thresholds, out-of-range limits, empty/duplicate grids and invalid
bin counts. Decision threshold defaults to 0.5 and the five grid thresholds are
0.30, 0.40, 0.50, 0.60 and 0.70.

Optional `min_precision`, `min_recall`, `min_f1`, `min_pr_auc`, and
`max_brier_score` all default to null. Configured minima fail strictly below
the limit; maximum Brier fails strictly above. Equality passes. Comparisons use
unrounded floating values without tolerance; never copy a rounded table value
as an exact boundary test. Thresholds are policy choices, not statistical facts.

Without a limit, a defined metric is a descriptive PASS with `applied=false`,
null threshold and explicit message: **not quality approval**. Class support is
a separate PASS/WARNING check. Counts, calibration bins, average precision and
threshold-grid rows are descriptive evidence, not independent policy checks.
The chosen decision threshold is the only threshold to which metric rules apply.

Example using invented inputs and illustrative (not business) limits:

```python
from modelgate.checks.model import ModelCheckPolicy, run_model_checks

policy = ModelCheckPolicy(min_recall=0.6, max_brier_score=0.2)
report = run_model_checks([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], policy)
# Recall 0.5 fails; Brier 0.158125... passes.
```

## Trusted local demonstration

```powershell
.venv\Scripts\python scripts/demo_model_checks.py --trust-local-models
# Use an unused path for another run:
.venv\Scripts\python scripts/demo_model_checks.py --trust-local-models --output reports/generated/phase3_another.json
```

The demo verifies the preparation manifest, model training manifest and both
artifact hashes before loading either model. It records model parameters,
training seed, split metadata and fingerprints, training provenance hash,
training/evaluation environment, policy, measurements and checks. No fitting or
download occurs. Fingerprints detect changes, not malicious content: only load
trusted locally produced joblib files. A malicious file and matching manifest
are still malicious. Existing output is never overwritten.

The script exits 0 for a successfully written demonstration, 2 for handled
input/output errors. It does not implement validation-status exit codes or a
complete audit envelope. Unexpected estimator/deserialization errors may still
propagate. Generated data/models/reports remain ignored by Git.

## Evidence

The unchanged 1,409-row reference split has 374 positives. Brier was
0.13706126432822094 (Logistic Regression) and 0.13930202436298936 (Random Forest).
Both repeated local reports were identical, and direct Scikit-learn checks
confirmed the Brier and nonempty calibration-bin measurements. All twelve
checks were descriptive/class-support PASS, with no configured quality limits.
See `EXPERIMENTS.md` for full threshold results and interpretation.
