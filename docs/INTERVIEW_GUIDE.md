# Interview Guide

This guide will grow with the implementation. For Phase 0, be prepared to
explain:

- why the project uses a `src` package layout;
- why runtime and development dependencies are separated;
- why YAML must be loaded with `safe_load`;
- why application code uses logging rather than scattered `print` calls; and
- the difference between local Git history and GitHub operations.

Phase 0 provides no model quality evidence by itself.

## Phase 1: plain-language explanation

We turned a published CSV into two reproducible training examples for the toolkit.
The workflow checks the exact input file, records its basic quality, splits it
once, trains two models on the same rows, and saves everything needed to inspect
their predictions. Data and models stay local; source code and measured aggregate
results are versioned.

## Main functions and data flow

- `sha256_bytes` / `sha256_file`: identify exact bytes. Whitespace changes count
  as changes; a fingerprint is not a leakage detector or a signature.
- `download_source`: fetch the approved revision, verify before writing, reuse
  an existing source only if the checksum matches, cap download size at 5 MiB.
- `read_csv_strings`: reject duplicate headers and preserve literal cell values.
- `audit_dataset`: measure missingness, duplicate counts, class counts, and
  invalid numerical values. Percentages use the full row count as denominator.
- `prepare_frame`: enforce the Telco column set, map Yes/No targets, parse numbers,
  retain missing values, validate IDs and segment presence, and sort by ID.
- `split_dataset`: produce the seed-controlled stratified train/reference split.
- `prepare_dataset`: orchestrate acquisition, preparation, audit, and manifests.
- `build_preprocessor`: construct the unfitted numerical/categorical transformations.
- `build_baselines`: create two independent pipelines using that same design.
- `load_prepared_data`: verify hashes, counts, classes, and disjoint IDs.
- `baseline_metrics`: compute descriptive class-1 metrics using reference labels
  and probabilities. At exactly 0.50 a prediction is positive.
- `_git_state`: record commit and whether local changes exist; return unknown
  state if Git is unavailable.
- `train_baselines`: fit on training predictors, score the reference data, save
  and reload each model, and write strict JSON via `write_json`.

The two scripts only parse arguments, call these functions, and summarize results.
They do not contain hidden model logic. The complete validation CLI is deferred.

## Statistical and ML concepts

- Stratification preserves approximate class proportions, not identical rows
  or distributions of every feature.
- Median imputation estimates a replacement from observed training values.
- Standard scaling subtracts the training mean and divides by its standard
  deviation; reference values never determine those statistics.
- One-hot encoding represents a category with indicators rather than imposing
  an artificial numeric ordering. `SeniorCitizen` is categorical despite 0/1
  text values. Missing categories receive a fixed marker.
- Logistic Regression estimates `p(churn) = sigmoid(b + w*x)` and minimizes a
  regularized classification loss. Coefficients are associations, not causal
  effects. `C` controls inverse regularization strength.
- Random Forest averages probabilities from randomized decision trees trained
  with bootstrapped samples and feature subsets. More complexity need not improve
  performance on a particular reference split.
- Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R). A zero predicted
  positive count makes precision undefined; our summary records zero and a warning.
- PR-AUC integrates precision versus recall over score thresholds. Average
  precision uses a different weighting rule, so both are named explicitly.

## Why these choices, and alternatives

The dataset is small enough for dense one-hot features and local CPU training.
Shared preprocessing and split membership make the comparison understandable.
Pipelines preserve the learned transformations used at inference.

Reasonable alternatives include 70/30 splitting, training-only cross-validation,
tree-specific preprocessing without scaling, class weighting, and dropping
incomplete rows. Their tradeoffs are evaluation sample size, runtime, complexity,
precision/recall behavior, and information loss. None is automatically superior.

## Failure modes and limitations

- A changed upstream file or mismatched local CSV fails checksum verification.
- A manifest is not tamper-proof; editing both content and hashes defeats that check.
- Duplicate IDs or overlapping partitions would bias evaluation, so this
  preparation/training workflow rejects them before fitting.
- The schema is dataset-specific, with no exhaustive category/range checks yet.
- Unknown categories become all-zero indicators; information is lost.
- All-missing numerical training columns use the imputer's zero fallback.
- A seed does not guarantee identical results across dependency versions or hardware.
- Joblib can execute code when loading; use only trusted artifacts.
- A random split does not model chronological drift; one split has sampling uncertainty.
- Reusing reference results for tuning compromises independent evaluation.
- Calibration, general leakage heuristics, drift, and segment checks are not built yet.

## Five likely interview questions

1. **Why split before imputing?** The imputation value is learned from data.
   Including reference values lets evaluation data influence the model pipeline.
2. **How can numerical parsing happen before splitting safely?** Parsing each
   value with a fixed rule does not estimate statistics from other rows. Median
   estimation is different and stays inside the fitted training pipeline.
3. **Why keep customerID in the CSV but out of the model?** It supports split
   tracing and overlap checks. Its arbitrary identity should not be learned as
   predictive signal in this baseline.
4. **Why is Random Forest not automatically better?** Nonlinear capacity may
   increase variance or provide little advantage. On this split it had higher
   precision but lower recall and F1; the comparison has sampling uncertainty.
5. **How do you demonstrate reproducibility?** Pin input bytes and split rules,
   record versions and parameters, retrain independently, and compare membership,
   probabilities, and metrics. Also verify model save/load preserves predictions.

## 60-second explanation

I built a reproducible dataset and training workflow for ModelGate. It downloads
IBM's Telco CSV from a pinned revision and checks its SHA-256 before saving it.
The audit found 7,043 rows and 11 blank TotalCharges cells. I encode churn as a
binary target and make a stratified 80/20 split with seed 42. Both Logistic
Regression and Random Forest use that exact split. Median imputation, scaling,
and category encoding are fitted only on training data inside each pipeline.
At threshold 0.50, their reference F1 scores were about 0.573 and 0.561. I retained
both baselines and documented the results without optimizing against the
reference set. A second run reproduced their probabilities within 1e-12.
The saved summaries record data hashes, parameters, versions, and source state.
These are baseline measurements for developing validation checks, not production
approval evidence.

## Understanding exercise

Training TotalCharges values are `[10, 20, missing, 40]`; reference values are
`[1000, missing]`. What value should replace each missing cell, which rows
determine it, and why must we avoid using the combined train/reference median?

## Git explanation

`git switch -c phase/1-data-baselines` created a local branch from verified main.
It did not publish anything. `git diff` shows tracked file changes; new files
only appear in a full proposed commit after staging. `git add` prepares the
snapshot without committing. Commit, push, and opening a pull request are separate
steps requiring the project's explicit approvals. No history is rewritten.
