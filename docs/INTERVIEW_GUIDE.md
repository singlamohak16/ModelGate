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

## Phase 2: what was built

ModelGate now describes data problems using consistent check results. It checks
schema, missing values, duplicate rows and IDs, train/reference overlap, and
simple suspicious target relationships. The functions read DataFrames without
modifying them, and independent checks continue when another reports a defect.

## Phase 2 functions and flow

1. `DataCheckPolicy` validates settings and column-role consistency before data
   checks run. Its defaults reference the Telco schema; it is not the complete
   YAML run configuration.
2. `run_data_checks` calls schema, missingness, duplicates, and leakage checks
   for train/reference and optional current data, then train/reference overlap.
3. `check_schema` reports structure, columns, numerical/category rules, target
   encoding and class counts, identifiers, and segment presence.
4. `check_missingness` counts missing cells and applies a per-column override
   or default. Exclusions retain their measurement but do not apply a limit.
5. `check_duplicates` counts occurrences after the first, separately for whole
   rows and complete composite identifier tuples.
6. `check_overlap` counts matching reference rows, separately by identity and
   by exact predictor tuples. Multiple training matches do not inflate the count.
7. `check_leakage` checks prohibited feature names, then exact two-value
   relationships to a valid binary target with adequate sample size/coverage.
8. Helpers in `_common.py` define missingness, target tokens, blocked outcomes,
   positional evidence, and comparison keys. `CheckResult` validates result
   fields and forbids nonfinite JSON measurements or unevaluated PASS results.
9. `demo_data_checks.py` demonstrates serialization locally. It does not aggregate
   a release decision or implement the future validation CLI exit codes.

## Phase 2 methods and alternatives

Most checks use counts, proportions, set membership, and exact equality. A
duplicate fraction counts repetitions after the first divided by all dataset
rows. Overlap divides matched reference rows by all reference rows. The leakage
test examines a two-by-two value relationship, not a p-value or causal inference.
Two distinct feature values must map one-to-one to two target classes.

The 5% missingness default and minimum leakage sample/coverage are policy choices,
not statistical guarantees. Exact maximum equality passes; examples are bounded
but total counts always cover the evaluated population.

Alternatives include Pandera for schema checks, stopping at the first failure,
fuzzy record matching, and a single-feature predictive model for leakage. These
trade dependency/implementation complexity, audit completeness, and false-positive
behavior. The selected approach keeps measurements straightforward to explain.

## Phase 2 limitations and failure modes

- Bad numerical text is distinct from a genuinely missing value; converting it
  silently to missing before validation would hide the parsing defect.
- Missing IDs are excluded from key matching and reported separately. They do
  not establish zero overlap for those unmatchable rows.
- Identical predictors across different IDs can be legitimate coincidences.
- Exact matching misses modified duplicates; target recoding checks miss noisy
  and multi-feature leakage. Legitimate strong binary features may warn.
- A check blocked by missing columns cannot pass. Its WARNING plus the schema
  FAIL explains why coverage is incomplete.
- Current labels must be declared available by the caller. Unlabeled current data
  is not assigned model-performance or degradation claims.
- A PASS applies to one configured check. It is not proof of release readiness.

## Phase 2: five interview questions

1. **What is the difference between a measurement and a rule?** Missingness of
   3% is observed data; whether it passes a 5% maximum is a configured decision.
2. **How do you avoid double-counting overlap?** Build the set of complete
   training keys, then count each reference row whose key belongs to that set.
   Repeated training keys cannot inflate the numerator.
3. **Why does predictor overlap warn instead of fail?** Different customers
   can share all recorded attributes. Our Telco split has 10 such matches but
   zero overlapping IDs, so matching attributes alone cannot prove contamination.
4. **Does a target-copy warning prove leakage?** No. It detects a suspicious
   relationship. We must investigate feature provenance and whether it would
   have been available at the intended prediction time.
5. **Why report unevaluated checks?** Missing prerequisites mean there is no
   measurement. Reporting zero or PASS would falsely claim that validation ran
   successfully. We retain a reason and `evaluated=false` instead.

## Phase 2: 60-second explanation

I added reusable data checks to ModelGate. Each result records a measurement,
threshold, PASS/WARNING/FAIL status, and bounded evidence. The policy defines
column roles and limits, while checks measure schema problems, missingness,
duplicates, and train/reference overlap. IDs use composite tuples, and overlap
counts each reference row once. Predictor equality warns because separate
customers can share attributes. Leakage checks identify prohibited columns and
exact two-value target relationships, with clear limits on what they can prove.
If a prerequisite is missing, the dependent check is explicitly unevaluated.
On the Telco split, 135 checks passed and one warned about 10 identical predictor
rows across the split, with no ID overlap. I independently verified that count
and retained the warning rather than changing the data to make every check pass.

## Phase 2 understanding exercise

Training IDs are `[A, A, B]`; reference IDs are `[A, A, C, missing]`. What are
the training duplicate-ID count, the overlapping reference-row count, and the
overlap fraction using the documented denominator? How should the missing ID
appear in the evidence?
# Phase 3 interview readiness

## Plain-language explanation

I added a way to evaluate a saved churn model without retraining it. It reports
correct predictions, missed churners, false alarms, probability error and how
those outcomes change across five decision thresholds. It preserves evidence
when a metric cannot be calculated instead of pretending it passed.

## Technical explanation

`positive_probabilities` maps fitted `classes_` to the configured positive label
and validates the probability matrix. The metric functions compute confusion
counts, classification metrics, PR summaries, Brier and calibration bins.
`run_model_checks` applies an independently validated policy and creates existing
`CheckResult` objects before serializing them. The demo verifies data/model
fingerprints, loads trusted artifacts, evaluates both models and writes JSON.
It never fits preprocessing or a classifier on reference data.

The smaller helpers have explicit jobs: `label_token` validates/canonicalizes
supported label representations; `validate_labels` checks the two-class contract;
`validate_fraction` validates probability thresholds; `validate_inputs` checks
alignment/ranges and encodes outcomes. `_classification` converts thresholded
probabilities to TP/FP/TN/FN. The public helpers add PR summaries, Brier or bins.

## Statistics and design reasoning

Precision is TP/(TP+FP); recall is TP/(TP+FN); F1 is 2TP/(2TP+FP+FN).
Predicted-positive rate is (TP+FP)/N, distinct from prevalence (TP+FN)/N.
PR-AUC describes ranking across thresholds, whereas F1 describes a selected
operating threshold. Trapezoidal PR area and average precision use different
integration conventions, so both are labeled explicitly.

Brier averages (p-y)^2. Predicting 0.8 when the outcome is 1 contributes 0.04;
predicting 0.8 when the outcome is 0 contributes 0.64. Calibration asks whether
groups assigned similar probabilities show similar observed event rates.
Brier also reflects discrimination and uncertainty; it is not pure calibration.

Manual confusion counts are small and explainable, and independent Scikit-learn
comparisons guard correctness. Equal-width bins preserve interpretable ranges;
quantile bins would balance counts but change those ranges. Zero-division
fallbacks would simplify numbers but hide undefined evidence. Automatic
threshold selection would need a separate tuning set and an explicit objective.

## Limitations and failure modes

One split cannot establish generalization, causal value or business savings.
Sparse calibration bins are unstable. Repeatedly choosing settings based on the
reference set would turn it into tuning data. Misaligned arrays can produce
plausible but wrong metrics. A model's probability column order may differ from
assumptions, which is why `classes_` is checked. Fingerprints do not make an
untrusted joblib artifact safe. No-limit PASS means descriptive only.

## Five interview questions

1. **Why not rely on accuracy?** It can conceal missed churners under imbalance.
   Precision/recall expose false-alarm and missed-positive tradeoffs.
2. **Does lower Brier prove better calibration?** No. It measures overall
   probability error and also reflects discrimination and outcome uncertainty.
3. **What happens when no positives are predicted?** Precision is undefined and
   reported as null with a warning. If actual positives exist, recall and F1
   genuinely equal zero.
4. **Why not choose threshold 0.30 from this table?** It is an observed operating
   point, not an optimized deployment choice. Selection needs costs/objectives,
   tuning data and an untouched evaluation set.
5. **How do you trust the computations?** Small hand-calculated fixtures,
   Scikit-learn agreement, boundary/invalid-input tests, real-pipeline integration,
   unchanged fingerprints and identical repeated local reports.

## 60-second explanation

ModelGate evaluates saved binary churn models offline. In Phase 3 I added
precision, recall, F1, PR-AUC, confusion counts and probability evaluation using
Brier score and calibration bins. It evaluates five thresholds, exposing the
tradeoff between catching churners and generating false alarms. I select the
positive probability column from the model's class labels and keep undefined
metrics as explicit warnings. Rules are configurable, separate from measurement,
and there are no invented default quality limits. I tested the calculations
against Scikit-learn and evaluated both existing pipelines on the unchanged
reference data. Logistic Regression's Brier was about 0.1371 and Random Forest's
about 0.1393. These are single-split observations, not deployment approval or
proof that either model is better calibrated.

## Understanding exercise

For labels `[0,0,1,1]` and probabilities `[0.1,0.4,0.35,0.8]`, calculate TP,
FP, TN, FN, precision and recall at thresholds 0.50 and 0.30. Which errors
increase when the threshold drops? Does Brier change, and why?
