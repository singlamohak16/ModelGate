# Decision Log

## D001 — Use a `src` package layout

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Place importable code under `src/modelgate`.
- **Reason:** This makes tests exercise the installed package and reduces the
  risk of imports succeeding only because the repository root is on Python's
  path.
- **Alternatives:** A flat `modelgate/` package is simpler initially but gives
  weaker protection against packaging mistakes.

## D002 — Use setuptools as the build backend

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Use setuptools with configuration in `pyproject.toml`.
- **Reason:** It is mature, conventional, and sufficient for this small Python
  package without adding a separate environment-management workflow.
- **Alternatives:** Hatchling offers a smaller modern build configuration;
  Poetry combines building and dependency management but adds concepts the
  project does not currently need.

## D003 — Use Pydantic and PyYAML for configuration

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Parse YAML with `yaml.safe_load` and use Pydantic to enforce a
  string-keyed mapping at the document root.
- **Reason:** Safe parsing plus typed validation provides a clear path to the
  complete configuration model planned for Phase 6.
- **Alternatives:** Standard dataclasses would avoid Pydantic but require custom
  validation and error formatting; JSON is stricter but does not meet the YAML
  requirement.

## D004 — Use standard-library logging

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Configure a named `modelgate` logger using Python's built-in
  logging package.
- **Reason:** It is adequate for an offline CLI and avoids an unnecessary
  runtime dependency.
- **Alternatives:** structlog and Loguru provide richer structured or ergonomic
  APIs but are not justified in the foundation.

## D005 — Recommend the MIT licence for source code

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Licence ModelGate's original source under MIT.
- **Reason:** MIT is short, permissive, and common for portfolio tooling.
- **Limitation:** This decision does not grant rights to any dataset. Dataset
  provenance and licensing require separate review and approval in Phase 1.

## D006 — Pin IBM data and keep it local

- **Date:** 2026-09-15
- **Status:** Accepted with explicit user source approval
- **Decision:** Download the official IBM CSV at a fixed Git revision and verify
  its SHA-256 before writing. Keep raw/prepared data and model artifacts ignored.
- **Reason:** Revision and checksum pinning make input changes detectable, while
  avoiding redistribution of data whose separate licence grant was not verified.
- **Alternatives:** A Kaggle mirror adds another source and possibly account
  setup; committing the CSV would simplify offline setup but republishes it.
- **Limitation:** The first download still needs the upstream service. See
  `DATASET.md` for the precise licence findings.

## D007 — Use an ID-ordered, stratified 80/20 split

- **Date:** 2026-09-15
- **Status:** Accepted
- **Decision:** Reject duplicate/missing IDs, sort by ID, split with seed 42 and
  target stratification, then sort each partition by ID again.
- **Reason:** Both classifiers receive the same membership; ordering of incoming
  rows cannot change the split. Stratification preserves approximate prevalence.
- **Alternatives:** A 70/30 split provides more evaluation examples; training-only
  cross-validation provides a less split-dependent estimate but is deferred.
- **Limitation:** A single random split is not temporal validation. Do not tune
  against its reference results; future tuning belongs inside the training data.

## D008 — Fit all statistical preprocessing inside each pipeline

- **Date:** 2026-09-15
- **Status:** Accepted
- **Decision:** Fixed cell parsing precedes splitting. Learned numerical median
  imputation/scaling and categorical vocabulary fitting use training rows only.
  Each model has an independent ColumnTransformer and classifier.
- **Reason:** This prevents reference statistics from leaking into preprocessing
  and preserves a reusable prediction artifact. IDs and targets are excluded by
  explicit feature selection. Contract remains a predictor and segment field.
- **Alternatives:** Dropping incomplete rows loses examples; tree-specific
  preprocessing omits unnecessary scaling but complicates the initial comparison.
- **Tradeoff:** Dense one-hot arrays are easy to inspect for this small dataset;
  this is not a design for arbitrary high-cardinality datasets. Unknown categories
  become all-zero indicators; an all-missing numerical training column uses the
  imputer's zero fallback, not an estimated median.

## D009 — Record fixed, untuned baseline metrics

- **Date:** 2026-09-15
- **Status:** Accepted
- **Decision:** Train Logistic Regression (`lbfgs`, C=1, max_iter=2000) and Random
  Forest (200 trees, min_samples_leaf=2, n_jobs=1), with seed 42 and no class
  weighting. Record threshold-0.50 metrics without imposing a success score.
- **Reason:** These create a reproducible starting point for validation tooling.
- **Alternatives:** Class weighting can trade precision for recall; parameter
  search could improve fit but requires training-only evaluation and extra scope.
- **Measurement:** PR-AUC means trapezoidal integration of the precision-recall
  curve. Average precision is recorded separately using sklearn's noninterpolated
  calculation. Neither is calculated from thresholded predictions.
- **Limit:** These descriptive metrics do not implement Phase 3's calibrated
  model checks or threshold sweep.

## D010 — Protect outputs and record the actual environment

- **Date:** 2026-09-15
- **Status:** Accepted
- **Decision:** Reject existing prepared/training output folders. Save full
  pipelines with joblib, parameters, data/artifact hashes, versions, timestamps,
  and Git revision/dirty state. Record tested dependency versions in
  `constraints-python312.txt`.
- **Reason:** Repeated runs remain separately inspectable; changed input files
  cannot silently reuse a preparation manifest.
- **Alternatives:** Automatic replacement risks losing prior results; MLflow
  could manage runs but is explicitly scheduled after the core works.
- **Limit:** A failed run can leave partial outputs. Model hashes and timestamps
  need not match between runs; compare predictions and metrics instead. A seed
  alone does not guarantee cross-version/platform reproducibility. A joblib
  artifact is executable deserialization and must only be loaded when trusted.

## D011 — Use argparse only for the development scripts

- **Date:** 2026-09-15
- **Status:** Accepted
- **Decision:** Thin `prepare_data.py` and `train_baselines.py` scripts use
  standard-library argparse. The final validation CLI choice stays in Phase 6.
- **Reason:** Two small commands need no additional dependency.
- **Alternatives:** Typer provides richer CLI ergonomics; a notebook helps
  exploration but is less direct for reproducible command-driven runs.

## D012 — Separate measurements from data-check policy

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** Return validated `CheckResult` objects containing measurement,
  threshold, status, evidence, and evaluation state. Keep policy in a small
  Pydantic settings object, with full YAML/run orchestration deferred to Phase 6.
- **Reason:** Multiple independent defects should be inspectable in one run.
  Unexpected programming errors still propagate rather than appearing as results.
- **Alternatives:** Raising on every violation is simpler but hides later defects;
  using Pandera adds schema capabilities and another dependency to explain.

## D013 — Define missingness and strict threshold boundaries

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** Null, empty, and whitespace-only values are missing; other strings
  remain literal. Fractions use total dataset rows, and values strictly greater
  than the configured maximum fail. Default feature missingness maximum is 5%.
- **Reason:** Clear denominators and equality behavior make results testable.
  Target, ID, and segment requirements are independent of missingness exclusions.
- **Alternatives:** Treating every common NA token as missing could erase legitimate
  categories; using `>=` would fail data exactly at a documented maximum.
- **Limit:** The 5% default is an illustrative policy, not a statistically derived
  or universally acceptable amount of missing data.

## D014 — Keep identity and predictor overlap separate

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** ID columns form a composite tuple. Duplicate occurrences after
  the first and matching reference rows have separate measurements. Missing ID
  rows are excluded and reported; they cannot produce a fully clean ID result.
  Predictor matches exclude IDs/target and warn instead of failing automatically.
- **Reason:** Two customers can share all recorded attributes. Structured keys
  avoid delimiter collisions, and Decimal numeric keys avoid float rounding of
  distinct numerical text values. No Python hash is treated as proof of equality.
- **Alternatives:** Fuzzy matching adds similarity thresholds and false positives;
  making every predictor match a failure would overstate evidence of contamination.
- **Observed:** The Telco split had 10 matching reference predictor rows and zero
  overlapping IDs; an independent pandas merge confirmed this distinction.

## D015 — Use bounded, explainable leakage heuristics

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** Prohibited feature presence fails its explicit policy. Exact
  two-value target copies, inversions, and recodings warn. Scan unexpected
  features too, excluding target and ID roles. Require at least 20 comparable
  rows, 80% coverage, and both target classes; report inadequate evidence clearly.
- **Reason:** This catches simple suspicious relationships without fitting another
  predictive model or confusing unique IDs with a target-derived feature.
- **Alternatives:** A single-feature classifier could identify more complex
  relationships but needs separate split/scoring rules; broad correlation rules
  can miss categorical recodings and introduce arbitrary cutoffs.
- **Limits:** Minimum size/coverage are safeguards, not significance tests. Noisy,
  high-cardinality, multi-feature, temporal, or externally introduced leakage may
  pass. A legitimate binary predictor may warn. Neither outcome proves leakage
  or its absence.

## D016 — Make unevaluated checks visible

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** A missing prerequisite produces WARNING with `evaluated=false`
  and a reason. Structural schema problems fail independently. Unavailable
  measurements are null, not zero. No overall status is computed in Phase 2.
- **Reason:** A blocked check cannot be mistaken for successful validation.
- **Alternatives:** A separate SKIP status is expressive but would extend the
  requested three-status contract; silently omitting checks would hide coverage.
- **Limit:** Unlabeled current data has an explicit unevaluated leakage warning,
  even though absent labels are allowed by its schema.

## D017 — Separate model measurement from configured decisions

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** Pure metric helpers plus a model-check policy; no default quality
  limits. Retain trapezoidal PR-AUC and separately named average precision.
  Minimum checks use >=, maximum Brier uses <=, without rounding/tolerance.
- **Reason:** Requirements should not be invented or tuned to make a baseline
  pass. Measurements can later serve segments without duplicating the formulas.
- **Alternatives:** Hard-coded quality limits hide assumptions; replacing Phase 1
  helpers would needlessly alter its recorded undefined-metric convention.
- **Limit:** Descriptive PASS needs its `applied=false` evidence and message to
  prevent misinterpretation. No overall release judgment exists.

## D018 — Explicit nulls and binary prediction contracts

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** Undefined metrics return null plus reasons and unevaluated
  WARNINGs. One observed class warns and withholds PR summaries. Select the
  positive column from two fitted `classes_`, validate the entire probability
  matrix, reject scores/booleans/nonfinite values and require row sums near one.
- **Reason:** Undefined precision is not the same as precision zero. Column 1
  is not necessarily the positive class; silent clipping hides upstream defects.
- **Alternatives:** Scikit-learn's zero-division fallback is convenient but can
  mislead policy evaluation; automatically applying a sigmoid assumes a score
  scale that the toolkit cannot justify.
- **Limit:** Generic arrays have no row identities. The caller must preserve
  alignment; matching Series indexes only catches some ordering mistakes.

## D019 — Descriptive calibration and threshold analysis

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** Binary Brier on [0,1], ten equal-width calibration bins with
  explicit empty bins, and threshold grid 0.30/0.40/0.50/0.60/0.70 using >=.
  Match Scikit-learn binning and test its nonempty values directly.
- **Reason:** Small, explainable formulas and counts expose the evidence. Brier
  measures probability error, not calibration alone. Counts show sparse bins.
- **Alternatives:** Quantile bins balance counts but change probability ranges;
  threshold optimization or recalibration would require separate tuning data
  and an objective. Neither is implemented.
- **Limit:** Single-split estimates, binning sensitivity, no uncertainty bounds,
  no business-cost estimate and no guarantee of future performance.

## D020 — Preserve local artifact trust and provenance

- **Date:** 2026-09-17
- **Status:** Accepted
- **Decision:** Demo requires `--trust-local-models`, verifies both artifact
  hashes before loading either, matches training/preparation manifests, records
  evaluation versions and never overwrites an output report.
- **Reason:** joblib can execute code. A fingerprint detects changed bytes, not
  trustworthiness. The phase uses existing trusted local artifacts without fitting.
- **Alternatives:** Retraining inside evaluation blurs responsibilities; a safer
  serialization format may be useful later but adds scope and dependencies.
- **Limit:** A malicious artifact plus a matching malicious manifest remains
  unsafe. The library helper's caller is responsible for establishing trust.

## D021 — Reuse pooled binary metrics for one-column segments

- **Date:** 2026-09-19
- **Status:** Accepted
- **Decision:** Compute group metrics with Phase 3's existing helpers and one
  shared threshold; compute overall metrics directly from all rows. Compare
  precision/recall/F1/PR-AUC via signed segment-minus-comparator differences.
- **Reason:** Averaging segment F1 or PR-AUC need not reproduce pooled results.
  One shared operating point keeps comparisons interpretable.
- **Alternatives:** Separate group models/thresholds require a new decision policy
  and tuning data; relative gaps are unstable when the denominator is near zero.
- **Limit:** Gaps are descriptive and group prevalence/difficulty can differ;
  no causal, significance or fairness claim follows.

## D022 — Explicit sample, class and comparison support

- **Date:** 2026-09-19
- **Status:** Accepted
- **Decision:** Default minimum 50 rows, configurable as a positive integer.
  Below minimum, retain counts/prevalence but withhold performance with WARNINGs.
  Use only size-eligible, metric-defined groups for each best comparison and
  retain all exact ties. Warn when fewer than two candidates exist for a metric.
- **Reason:** A tiny perfect group should not become a benchmark, and a zero
  self-gap should not look like evidence that groups perform equally well.
- **Alternatives:** Displaying all small-group metrics increases available detail
  but risks overinterpretation; bootstrap intervals could quantify uncertainty
  but require additional statistical choices outside this phase.
- **Limit:** Fifty rows is illustrative, not a statistical guarantee. Few positive
  outcomes can remain even in large groups; counts and class-support warnings
  do not replace uncertainty estimates.

## D023 — Preserve missing-group coverage and optional requirements

- **Date:** 2026-09-19
- **Status:** Accepted
- **Decision:** Missing/blank segment values fail completeness and stay in the
  overall denominator, but not named comparisons. Accept string categories
  without trimming nonblank names; no sentinel-string bucket or silent coercion.
  Apply optional shared minimum metrics, with strict unrounded comparisons and
  equality passing. Defaults remain descriptive, not invented quality requirements.
- **Reason:** Silently dropping rows changes the evaluation population. A literal
  category called 'missing' must not collide with actual missing values.
- **Alternatives:** A separate missing-value group can be informative, but would
  conflate completeness defects with meaningful segment definitions; per-group
  quality overrides increase policy complexity without a current requirement.
- **Limit:** Numeric category codes require explicit conversion to intended string
  labels. Gap measurements do not themselves trigger failure. Unconfigured zero
  recall can be a descriptive PASS, so messages and applied flags matter.
