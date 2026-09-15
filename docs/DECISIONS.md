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
