# ModelGate

ModelGate is an educational, offline toolkit for validating the data,
predictions, and performance of scikit-learn binary-classification models
before a potential release.

The project is being built incrementally. Phase 1 provides reproducible IBM
Telco data preparation and Logistic Regression / Random Forest baselines.
Phase 2 adds a library of evidence-based data checks for schema, missingness,
duplicates, train/reference overlap, and limited leakage heuristics.
Phase 3 adds model metrics, Brier score, calibration bins, threshold analysis,
and optional model-quality rules for fitted binary classifiers.
Phase 4 adds one-column segment evaluation, sample-size warnings, overall/best
comparisons and optional shared segment-quality limits.
Phase 5 adds feature/prediction drift and explicitly labeled performance-change
checks, without inferring performance loss from unlabeled data.
Phase 6 connects these checks through validated YAML, the `modelgate validate`
command, versioned JSON reports, and explicit validation/error exit codes.
Phase 7 adds deterministic synthetic controls that verify four deliberately
injected defects through that same CLI with both model pipelines.
Phase 8 adds a CI workflow, subprocess-aware coverage gates, distribution checks
and an isolated installed-CLI smoke test. Local verification is recorded;
the first hosted GitHub Actions run is pending an approved push.

ModelGate produces evidence-based `PASS`, `WARNING`, and `FAIL` results.
It will not automatically approve, reject, retrain, or deploy a model, and its
checks will not guarantee the absence of leakage, bias, or production risk.

## Development setup

ModelGate requires Python 3.11 or newer.
Phase 1 was tested with CPython 3.12.14 on Windows 11. Use Python 3.12 to
reproduce the recorded environment; other versions are not yet CI-tested.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

For the exact installed dependency versions used for the baseline results,
add `-c constraints-python312.txt` to the last command. This pins package
versions, not the operating system, BLAS implementation, or build tools.

Run the checks (tests use invented fixtures and never need a dataset download):

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
```

## Prepare and train the baselines

First review [dataset provenance and licensing](docs/DATASET.md). The source is
an IBM-hosted sample, pinned to an immutable revision and SHA-256. The CSV and
model artifacts stay local and are excluded from Git.

From the repository root, after installing the package:

```powershell
.venv\Scripts\python scripts/prepare_data.py
.venv\Scripts\python scripts/train_baselines.py
```

Preparation needs internet access for the first download. Training runs offline.
Preparation writes `data/raw/Telco-Customer-Churn.csv` and the following files
under `data/prepared/`: `train.csv`, `reference.csv`, `audit.json`, `manifest.json`.
The target is encoded as `No=0`, `Yes=1` in both prepared CSVs; numerical blanks
remain missing until training-only pipeline imputation.

Training writes `logistic_regression.joblib`, `random_forest.joblib`, and
`baseline_metrics.json` under `models/baseline/`. Each artifact contains fitted
preprocessing and its classifier; inputs must follow the documented prepared
schema. Only load joblib artifacts you trust: deserialization can execute code.

Output directories are protected against overwrite. For another run:

```powershell
.venv\Scripts\python scripts/train_baselines.py --output-dir models/repeat
# To prepare another independent split using the same approved source:
.venv\Scripts\python scripts/prepare_data.py --output-dir data/repeat --seed 42
```

Choose unused output directories if these already exist. A partial directory
from an interrupted run is also protected. `--help` documents each script.
The scripts exit 0 on success and 2 for handled input errors; they are development
entry points, separate from the Phase 6 validation-status CLI.

## Measured baseline results

On 2026-09-15, a seed-42 stratified 80/20 split produced 5,634 training and 1,409
reference rows. At probability threshold 0.50:

| Model | Precision | Recall | F1 | PR-AUC (trapezoidal) |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6352 | 0.5214 | 0.5727 | 0.6597 |
| Random Forest | 0.6525 | 0.4920 | 0.5610 | 0.6398 |

Both models reproduced reference probabilities within absolute tolerance
`1e-12` in a second run in the same environment. These are single-split baseline
measurements, not production, temporal, fairness, or calibration evidence. See
[full results and limitations](docs/EXPERIMENTS.md).

## Run the Phase 2 data-check demonstration

After preparing the data, run:

```powershell
.venv\Scripts\python scripts/demo_data_checks.py
```

This writes individual check results to ignored
`reports/generated/phase2_data_checks.json`. Choose another `--output` path if
the file already exists. The library API is `run_data_checks(train, reference,
policy)`; see [Data validation](docs/DATA_VALIDATION.md) for settings and semantics.
The Phase 6 CLI below adds complete YAML, overall status and validation exit
codes. This earlier demo's success exit only means its output was written.

On 2026-09-17, the prepared Telco split produced **135 PASS, 1 WARNING, 0 FAIL**
individual results. The warning identified 10 reference rows (0.7097%) whose
predictors match training rows. There was zero identifier overlap. Identical
attributes can belong to different customers; the warning is not proof of
contamination or a reason to modify the split.

## Run the Phase 3 model-check demonstration

After the existing preparation and training steps:

```powershell
.venv\Scripts\python scripts/demo_model_checks.py --trust-local-models
```

Use the trust flag only for artifacts from your own trusted local training run;
joblib loading can execute code. The demonstration checks dataset and artifact
fingerprints, evaluates both saved pipelines without fitting, and writes ignored
`reports/generated/phase3_model_checks.json`. Choose a fresh `--output` on reruns.

On the unchanged 1,409-row reference split, Brier scores were **0.137061** for
Logistic Regression and **0.139302** for Random Forest. Ten calibration bins and
the five thresholds 0.30–0.70 are included. No quality limits are configured in
this descriptive demo: its PASS measurements do not imply release approval.
Brier is probability error, not a pure measure of calibration. See
[model validation](docs/MODEL_VALIDATION.md) for APIs, rules, edge cases and limits.

## Run the Phase 4 segment-check demonstration

```powershell
.venv\Scripts\python scripts/demo_segment_checks.py --trust-local-models
```

This evaluates the trusted saved baselines by `Contract`, with a shared 0.50
decision threshold and illustrative minimum segment size of 50. It writes ignored
`reports/generated/phase4_segment_checks.json`; use a new `--output` if it exists.

On the unchanged reference split, **both models had zero recall for one-year
and two-year contracts** at 0.50. Neither predicted positives in those groups, so
precision was undefined rather than zero. Each model produced 20 PASS and
3 WARNING results with no configured quality limits. These are descriptive
measurements, not release approval or a fairness verdict. See
[segment validation](docs/SEGMENT_VALIDATION.md) for rules and limitations.

## Run the Phase 5 drift controls

```powershell
.venv\Scripts\python scripts/demo_drift_checks.py --trust-local-models
```

This is deliberately **reference versus an exact copy of itself**, not a new
current dataset or temporal evaluation. Both saved models run in labeled and
unlabeled modes. The local report is `reports/generated/phase5_controls.json`;
choose a fresh `--output` if it exists. All 80 drift distances and ten available
metric changes were zero; four class-support checks passed. Unlabeled mode
produced no performance metrics or checks. These 94 descriptive/control PASS
results do not establish real-world stability or quality approval.

Invented shifted fixtures separately verify numerical KS, categorical
Jensen-Shannon distance, prediction KS and labeled deterioration detection.
See [drift rules, smoothing and limitations](docs/DRIFT_VALIDATION.md).
SciPy is now a declared direct dependency; its recorded installed version remains
1.18.1. No retraining, new data download or production monitoring was added.

## Run the Phase 6 validation CLI

After local preparation/training, refresh the editable installation to register
the command (no new runtime dependency was added):

```powershell
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\modelgate validate --config configs/example.yaml
.venv\Scripts\modelgate validate --config configs/random_forest.yaml
```

Equivalent module entry point:

```powershell
.venv\Scripts\python -m modelgate validate --config configs/example.yaml
```

Paths inside YAML resolve relative to that YAML, not the terminal directory.
Outputs are never overwritten: for a rerun, copy the example to an ignored
`configs/local-*.yaml` file and change its `output` to a fresh path. The examples
explicitly trust only your locally trained artifacts; do not load downloaded or
untrusted joblib files. SHA256 verification does not make an artifact safe.

The unchanged Telco baselines each produce **161 PASS, 4 WARNING, 0 FAIL**, overall
WARNING and exit **0**. Warnings retain the known predictor overlap, undefined
precision for one-/two-year contracts, and limited precision comparison support.
No quality limit was invented to claim release readiness. Current data is absent:
drift/performance comparison is **not applicable**, not a measured zero.

Exit **0** means completed without failed rules (PASS or WARNING); **1** means
completed with failed rules; **2** means configuration/input/execution/report-write
error. An error report has a null overall verdict, not a new validation status.
See [CLI, configuration and report contracts](docs/CLI_REPORTING.md).
The synthetic scenario framework is described below. CI and MLflow remain
Phases 8–9.

## Run the Phase 7 controlled experiments

No IBM download or existing model artifacts are needed:

```powershell
.venv\Scripts\python scripts/run_experiments.py --output-dir artifacts/phase7
```

If that directory exists, choose a new name, such as `artifacts/phase7-repeat`.
The runner refuses existing directories and never overwrites earlier evidence.
It creates invented Telco-shaped fixtures, fits the existing LR/RF pipelines
once each on clean synthetic training data, then runs ten real CLI subprocesses.
All generated CSVs, models, configurations, manifests and reports stay ignored.
Your actual Telco data and fitted baselines are not read or changed.

| Control (both models) | Measured result | Audit status / exit |
| --- | --- | --- |
| Clean | Overall F1 and both segment recalls = 1.00; self-comparison drift/change = 0 | PASS / 0 |
| Contamination | Exactly 20/1,000 reference IDs overlap training | FAIL / 1 |
| Target leakage | Exact target-copy column identified, excluded from predictions | WARNING / 0 |
| Feature drift | Shifted tenure KS = 1.00; other feature distances = 0 | WARNING / 0 |
| Segment degradation | Selected segment recall = 0.00; overall F1 = 0.90 still meets 0.85 | FAIL / 1 |

The runner returns **0 when all expected findings match**, including deliberately
failed audits; **1** for unexpected findings and **2** for execution errors.
Read `summary.json` for expected/observed assertions and individual reports for
the underlying evidence. These easy synthetic controls demonstrate software
behavior, not realistic churn accuracy, detection rates or production readiness.
See [controlled experiments](docs/CONTROLLED_EXPERIMENTS.md) for transformations,
fixed illustrative thresholds, reproduction and limitations.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Dataset source and schema](docs/DATASET.md)
- [Data validation rules and evidence](docs/DATA_VALIDATION.md)
- [Model validation and threshold analysis](docs/MODEL_VALIDATION.md)
- [Segment validation and comparisons](docs/SEGMENT_VALIDATION.md)
- [Drift and labeled performance changes](docs/DRIFT_VALIDATION.md)
- [CLI, configuration and JSON reporting](docs/CLI_REPORTING.md)
- [Controlled experiments and expected results](docs/CONTROLLED_EXPERIMENTS.md)
- [Testing, coverage and CI](docs/TESTING_CI.md)
- [Decisions](docs/DECISIONS.md)
- [Build log](docs/BUILD_LOG.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Interview guide](docs/INTERVIEW_GUIDE.md)

## Licence

ModelGate source code is available under the [MIT License](LICENSE). Dataset
licensing must be evaluated separately before any data is downloaded or
committed.
