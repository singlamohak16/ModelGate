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
It does **not** yet provide the general validation engine or
`modelgate validate` command.

ModelGate will produce evidence-based `PASS`, `WARNING`, and `FAIL` results.
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
entry points, not the future validation-status CLI.

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
The complete YAML schema, public CLI, overall status, and validation exit codes
are still Phase 6 work. The demo's success exit only means its output was written.

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

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Dataset source and schema](docs/DATASET.md)
- [Data validation rules and evidence](docs/DATA_VALIDATION.md)
- [Model validation and threshold analysis](docs/MODEL_VALIDATION.md)
- [Decisions](docs/DECISIONS.md)
- [Build log](docs/BUILD_LOG.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Interview guide](docs/INTERVIEW_GUIDE.md)

## Licence

ModelGate source code is available under the [MIT License](LICENSE). Dataset
licensing must be evaluated separately before any data is downloaded or
committed.
