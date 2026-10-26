# Phase 6: configuration, CLI and JSON reporting

One offline command evaluates one trusted fitted binary classifier. It never
fits preprocessing/model parameters, retrains, deploys, or approves production
use. The approved real examples remain the local Telco LR/RF pipelines.

## Running and rerunning

```powershell
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\modelgate validate --config configs/example.yaml
.venv\Scripts\modelgate validate --config configs/random_forest.yaml
```

`python -m modelgate validate --config ...` calls the same function. `--help`
returns 0; invalid arguments return 2. An activated environment makes the command
available as `modelgate`. No Typer/Click dependency is needed: argparse supplies
the one required subcommand and helpful usage errors.

Each output must be new. For another run, copy an example to
`configs/local-rerun.yaml` and edit its `output`, e.g.
`../reports/generated/phase6_rerun.json`. Local configurations and generated
reports are ignored. Do not delete old evidence merely to make a command succeed.
Both data and model artifacts must already exist; validation downloads nothing.

## Configuration contract

The YAML root and nested settings reject unknown keys. Duplicate YAML keys,
including duplicate merged keys, are errors rather than silently last-write-wins.
Paths resolve relative to the configuration directory. Output cannot equal an
input/model/config path; existing hard-link aliases are also rejected. An existing
output is refused before model loading, with exclusive creation as a second guard.

| Section | Settings |
| --- | --- |
| `inputs` | Required `train`, `reference`; optional `current`; `current_labeled` must be explicitly true/false when current is supplied. True without current is invalid. |
| `artifact` | Required `path`; `trust` defaults false; optional lowercase 64-character `expected_sha256`. |
| Root | Required `output`; `seed` integer defaults 42; `decision_threshold` defaults 0.5; ordered `predictors` defaults to the 19 Telco features. |
| `data` | Authoritative required columns, predictor types, target labels/positive label, IDs and segment; all Phase 2 settings described below. Core data checks cannot be disabled. |
| `model` | `enabled`, threshold grid, calibration bins, optional minimum precision/recall/F1/PR-AUC and maximum Brier. |
| `segment` | `enabled`, minimum segment size and optional common precision/recall/F1/PR-AUC minima. |
| `drift` | `enabled`, minimum sample size, smoothing, bounded examples, optional maximum numerical KS/categorical distance/prediction KS. |
| `performance` | `enabled`, minimum sample size, optional maximum precision/recall/F1/PR-AUC drops and Brier increase. |

All optional families default enabled; an absent current sample makes comparison
families not applicable. Unlabeled current data never produces labeled model,
segment, or performance-change claims, even if a target-like column is present.
Disabled settings are still validated, preventing latent invalid policies.

The authoritative `data` section supports:

- `required_columns`, `column_types` (`number`, `integer`, `category`),
  `target_column`, exactly two `target_labels`, `positive_class_label`,
  `identifier_columns`, `segment_column`, and `allowed_categories`.
- `strict_extra_columns`, `max_missing_fraction`, `missing_overrides`,
  `missing_exclusions`, `max_duplicate_row_fraction`,
  `max_duplicate_id_fraction`, `max_identifier_overlap_fraction`.
- `prohibited_columns`, `max_examples`, `min_leakage_rows`,
  `min_leakage_coverage`.

See [data rules](DATA_VALIDATION.md) for denominators, boundary behavior and
defaults. `predictors` must contain every `column_types` key exactly once and no
IDs/target; a segment is a categorical predictor. Roles/labels are not duplicated
in model/segment/performance settings. The loader builds their existing policies
from the common roles and decision threshold. Optional limits default to null,
meaning descriptive measurements, not an implicit quality guarantee.

Example comparison settings (add to a copied example):

```yaml
inputs:
  train: ../data/prepared/train.csv
  reference: ../data/prepared/reference.csv
  current: ../data/prepared/current.csv
  current_labeled: false
drift:
  max_numerical_ks: 0.2
  max_categorical_distance: 0.2
  max_prediction_ks: 0.2
```

These limits are illustrative, not statistically calibrated recommendations.
No real current dataset has been supplied. Do not describe a copy of reference
as new temporal evidence. With `current_labeled: true`, the configured labels
must be valid; invalid declared labels fail data checks rather than silently
switching to unlabeled mode. Metric-drop limits require actual labeled evidence
to be evaluated; otherwise they are recorded as not applicable.

## Data flow and dependency behavior

1. `load_run_config` validates YAML, resolves paths, checks collisions and stores
   the SHA256 of the exact bytes parsed. The low-level `load_yaml_config` mapping
   API remains for Phase 0 compatibility; it is not sufficient to validate a run.
2. `read_audit_csv` hashes and parses the same bytes as literal UTF-8 CSV values.
   Leading-zero IDs, literal `NA`, blank cells and embedded newlines survive.
   Duplicate/blank headers remain data-schema defects. Ragged/unparseable CSV
   files are input errors, not repaired data. Files are loaded into memory.
3. `run_data_checks` audits raw train/reference/current frames before conversion.
   It never uses Phase 1 `prepare_frame`, which sorts IDs and rejects defects
   before they can become audit evidence.
4. Only after explicit trust and optional hash verification is the joblib byte
   snapshot deserialized. Identity, classes and parameters are recorded. Complex
   sklearn parameter objects use a clearly labeled repr/type, not a false promise
   of reconstructing an estimator from JSON. Pickle/joblib is executable content;
   a hash identifies bytes, not trust or a safe dependency environment.
5. `prediction_features` makes an ordered predictor-only copy, normalizes blank
   cells to missing, and parses validated numeric values. Row order/index remains
   unchanged. Missing/invalid predictors block inference; missing IDs do not.
   When present, fitted `feature_names_in_` must exactly match configured order.
6. `positive_probabilities` is called once per needed dataset on the same model;
   the positive class is selected by label, not assumed column 1. Predictions
   are reused by model, segment, drift and performance checks. No `.fit` occurs.
7. Missing labels block labeled checks but not valid feature/prediction drift.
   A missing numerical column can block numerical/prediction comparisons while
   an independent categorical drift check still runs. A model execution failure
   stops the run with error 2; it is not downgraded to a WARNING.
8. The report aggregates check severity, validates finite JSON and is created
   exclusively. Output writing is separate from measurement calculations.

## Verdict and exit codes

| Execution | Overall status | Exit |
| --- | --- | --- |
| Completed, no warnings/failures | PASS | 0 |
| Completed, warning(s), no failures | WARNING | 0 |
| Completed, any failed rule | FAIL | 1 |
| Config/input/model/report execution error | null (no completed verdict) | 2 |

FAIL overrides WARNING, which overrides PASS. Zero is **not** warning-free and
does not mean production approval. Undefined precision remains null/WARNING,
never an invented zero. Drift-limit exceedance warns; configured labeled quality
or deterioration violations fail, retaining earlier phase semantics.

Examples: duplicate IDs in a readable CSV => 1; missing predictor => 1 plus
blocked dependent checks; missing file => 2; bad `predict_proba` output => 2.
An error report retains evidence collected before the error when a valid output
is known/writable. Invalid configuration may produce no report. A report write
failure returns 2 on stderr; no valid error report can be promised at a failing
destination. An OS failure during writing can leave a partial file; inspect it
and choose a fresh output. Exclusive creation prevents overwrite but is not a
transactional/atomic publication guarantee.

## JSON schema version 1

`AuditReport` requires run ID, start/end UTC timestamps, tool/package versions,
declared seed, file fingerprints, resolved effective configuration, model
identity/parameters, `checks`, `coverage`, `results`, `warnings`, `errors`,
execution state, overall status, and exit code. The seed is configuration
provenance only; it does not prove which seed trained an arbitrary artifact.
The Phase 1 manifest/training summary remains the source of split/training
provenance; this generic runner does not require that project-specific manifest.

Each check retains measurement, threshold, status, evidence and message. Checks
may share a rule ID across columns/groups. Use array position plus dataset and
evidence column/segment to locate one instance; warnings include `check_index`.
Coverage distinguishes evaluated, blocked, disabled and not applicable. Family
coverage includes evaluated/blocked check counts; partial families can be
evaluated while some individual checks remain unevaluated. Undefined metrics
also count as blocked/unevaluated, with their precise reason in the check.
An execution error yields partial coverage, not a claim all families ran.

`results` stores model metrics/calibration/grid and segment/performance details;
drift measurements live in `checks`. Missing current comparisons do not receive
invented measurements. Schema validation rejects inconsistent verdict/exit
combinations, completed empty audits and NaN/Infinity. `write_report` revalidates
before creating even the destination directory.

Reports can contain local paths, identifiers and limited examples. They are local
and ignored; inspect/redact before sharing. Fingerprints do not establish
authenticity, privacy, representativeness, no leakage, or fairness. Runs differ
in IDs/timestamps; compare measurements rather than whole JSON bytes.

## Tests and limitations

Focused tests cover valid/invalid YAML, common roles, path resolution, collisions,
trust/hash ordering, no-fit inference, row/feature order, partial dependencies,
absent/labeled/unlabeled current modes, all exit codes, finite serialization,
overwrite races, write errors, and the module entry point. They fit only a tiny
invented fixture locally and need no IBM data or network. The full defect
scenario generator remains Phase 7; hosted CI remains Phase 8.

Data/model files should remain stable while loaded; each dataset/model is a
consistent byte snapshot but the entire run is not a transactional snapshot of
all files. Large CSVs/models may exhaust memory. Report structure is versioned,
but nested measurement payloads retain the existing check contracts rather than
being individually separate versioned schemas. This is educational offline
evidence, not a hardened untrusted-model service.
