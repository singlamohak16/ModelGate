# Architecture

## Phase 0 boundaries

ModelGate uses a `src` package layout so tests and users import the installed
package rather than accidentally importing code from the repository root.

The current package has two small boundaries:

1. `modelgate.config` reads YAML safely and validates that its root is a
   string-keyed mapping.
2. `modelgate.logging` configures one reusable package logger without adding a
   duplicate handler when called repeatedly.

## Phase 1 data flow

1. `scripts/prepare_data.py` parses development arguments and calls
   `dataset.prepare_dataset`.
2. `download_source` retrieves the approved revision, checks SHA-256 before
   writing, and reuses an existing raw file only when its checksum matches.
3. `read_csv_strings` preserves literal CSV values and checks duplicate headers.
   `audit_dataset` records observed quality measurements.
4. `prepare_frame` applies the schema in `schema.py`: column roles, numerical
   parsing, target mapping, identifier checks, and stable ID ordering. No
   statistical preprocessing is fitted here.
5. `split_dataset` produces deterministic stratified training/reference CSVs.
   `fingerprint.py` hashes file bytes; the preparation manifest records source
   and split provenance.
6. `scripts/train_baselines.py` calls `training.train_baselines`, which verifies
   partition fingerprints, class presence, counts, and ID separation.
7. `preprocessing.build_baselines` creates independent pipelines sharing a
   design: ColumnTransformer -> classifier. Explicit predictor columns exclude
   target and IDs. Each pipeline fits only on training rows.
8. Class-1 reference probabilities feed `baseline_metrics`; fitted pipelines
   and descriptive JSON metadata are written to an unused local output folder.

The five Phase 1 modules separate schema, bytes/provenance, data preparation,
pipeline construction, and training. The scripts are intentionally thin.
The shared `dataset.write_json` helper enforces strict JSON; it is not yet a
general audit-report model.

## Current boundaries

The prepared CSVs still contain IDs and `Contract` for later validation, while
model fitting receives the explicit 19 predictors. The full fitted pipeline is
saved, including imputation/scaling statistics and category vocabularies.

Byte hashes detect accidental input changes, including whitespace and newline
changes. A manifest is local provenance, not a signed authenticity certificate:
someone changing both a file and its hash can bypass the checksum comparison.

Training summaries include actual timestamps, package versions, Git commit and
dirty state, model parameters, and artifact hashes. Before committing this
phase, the source commit identifies the Phase 0 parent with `dirty=true`.
Do not interpret that parent commit as containing the new training code.

Output directories are never overwritten. A failed run may leave partial output;
inspect it and use a new directory for retry. Data/model files remain ignored.

The YAML loader remains the Phase 0 skeleton. The full YAML schema, public
validation CLI, structured check results, and status aggregation arrive in
later phases. Baseline metric recording here does not implement Phase 3's
calibration, threshold analysis, or release rules.

## Phase 2 check library

`run_data_checks(train, reference, policy, current=None, current_labeled=False)`
returns a flat list of validated `CheckResult` objects. It does not load models,
fit anything, alter input frames, or aggregate an overall decision.

- `results.py`: status enum, result fields, finite-JSON and unevaluated-status rules.
- `checks/policy.py`: settings, defaults, and cross-field validation. It reuses
  the Telco schema but accepts another explicit CSV schema within binary scope.
- `checks/_common.py`: common blank detection, scalar conversion, prerequisites,
  positional examples, and structured ID/predictor keys.
- `checks/data.py`: schema, per-column missingness, row/ID duplicates, and two
  distinct overlap checks.
- `checks/leakage.py`: configured prohibitions and limited two-value target checks.
- `checks/runner.py`: deterministic orchestration for each dataset and the
  train/reference pair; independent checks continue after a reported defect.
- `scripts/demo_data_checks.py`: a small local demonstration of the library
  results, with overwrite protection. It is not the Phase 6 validation CLI.

Use `read_csv_strings` before these checks. Phase 1's `prepare_frame` deliberately
raises on malformed data, so using it first would prevent the checker from
describing those defects. CSV parsing errors, including duplicate raw headers,
remain input errors in the existing loader. Duplicate DataFrame column labels
are a structural FAIL when the frame is passed directly to the library.

Schema failures and dependency warnings are separate evidence. For example,
missing identifiers fail schema columns and block ID overlap; predictor overlap
can still run. Unlabeled current data has no required target; its dependent
target-like heuristic is explicitly unevaluated rather than silently passing.

The full JSON run envelope, YAML settings integration, user-facing validation
CLI, and overall-status/exit-code policy remain later-phase work.

## Phase 3 model validation

The data-check library remains independent. A caller supplies aligned labels
and positive-class probabilities to the model-check library:

`fitted pipeline + ordered predictors -> positive_probabilities -> run_model_checks`

- `predictions.py` validates binary classes, selects the configured positive
  probability column and validates both probability columns. It never fits.
- `metrics.py` validates vectors, computes confusion counts and metrics, Brier,
  uniform calibration bins and the five-threshold grid. Policy-free functions
  can later be reused by segment/drift analysis without implementing those phases.
- `checks/model.py` validates model policy and applies optional minimum metrics
  and maximum Brier rules. Existing `CheckResult` ensures finite JSON and visible
  unevaluated WARNINGs; the returned dictionary contains serialized checks.
- `scripts/demo_model_checks.py` verifies prepared data and artifact provenance,
  requires explicit local-artifact trust, evaluates both saved models and writes
  an exclusive-create JSON demonstration. Training files are unchanged.

Undefined metrics are null with reasons, distinct from legitimate zero scores.
No-limit PASS measurements explicitly say descriptive only. Invalid vectors
raise input errors; one observed class warns and blocks PR summaries but not
all available measurements. The adapter requires two trained model classes.
No threshold is optimized; no probabilities are recalibrated. The demo uses
the fitted Phase 1 preprocessing contract and does not replace Phase 2 auditing
of malformed data. See `MODEL_VALIDATION.md` for the complete API contract.
