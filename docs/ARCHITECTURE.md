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
