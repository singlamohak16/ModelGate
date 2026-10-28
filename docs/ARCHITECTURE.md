# Architecture

## Phase 0 boundaries

ModelGate uses a `src` package layout so tests and users import the installed
package rather than accidentally importing code from the repository root.

The initial package had two small boundaries:

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

## Phase 1 boundaries

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

At Phase 1, YAML was still a mapping skeleton. Structured checks were added in
Phases 2–5 and the full run schema/CLI arrived in Phase 6 below. Baseline metric
recording remains separate from calibration, threshold analysis and rule checks.

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

Phase 6 below connects this library to YAML, the CLI, a full report envelope
and overall-status/exit-code policy.

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

## Phase 4 segment validation

`ordered reference labels + positive probabilities + one segment column`
feed `analyze_segments`, then `run_segment_checks` applies an independent policy.

- `segments.py`: alignment and string-category validation, overall measurements,
  observed-group masks, size eligibility, class counts/prevalence, Phase 3 group
  metrics, per-metric best candidates/ties, signed differences and row coverage.
- `checks/segment.py`: validated settings and existing `CheckResult` creation for
  completeness, comparison support, sample size, class support and optional
  minimum precision/recall/F1/PR-AUC. Returns serialized checks with the evidence.
- `scripts/demo_segment_checks.py`: verifies trusted local data/artifacts,
  predicts once per model, runs Contract checks and exclusively creates JSON.
  No changes were needed in existing model training or Phase 3 metric code.

Small groups retain counts/prevalence but no performance measurements; missing
segment rows remain in overall metrics and are counted explicitly. Best values
are metric-specific and exclude small/undefined results. A single candidate's
self-gap is not evidence of parity, so comparison support warns independently.
Each group's rule is identified by `check_id` plus `evidence.segment`.

Data checks, model checks and segment checks remain separate library APIs. Their
integration into full YAML, one CLI and an overall audit status is in Phase 6.
The Phase 4 implementation did not introduce drift or an experiment framework.

## Phase 5 comparisons

- `drift.py`: validated finite/probability vectors, numerical usable-sample
  profiles, direct empirical KS statistic, typed categorical support and smoothed
  base-2 Jensen-Shannon distance. KS is cross-checked against SciPy; categorical
  distance uses SciPy directly. No p-value is computed for rule decisions.
- `checks/drift.py`: validates feature roles and drift policy; iterates independent
  feature checks plus a probability check. Blocked features preserve reasons
  without suppressing valid other features. Returns `CheckResult` objects.
- `checks/performance.py`: explicit labeled-mode gate, same-threshold metrics on
  both populations, signed changes, direction-aware deterioration limits and
  class-support warnings. Returns serialized checks and descriptive measurements.
- `scripts/demo_drift_checks.py`: verified trusted artifacts and reference-copy
  controls in labeled/unlabeled modes. Records 100% reference/current overlap;
  does not misrepresent copied data as a new deployment population.

Prediction extraction is still the Phase 3 helper. Comparison functions accept
probabilities and require callers to establish common model/provenance; they
cannot prove that provenance from arrays. Series align to their own frame only,
not across independent datasets. Optional absent current data creates explicit
unavailable evidence. No-label mode returns no performance claims at all.

Drift warnings are separate from configured labeled-deterioration failures.
The full application-level connection among data/model/segment/drift checks and
overall-status/exit-code decisions is in Phase 6 below. Small shifted tests in this
phase are not the Phase 7 controlled-experiment framework.

## Phase 6 application boundary

`CLI -> validated config -> raw data audit -> trusted fitted inference -> checks
-> aggregate verdict -> validated JSON report -> process exit code`.

- `config.py`: mapping loader plus full nested Pydantic run schema, duplicate-key
  rejection, common role/threshold policy adapters, relative-path resolution,
  collision guards and a fingerprint of the exact parsed YAML bytes.
- `validation.py`: raw CSV snapshot parser, predictor-only conversion without
  sorting/dropping rows, explicit dependency gating, one prediction per needed
  dataset, model provenance and reuse of unchanged check calculations.
- `reporting.py`: schema-version-1 envelope, coverage, finite JSON validation,
  severity aggregation and exclusive output creation.
- `cli.py` and `__main__.py`: argparse entry point shared by installed `modelgate`
  and `python -m modelgate`. Exit 0 includes warnings, 1 indicates failed rules,
  and 2 denotes execution failure with no completed validation verdict.
- `configs/example.yaml` and `random_forest.yaml`: local model examples, no
  current data invented, no default model quality limit invented.
- `tests/test_cli_reporting.py`: focused application integration/serialization
  tests on invented fixtures, independent of dataset downloads or stored models.

The raw CLI loader intentionally differs from Phase 1's `read_csv_strings`:
duplicate/blank headers survive to a schema FAIL instead of failing parsing.
Ragged or unparseable CSV remains an execution error. `prepare_frame` is never
used by validation because it sorts IDs and prematurely rejects audit defects.
Prediction features are copies; only the already-fitted pipeline imputes/scales.

Families can be evaluated, blocked, disabled or not applicable. Partial-family
coverage retains evaluated/unevaluated counts and per-check reasons. Independent
feature drift continues when predictions cannot be made. Unexpected execution
exceptions stop the run, retain partial evidence where possible, and return 2.
The writer cannot guarantee a valid report on disk failure; it never overwrites
an existing destination. See [the full contract](CLI_REPORTING.md).

No scenario generator was included in Phase 6. Core library functions remain
independently testable; the Phase 7 layer below adds controls without changing them.

## Phase 7 controlled-experiment layer

`generate_fixtures -> build_scenarios -> fit two clean pipelines once -> write
configs -> ten CLI subprocesses -> assess existing reports -> summary`.

- `experiments.py` defines laboratory fixtures, copy-on-mutation scenarios,
  predeclared thresholds, artifact/config/data fingerprints, and expected-result
  comparisons. It calls the existing baseline builders and ordinary Phase 6 CLI;
  it does not reimplement metrics, check rules or status aggregation.
- `scripts/run_experiments.py` is a thin argument/output wrapper. Experiment
  success means expected audit behavior, not that every audit returned zero.
- `tests/test_experiments.py` checks transformations, repeated complete suites,
  evidence identity, model reuse and failures of the experiment harness itself.
- All generated files live in a new ignored output directory by default.
  Original Telco data/model paths are never read or mutated by this layer.

The manifest records every intentional mutation. Report fingerprints must map
to the generated roles/files; a final hash check detects accidental validation
mutation. The runner distinguishes unexpected findings (1) from execution errors
(2), while expected deliberate audit failures contribute to runner success (0).
No new runtime dependency, CI, Docker, MLflow or deployment integration is added.

## Phase 8 verification layer

The CI workflow runs independent quality, test-matrix and package jobs. The
application dependency graph and validation calculations are unchanged. Coverage
instruments the application plus subprocesses, combines their measurements, and
gates both line and combined line/branch coverage at 90%.

`scripts/check_coverage.py` validates line coverage and module CLI evidence.
`scripts/check_distributions.py` reads archive member names and entry-point
metadata without extracting them, rejecting missing core files or included local
data/models. `scripts/ci_smoke.py` reuses Phase 7 fixture and evidence helpers while
invoking the installed console executable on expected audit exits 0, 1 and 2.
The package job creates a new environment, installs the wheel, and runs that
script from outside the checkout with an installed-package location guard.

Only development dependencies gain coverage.py and build. `MANIFEST.in` supplies
the sdist's public examples, test helpers and documentation. Explicit artifact
upload paths retain synthetic JSON and test/coverage reports for seven days.
Console discovery uses Python's configured scripts directory, supporting both
standalone Windows installations and virtual environments. The first hosted run
passed Linux/coverage/package jobs and exposed the now-locally-fixed Windows
discovery assumption; hosted verification of the fix remains pending.
Docker remains deferred. See
[Testing and CI](TESTING_CI.md) for reproduction and boundaries.
