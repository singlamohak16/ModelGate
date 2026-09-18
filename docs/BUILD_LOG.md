# Build Log

This log records work when it actually occurs. It does not reconstruct or
backdate project activity.

## 2026-09-14 — Phase 0 foundation

- Verified that the workspace was empty and contained no `AGENTS.md`.
- Confirmed Python 3.12 was available and pytest, Ruff, and GitHub CLI were not.
- Initialized an empty local Git repository on `main`.
- Added the initial package, configuration, logging, documentation, and test
  skeletons.
- Created an ignored project-local `.venv` and installed the package in editable
  mode with its development dependencies.
- Ran 5 tests successfully with Python 3.12.14 and pytest 9.1.1.
- Passed `ruff check .` and `ruff format --check .` with Ruff 0.16.7.
- Confirmed `pip check` reports no broken requirements.
- Confirmed the installed `modelgate` package imports at version 0.1.0.
- Verified that virtual environments, datasets, model artifacts, generated
  reports, MLflow state, and `.env` files are ignored.

## 2026-09-15 — Phase 1 data and baselines

- Received explicit Phase 1 and IBM source approval, with data excluded from Git.
- Verified remote main matches local Phase 0 commit `e71b48b` and created the
  local `phase/1-data-baselines` branch.
- Pinned IBM revision `d5371f5d83a446ad5673cbcca3b814b926491f8a` and verified the
  970,457-byte source's SHA-256 before preparing it.
- Added acquisition, audit, schema, parsing, fingerprint, split, preprocessing,
  training, and two small argparse development entry points.
- Installed pandas, NumPy, scikit-learn, and joblib; captured tested package
  versions in `constraints-python312.txt`.
- Audited 7,043 rows / 21 columns, with 11 blank TotalCharges values and zero
  exact duplicate rows or duplicate IDs.
- Created a seed-42 stratified split: 5,634 training and 1,409 reference rows.
- Trained Logistic Regression and Random Forest without training warnings.
  Reference F1 values were 0.57268722 and 0.56097561 at threshold 0.50.
- Ran training twice; probability arrays agreed within `rtol=0, atol=1e-12` and
  metric dictionaries were identical. Verified save/load equivalence and repeated
  split membership. Models and data were saved only under ignored directories.
- Passed 34 tests using synthetic fixtures, including the five Phase 0 tests.
- Passed Ruff linting and formatting on the implementation.
- Updated dataset, architecture, decision, experiment, setup, and interview docs
  with actual results and limitations. No Phase 1 commit or GitHub write yet.

## 2026-09-17 — Phase 2 data validation

- Received explicit Phase 2 implementation approval and verified that local and
  remote main matched the merged Phase 1 commit `1582c28` with a clean worktree.
- Created local branch `phase/2-data-validation`.
- Added validated data-check settings, a shared CheckResult contract, and checks
  for schema, target format, identifiers, segment presence, missingness, exact
  duplicates, composite IDs, overlap, and limited target-like/prohibited features.
- Defined strict maximum boundaries, full-row denominators, bounded evidence,
  explicit exclusions, and unevaluated WARNING results with prerequisite reasons.
- Added a small local JSON demonstration without implementing the Phase 6
  validation CLI, complete YAML schema, overall status, or validation exit codes.
- Passed 100 tests: all 34 existing tests plus 66 Phase 2 tests. Coverage includes
  measurements, boundary equality, missing IDs, large numeric keys, recodings,
  invalid policies, JSON output, and demonstration overwrite protection.
- Passed Ruff linting, Ruff formatting, and dependency integrity checks. No new
  dependencies were added.
- Ran the checks on the original prepared split: 135 PASS, 1 WARNING, 0 FAIL;
  all 136 checks were evaluated. The warning identified 10 reference predictor
  matches (0.709723%) and ID overlap remained zero.
- Independently confirmed those 10 matches with a pandas merge. Rechecked the
  source partition hashes against Phase 1; both datasets remained unchanged.
- Generated final local evidence at `reports/generated/phase2_verified.json`.
  Verified that data, models, and generated reports remain ignored by Git.
- Updated architecture, decisions, measured evidence, setup, and interview notes.
  No Phase 2 commit or GitHub write has been made at this point.

## 2026-09-17 — Phase 3 model validation

- Received explicit Phase 3 implementation approval after presenting its proposal.
  Verified clean main against fetched origin/main at merged Phase 2 commit
  `2189ba9` and created local branch `phase/3-model-validation`.
- Added validated binary probability extraction using fitted class ordering,
  policy-independent classification/Brier/calibration/threshold helpers, and
  six evidence-based checks per model with optional quality limits.
- Kept undefined metrics as null plus explicit reasons and WARNING status.
  Documented one-class behavior, array alignment, strict threshold comparisons,
  descriptive PASS semantics and Brier's limitations as a calibration measure.
- Added a trusted-local-model demonstration that checks dataset/training
  manifests and both artifact hashes, does not fit anything, records provenance
  and protects existing reports. No new dependencies or training-code changes.
- Evaluated both existing pipelines on the unchanged 1,409-row reference split.
  Classification measurements matched Phase 1. Brier was 0.13706126432822094
  for Logistic Regression and 0.13930202436298936 for Random Forest.
- Generated `reports/generated/phase3_model_checks.json` and
  `reports/generated/phase3_repeat.json`; the two report contents were identical.
  All 12 checks were evaluated PASS with no configured model-quality limits
  (ten descriptive metric results and two class-support checks).
- Independently checked real-data Brier and nonempty calibration bins against
  Scikit-learn. Recorded the full threshold grid in the experiments document.
- Passed 171 tests (100 existing plus 71 new), Ruff linting and formatting, and
  dependency integrity checks. Tests cover hand counts, Scikit-learn agreement,
  positive-label direction, invalid inputs, policy boundaries, JSON, trusted
  artifact integration, repeatability and overwrite protection.
- Corrected an initial boundary fixture to avoid using a rounded decimal Brier
  limit as an exact floating-point equality; production comparisons remain
  strict and unrounded. No tolerance was added to make a rule pass.
- Updated README, architecture, decisions, measured experiments, model-validation
  guide and interview notes. No Phase 4 work or Phase 3 commit/push performed.
