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
