# ModelGate

ModelGate is an educational, offline toolkit for validating the data,
predictions, and performance of scikit-learn binary-classification models
before a potential release.

The project is being built incrementally. The current Phase 0 foundation
contains packaging, configuration-loading and logging boundaries, tests, and
documentation scaffolding. It does **not** yet validate datasets or models.

ModelGate will produce evidence-based `PASS`, `WARNING`, and `FAIL` results.
It will not automatically approve, reject, retrain, or deploy a model, and its
checks will not guarantee the absence of leakage, bias, or production risk.

## Development setup

ModelGate requires Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

Run the Phase 0 checks:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
```

## Current scope

- Python package using a `src` layout
- Safe YAML document loading with a mapping root
- Reusable standard-library logging setup
- Smoke tests and Ruff configuration

Dataset acquisition, model training, validation checks, reporting, and a CLI
belong to later approved phases.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Decisions](docs/DECISIONS.md)
- [Build log](docs/BUILD_LOG.md)
- [Experiments](docs/EXPERIMENTS.md)
- [Interview guide](docs/INTERVIEW_GUIDE.md)

## Licence

ModelGate source code is available under the [MIT License](LICENSE). Dataset
licensing must be evaluated separately before any data is downloaded or
committed.
