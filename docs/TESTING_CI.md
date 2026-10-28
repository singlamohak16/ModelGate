# Testing, coverage and CI

Phase 8 adds automated verification without changing model or audit calculations.
Local Windows/Python 3.12 verification is recorded in [EXPERIMENTS.md](EXPERIMENTS.md).
Hosted Linux/Python 3.11 and the GitHub runner matrix remain unverified until an
approved push runs `.github/workflows/ci.yml`. Docker is explicitly deferred.

## Workflow

| Job | Environment | Required result |
| --- | --- | --- |
| Quality | Ubuntu 24.04, Python 3.12 | Ruff lint and formatting pass |
| Tests | Ubuntu 24.04, Python 3.11 | Full pytest suite passes |
| Tests with coverage | Ubuntu 24.04, Python 3.12 | Full suite; line and combined coverage each >=90% |
| Tests | Windows 2025, Python 3.12 | Full pytest suite passes |
| Package | Ubuntu 24.04, Python 3.12 | Build sdist/wheel; inspect contents; fresh wheel install; console smoke passes |

Pushes to `main` and `phase/**`, and pull requests targeting `main`, trigger CI.
An updated revision cancels older runs for the same branch or pull request.
Jobs have 10–20 minute timeouts and matrix failures do not cancel sibling jobs.
The suite uses invented fixtures; no IBM CSV, stored model, network service,
credential or dataset download is required during testing. Dependency installation
and isolated build setup require package-index access.

Dependencies resolve within the supported ranges. `pip freeze` records the actual
environment in logs; `pip check` rejects inconsistent dependencies. The historical
`constraints-python312.txt` is a Windows baseline snapshot, not a universal lock
for all Python versions and operating systems. This matrix samples compatibility;
it does not establish support for every version allowed by the metadata.

## Local test and coverage commands

From the repository root, install the development dependencies and run:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m pip check
$env:COVERAGE_FILE = Join-Path (Get-Location) 'artifacts/ci/.coverage'
.venv\Scripts\python -m coverage erase
.venv\Scripts\python -m coverage run --rcfile=pyproject.toml -m pytest -q -p no:cacheprovider --junitxml=artifacts/ci/junit.xml
.venv\Scripts\python -m coverage combine
.venv\Scripts\python -m coverage report
.venv\Scripts\python -m coverage json
.venv\Scripts\python -m coverage xml
.venv\Scripts\python scripts/check_coverage.py artifacts/ci/coverage.json
```

Stop if any command fails. Erase clears only prior coverage measurements at the
selected location; retain evidence elsewhere before repeating if needed.
Use a writable temporary directory when Windows permissions prevent pytest from
creating fixtures. Disabling its optional cache does not disable any tests.

Coverage measures `src/modelgate`, including CLI child processes through
coverage.py's `subprocess` patch. Parallel data files are combined before reporting.
`coverage report` gates the combined line/branch percentage at 90%; a separate
checker requires at least 90% line coverage and complete statement coverage of
`modelgate/__main__.py`, providing evidence that module CLI children were measured.
Branch coverage is reported explicitly, with no separate branch-only threshold.
No application lines are excluded to meet the target. Development helper scripts
are tested but are outside the application coverage denominator.

Line coverage is executed statements / executable statements. Branch coverage
measures traversed branch destinations; combined coverage includes both in its
denominator. None measures assertion quality, realistic model quality, fairness,
or a probability that the software is correct.

## Build and exercise an installed wheel

Use fresh output paths; the smoke runner refuses an existing destination.

```powershell
.venv\Scripts\python -m build --outdir artifacts/package-check
.venv\Scripts\python scripts/check_distributions.py artifacts/package-check
.venv\Scripts\python -m venv artifacts/package-env
artifacts\package-env\Scripts\python -m pip install artifacts/package-check/modelgate-0.1.0-py3-none-any.whl
artifacts\package-env\Scripts\python -m pip check
$wheelPython = (Resolve-Path artifacts/package-env/Scripts/python.exe).Path
$smokeOutput = Join-Path (Get-Location) 'artifacts/package-smoke'
$smokeTemp = Join-Path $env:TEMP ('ModelGate-smoke-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $smokeTemp | Out-Null
Copy-Item scripts/ci_smoke.py (Join-Path $smokeTemp 'ci_smoke.py')
Push-Location $smokeTemp
try { & $wheelPython -I ci_smoke.py --require-wheel --output-dir $smokeOutput }
finally { Pop-Location }
```

Again, stop on a nonzero command result. Building uses an isolated build backend;
the default build creates the wheel from the generated source distribution.
`MANIFEST.in` includes source-test support, public example configs and documentation
in the sdist. The wheel contains the installed application and package metadata.
The archive checker refuses unsafe paths and known data/model/generated-output
locations or extensions, checks core modules, and requires the exact console entry
point. It is a packaging regression guard, not a general secret scanner.

The smoke script is copied outside the checkout and runs with Python isolated
mode. Its guard verifies imports come from installed package metadata rather than
an editable checkout. It trains one existing Logistic Regression pipeline on
invented clean data, then invokes the installed `modelgate` console command:

| Case | Required audit result | What is verified |
| --- | --- | --- |
| Clean | PASS, exit 0 | Exact clean-case evidence from Phase 7 |
| Contamination | FAIL, exit 1 | Exactly 20 overlapping reference IDs and expected rules |
| Missing input | No verdict, exit 2 | Error execution state, populated errors, valid report schema |

The smoke runner itself returns 0 only when all three expectations match; every
unexpected exit, missing/invalid report or mismatched finding fails the runner.
Full pytest retains both-model and all-five-scenario controls from Phase 7.

## Permissions and retained evidence

Actions use immutable commit SHAs, read-only repository contents, and checkout
does not persist credentials. There is no `pull_request_target`, secret use,
publishing, deployment or repository-write step. SHA pins were verified against
official release tags: checkout v7.0.1, setup-python v7.1.0 and upload-artifact
v7.0.2. Updates need deliberate review.

Artifacts expire after seven days. Upload paths select JUnit XML, coverage JSON/XML
and synthetic smoke summary/report JSON only. Generated CSVs, joblib models,
virtual environments and real Telco artifacts are not uploaded. The upload steps
run after failure, retaining whichever evidence was actually produced; an early
failure can leave no report. Reports include runner paths and synthetic metadata.
Generated local evidence remains ignored by Git.

Docker was deferred because a working local engine was unavailable, and it is
unnecessary for these package and CLI checks. No image is built or published.
Hosted execution, branch protection and required-check settings are separate
from defining a workflow; local success does not establish any of them.
