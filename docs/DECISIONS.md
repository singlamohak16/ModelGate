# Decision Log

## D001 — Use a `src` package layout

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Place importable code under `src/modelgate`.
- **Reason:** This makes tests exercise the installed package and reduces the
  risk of imports succeeding only because the repository root is on Python's
  path.
- **Alternatives:** A flat `modelgate/` package is simpler initially but gives
  weaker protection against packaging mistakes.

## D002 — Use setuptools as the build backend

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Use setuptools with configuration in `pyproject.toml`.
- **Reason:** It is mature, conventional, and sufficient for this small Python
  package without adding a separate environment-management workflow.
- **Alternatives:** Hatchling offers a smaller modern build configuration;
  Poetry combines building and dependency management but adds concepts the
  project does not currently need.

## D003 — Use Pydantic and PyYAML for configuration

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Parse YAML with `yaml.safe_load` and use Pydantic to enforce a
  string-keyed mapping at the document root.
- **Reason:** Safe parsing plus typed validation provides a clear path to the
  complete configuration model planned for Phase 6.
- **Alternatives:** Standard dataclasses would avoid Pydantic but require custom
  validation and error formatting; JSON is stricter but does not meet the YAML
  requirement.

## D004 — Use standard-library logging

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Configure a named `modelgate` logger using Python's built-in
  logging package.
- **Reason:** It is adequate for an offline CLI and avoids an unnecessary
  runtime dependency.
- **Alternatives:** structlog and Loguru provide richer structured or ergonomic
  APIs but are not justified in the foundation.

## D005 — Recommend the MIT licence for source code

- **Date:** 2026-09-14
- **Status:** Accepted
- **Decision:** Licence ModelGate's original source under MIT.
- **Reason:** MIT is short, permissive, and common for portfolio tooling.
- **Limitation:** This decision does not grant rights to any dataset. Dataset
  provenance and licensing require separate review and approval in Phase 1.
