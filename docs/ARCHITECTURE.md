# Architecture

## Phase 0 boundaries

ModelGate uses a `src` package layout so tests and users import the installed
package rather than accidentally importing code from the repository root.

The current package has two small boundaries:

1. `modelgate.config` reads YAML safely and validates that its root is a
   string-keyed mapping.
2. `modelgate.logging` configures one reusable package logger without adding a
   duplicate handler when called repeatedly.

Future validation logic will be added in later phases. Data loading, model
training, checks, report models, and CLI orchestration are intentionally absent
from this foundation.
