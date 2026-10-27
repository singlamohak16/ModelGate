"""Run Phase 7 controls; expected audit failures do not fail the experiment suite."""

import argparse
import sys
from pathlib import Path

from modelgate.experiments import run_experiments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase7"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        summary = run_experiments(args.output_dir, seed=args.seed)
    except Exception as error:
        print(f"Experiment execution error: {error}", file=sys.stderr)
        return 2
    for row in summary["rows"]:
        outcome = "MATCH" if row["matched"] else "MISMATCH/ERROR"
        print(
            f"{row['model']} / {row['scenario']}: {outcome} "
            f"(audit {row['audit_status']}, exit {row['process_exit']})"
        )
    print(f"Summary: {args.output_dir.resolve() / 'summary.json'}")
    return summary["runner_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
