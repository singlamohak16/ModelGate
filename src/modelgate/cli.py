"""Command-line entry point; code 0 includes completed WARNING runs."""

import argparse
import sys

from modelgate.config import load_run_config
from modelgate.reporting import write_report
from modelgate.validation import run_validation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelgate",
        description="Offline model validation evidence, not production approval.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser(
        "validate", help="Validate one trusted fitted model."
    )
    validate.add_argument("--config", required=True, help="YAML run configuration.")
    args = parser.parse_args(argv)
    try:
        config = load_run_config(args.config)
        if config.output.exists():
            raise FileExistsError(
                f"Report already exists; choose a new output: {config.output}"
            )
        report = run_validation(config, args.config)
        write_report(report, config.output)
    except Exception as error:
        print(
            f"ModelGate execution error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    if report.errors:
        print("ModelGate execution error: " + "; ".join(report.errors), file=sys.stderr)
    else:
        counts = {
            status: sum(c.status == status for c in report.checks)
            for status in ("PASS", "WARNING", "FAIL")
        }
        print(
            f"ModelGate: {report.overall_status} | {counts} | exit {report.exit_code}"
        )
    print(f"Report: {config.output}")
    return report.exit_code
