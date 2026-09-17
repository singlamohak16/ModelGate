"""Write a Phase 2 library demonstration, not the future validation CLI report."""

import argparse
import json
from collections import Counter
from pathlib import Path

from modelgate.checks import DataCheckPolicy, run_data_checks
from modelgate.dataset import read_csv_strings
from modelgate.fingerprint import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/generated/phase2_data_checks.json")
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(2, "Demo output already exists; choose a new --output path.\n")
    try:
        paths = {
            name: args.prepared_dir / f"{name}.csv" for name in ("train", "reference")
        }
        frames = {name: read_csv_strings(path) for name, path in paths.items()}
        results = run_data_checks(
            frames["train"], frames["reference"], DataCheckPolicy()
        )
        document = {
            "report_kind": "phase2_data_checks_demo",
            "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
            "results": [result.model_dump(mode="json") for result in results],
        }
        content = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Data-check demo error: {error}\n")
    counts = dict(Counter(result.status.value for result in results))
    print(f"Individual check counts: {counts}")
    print(f"Demo written to {args.output}; run-level status is not implemented yet.")


if __name__ == "__main__":
    main()
