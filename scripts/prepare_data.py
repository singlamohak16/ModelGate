"""Prepare the approved IBM dataset: run from the repository after editable install."""

import argparse
from pathlib import Path
from urllib.error import URLError

from modelgate.dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        manifest = prepare_dataset(args.output_dir, seed=args.seed)
    except (ValueError, OSError, URLError) as error:
        parser.exit(2, f"Preparation error: {error}\n")
    counts = {name: part["rows"] for name, part in manifest["partitions"].items()}
    print(
        f"Prepared {counts}; metadata: {args.output_dir / 'prepared' / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
