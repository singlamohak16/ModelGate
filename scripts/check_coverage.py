"""Require >=90% package line coverage and evidence from the module CLI child."""

import argparse
import json
from pathlib import Path


def check_coverage(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    totals = data["totals"]
    statements = totals["num_statements"]
    lines = 100 * totals["covered_lines"] / statements if statements else 0
    if lines < 90:
        raise ValueError(f"Package line coverage {lines:.2f}% is below 90%.")
    children = [
        v
        for k, v in data["files"].items()
        if k.replace("\\", "/").endswith("modelgate/__main__.py")
    ]
    if len(children) != 1 or children[0]["summary"]["missing_lines"]:
        raise ValueError("Module CLI subprocess coverage is missing or incomplete.")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(f"Verified line coverage: {check_coverage(args.report):.2f}%")
