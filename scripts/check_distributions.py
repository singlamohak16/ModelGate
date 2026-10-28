"""Inspect built archives without extracting them or executing their content."""

import argparse
import configparser
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "data",
    "models",
    "artifacts",
    "mlruns",
    "mlartifacts",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {".joblib", ".pkl", ".pyc", ".db", ".csv"}


def validate_names(names, *, wheel):
    normalized = []
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe archive path: {name}")
        if (
            FORBIDDEN_PARTS.intersection(path.parts)
            or path.suffix in FORBIDDEN_SUFFIXES
            or any(
                p == ".env" or p.startswith(".env.") or p.startswith("local-")
                for p in path.parts
            )
            or "reports/generated" in name
        ):
            raise ValueError(f"Generated/sensitive path in distribution: {name}")
        normalized.append(name)
    required = (
        "modelgate/__init__.py",
        "modelgate/__main__.py",
        "modelgate/cli.py",
        "modelgate/config.py",
        "modelgate/reporting.py",
        "modelgate/experiments.py",
    )
    for suffix in required:
        wanted = suffix if wheel else f"src/{suffix}"
        if not any(
            name == wanted or (not wheel and name.endswith("/" + wanted))
            for name in normalized
        ):
            raise ValueError(f"Missing package module: {wanted}")


def check_distributions(directory):
    directory = Path(directory)
    wheels, sources = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Require exactly one wheel and one source distribution.")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        validate_names(names, wheel=True)
        entries = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        parser = configparser.ConfigParser()
        if len(entries) == 1:
            parser.read_string(archive.read(entries[0]).decode())
        if (
            parser.get("console_scripts", "modelgate", fallback=None)
            != "modelgate.cli:main"
        ):
            raise ValueError("Wheel console entry point is missing or incorrect.")
    with tarfile.open(sources[0], "r:gz") as archive:
        validate_names(archive.getnames(), wheel=False)
    return wheels[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Verified distributions; wheel: {check_distributions(args.directory)}")
