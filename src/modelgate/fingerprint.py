"""Byte-level fingerprints for datasets and local model artifacts."""

import hashlib
from pathlib import Path


def sha256_bytes(content: bytes) -> str:
    """Return a SHA-256 hex digest; changes in line endings also change this hash."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded chunks rather than loading the entire file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
