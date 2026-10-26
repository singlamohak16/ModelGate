"""Support python -m modelgate with the same entry point as modelgate."""

from modelgate.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
