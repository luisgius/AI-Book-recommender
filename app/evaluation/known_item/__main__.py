"""python -m app.evaluation.known_item <stage> (see the package docstring)."""

import sys

from app.evaluation.known_item.cli import main

if __name__ == "__main__":
    sys.exit(main())
