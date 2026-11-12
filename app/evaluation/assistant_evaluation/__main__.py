"""python -m app.evaluation.assistant_evaluation <stage> (see the package docstring)."""

import sys

from app.evaluation.assistant_evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
