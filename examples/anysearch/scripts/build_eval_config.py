#!/usr/bin/env python3
"""Command-line entry point for ``anysearch build-eval-config``."""

import sys
from pathlib import Path

FORK_ROOT = Path(__file__).resolve().parents[3]
if str(FORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FORK_ROOT))

if __name__ == "__main__":
    from examples.anysearch.cli import main

    main(["build-eval-config", *sys.argv[1:]])
