#!/usr/bin/env python3
"""Outside-CWD-safe entrypoint for the frozen Exp439 CLI."""

from pathlib import Path
import sys


REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ANTIGRAVITY))

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.cli_antigravity import (
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
