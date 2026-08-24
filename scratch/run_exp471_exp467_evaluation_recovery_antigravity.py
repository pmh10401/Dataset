"""Closed launcher for the verified Exp471 recovery pipeline."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT_ANTIGRAVITY))

from experiments.exp471_exp467_evaluation_recovery_antigravity.cli_antigravity import (  # noqa: E402
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
