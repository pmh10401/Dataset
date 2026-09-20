#!/usr/bin/env python3
"""Outside-CWD entry point for the fixed Exp441 lifecycle."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(ROOT_ANTIGRAVITY))

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.cli_antigravity import main_antigravity


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
