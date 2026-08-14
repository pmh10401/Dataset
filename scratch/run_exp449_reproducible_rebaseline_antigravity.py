#!/usr/bin/env python3
"""Inactive outside-CWD entrypoint for the Exp449 read-only operations boundary."""
from __future__ import annotations

from pathlib import Path
import sys


if __name__ == "__main__":
    _repository_antigravity = Path(__file__).resolve().parents[1]
    if str(_repository_antigravity) not in sys.path:
        sys.path.insert(0, str(_repository_antigravity))
    from experiments.exp449_reproducible_rebaseline_antigravity.operations_antigravity import main_antigravity

    raise SystemExit(main_antigravity())
