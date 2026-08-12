#!/usr/bin/env python3
"""Repository entrypoint for the public anomaly-baseline lifecycle."""

from __future__ import annotations

import os
from pathlib import Path
import sys


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.paper_public_baselines.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
