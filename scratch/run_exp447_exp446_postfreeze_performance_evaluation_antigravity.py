"""Outside-CWD launcher for the closed Exp447 evaluation CLI."""
from __future__ import annotations

import os
from pathlib import Path
import sys


for _name_antigravity in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name_antigravity] = "1"

_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))

from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.cli_antigravity import main_antigravity


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
