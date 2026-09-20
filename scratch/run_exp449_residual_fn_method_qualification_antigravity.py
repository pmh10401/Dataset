"""Outside-CWD launcher for the exact Exp449 command surface."""
from __future__ import annotations

import os


for _name_antigravity in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "AEON_NUM_THREADS",
):
    os.environ[_name_antigravity] = "1"

from pathlib import Path
import sys


_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))

from experiments.exp449_residual_fn_method_qualification_antigravity.cli_antigravity import main_antigravity


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
