"""Outside-CWD launcher that binds the fixed interpreter before any Exp454 import."""

from __future__ import annotations

from pathlib import Path
import sys


_SCRATCH_ANTIGRAVITY = Path(__file__).parent
if str(_SCRATCH_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_SCRATCH_ANTIGRAVITY))

from exp454_runtime_binding_antigravity import ensure_fixed_runtime_antigravity

ensure_fixed_runtime_antigravity(Path(__file__), tuple(sys.argv[1:]))

import os


for _name_antigravity in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "AEON_NUM_THREADS",
    "MLX_NUM_THREADS",
):
    os.environ[_name_antigravity] = "1"

_REPOSITORY_ANTIGRAVITY = Path(__file__).parent.parent
if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))

from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.cli_antigravity import (
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
