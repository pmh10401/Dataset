"""Outside-CWD launcher that binds the fixed interpreter before any Exp459 import."""

from __future__ import annotations

from pathlib import Path
import os
import sys


_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
_FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
)
_REEXEC_MARKER_ANTIGRAVITY = "EXP459_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"


def _ensure_fixed_runtime_antigravity() -> None:
    if (
        sys.executable == _FIXED_EXECUTABLE_ANTIGRAVITY
        and sys.prefix == _FIXED_PREFIX_ANTIGRAVITY
    ):
        return
    if os.environ.get(_REEXEC_MARKER_ANTIGRAVITY) == "1":
        raise RuntimeError("Exp459 fixed runtime re-exec failed")
    env = dict(os.environ)
    env[_REEXEC_MARKER_ANTIGRAVITY] = "1"
    os.execve(
        _FIXED_EXECUTABLE_ANTIGRAVITY,
        [_FIXED_EXECUTABLE_ANTIGRAVITY, str(Path(__file__)), *sys.argv[1:]],
        env,
    )


_ensure_fixed_runtime_antigravity()

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

_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.cli_antigravity import (
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
