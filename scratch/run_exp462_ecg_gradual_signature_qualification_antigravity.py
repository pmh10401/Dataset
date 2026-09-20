"""Outside-CWD launcher that binds the fixed interpreter before Exp462 imports."""

from __future__ import annotations

import os
from pathlib import Path
import sys


_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
_FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
)
_REEXEC_MARKER_ANTIGRAVITY = "EXP462_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"


def _ensure_fixed_runtime_antigravity() -> None:
    if (
        sys.executable == _FIXED_EXECUTABLE_ANTIGRAVITY
        and sys.prefix == _FIXED_PREFIX_ANTIGRAVITY
    ):
        return
    if os.environ.get(_REEXEC_MARKER_ANTIGRAVITY) == "1":
        raise RuntimeError("Exp462 fixed runtime re-exec failed")
    environment = dict(os.environ)
    environment[_REEXEC_MARKER_ANTIGRAVITY] = "1"
    os.execve(
        _FIXED_EXECUTABLE_ANTIGRAVITY,
        [
            _FIXED_EXECUTABLE_ANTIGRAVITY,
            str(Path(__file__)),
            *sys.argv[1:],
        ],
        environment,
    )


_ensure_fixed_runtime_antigravity()

_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))

from experiments.exp462_ecg_gradual_signature_qualification_antigravity.cli_antigravity import (
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
