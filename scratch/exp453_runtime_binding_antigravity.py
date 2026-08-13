"""Stdlib-only exact executable/prefix binding used before any Exp453 import."""

from __future__ import annotations

import os
from pathlib import Path
import sys


FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
)
REEXEC_MARKER_ANTIGRAVITY = "EXP453_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"


def ensure_fixed_runtime_antigravity(launcher: Path, argv: tuple[str, ...]) -> None:
    """Re-exec once to the frozen interpreter when literal runtime strings differ."""
    if not isinstance(launcher, Path) or type(argv) is not tuple or any(type(item) is not str for item in argv):
        raise RuntimeError("Exp453 fixed runtime re-exec failed")
    if sys.executable == FIXED_EXECUTABLE_ANTIGRAVITY and sys.prefix == FIXED_PREFIX_ANTIGRAVITY:
        return
    if os.environ.get(REEXEC_MARKER_ANTIGRAVITY) == "1":
        raise RuntimeError("Exp453 fixed runtime re-exec failed")
    env = dict(os.environ)
    env[REEXEC_MARKER_ANTIGRAVITY] = "1"
    try:
        os.execve(
            FIXED_EXECUTABLE_ANTIGRAVITY,
            [FIXED_EXECUTABLE_ANTIGRAVITY, str(launcher), *argv],
            env,
        )
    except OSError as error:
        raise RuntimeError("Exp453 fixed runtime re-exec failed") from error
