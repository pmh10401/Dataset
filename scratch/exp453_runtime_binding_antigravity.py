"""Stdlib-only exact executable/prefix/version binding used before any Exp453 import."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import sys


FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
)
RUNTIME_VERSIONS_ANTIGRAVITY = ("3.14.6", "2.3.5", "1.4.0", "1.7.2")
REEXEC_MARKER_ANTIGRAVITY = "EXP453_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"


def _distribution_version_antigravity(name: str) -> str:
    return importlib.metadata.version(name)


def observe_exact_runtime_antigravity(*, distribution_version=None) -> tuple[str, str, str, str, str, str]:
    """Observe live executable, prefix, and four versions after path binding."""
    version = _distribution_version_antigravity if distribution_version is None else distribution_version
    if not callable(version):
        raise RuntimeError("Exp453 fixed runtime re-exec failed")
    try:
        numpy_version = version("numpy")
        aeon_version = version("aeon")
        sklearn_version = version("scikit-learn")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Exp453 fixed runtime re-exec failed") from error
    observed = (
        sys.executable,
        sys.prefix,
        platform.python_version(),
        numpy_version,
        aeon_version,
        sklearn_version,
    )
    if (
        observed[0] != FIXED_EXECUTABLE_ANTIGRAVITY
        or observed[1] != FIXED_PREFIX_ANTIGRAVITY
        or observed[2:] != RUNTIME_VERSIONS_ANTIGRAVITY
        or any(type(item) is not str or not item for item in observed)
    ):
        raise RuntimeError("Exp453 fixed runtime re-exec failed")
    return observed


def ensure_fixed_runtime_antigravity(launcher: Path, argv: tuple[str, ...]) -> None:
    """Re-exec once, then require the live exact runtime before any Exp453 import."""
    if not isinstance(launcher, Path) or type(argv) is not tuple or any(type(item) is not str for item in argv):
        raise RuntimeError("Exp453 fixed runtime re-exec failed")
    if sys.executable == FIXED_EXECUTABLE_ANTIGRAVITY and sys.prefix == FIXED_PREFIX_ANTIGRAVITY:
        observe_exact_runtime_antigravity()
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
