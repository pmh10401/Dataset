"""Outside-CWD launcher that binds the frozen Exp465 interpreter."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types


_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-aeon140-py314/bin/python"
)
_FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-aeon140-py314"
)
_REEXEC_MARKER_ANTIGRAVITY = "EXP465_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"
_PRIVATE_CAPABILITY_ENV_ANTIGRAVITY = "EXP465_TASK5_PRIVATE_CAPABILITY_ANTIGRAVITY"
_PRIVATE_CAPABILITY_MODULE_ANTIGRAVITY = "_exp465_cli_private_test_antigravity"
_PRIVATE_CAPABILITY_VALUE_ANTIGRAVITY = "exp465_cli_private_test_antigravity_v1"
_PRIVATE_ROOTS_ANTIGRAVITY = {
    "run1": Path("/private/tmp/exp465_task5_private_run1_antigravity"),
    "run2": Path("/private/tmp/exp465_task5_private_run2_antigravity"),
}


def _ensure_fixed_runtime_antigravity() -> None:
    if (
        sys.executable == _FIXED_EXECUTABLE_ANTIGRAVITY
        and sys.prefix == _FIXED_PREFIX_ANTIGRAVITY
    ):
        return
    if os.environ.get(_REEXEC_MARKER_ANTIGRAVITY) == "1":
        raise RuntimeError("Exp465 fixed runtime re-exec failed")
    environment = dict(os.environ)
    environment[_REEXEC_MARKER_ANTIGRAVITY] = "1"
    os.execve(
        _FIXED_EXECUTABLE_ANTIGRAVITY,
        [_FIXED_EXECUTABLE_ANTIGRAVITY, str(Path(__file__)), *sys.argv[1:]],
        environment,
    )


_ensure_fixed_runtime_antigravity()


def _install_private_capability_antigravity() -> None:
    """Install one preregistered private root without widening CLI arguments."""
    selector = os.environ.get(_PRIVATE_CAPABILITY_ENV_ANTIGRAVITY)
    if selector is None:
        return
    root = _PRIVATE_ROOTS_ANTIGRAVITY.get(selector)
    if root is None:
        raise RuntimeError("private capability is not preregistered")
    private = types.ModuleType(_PRIVATE_CAPABILITY_MODULE_ANTIGRAVITY)
    private.CAPABILITY_ANTIGRAVITY = _PRIVATE_CAPABILITY_VALUE_ANTIGRAVITY
    private.root_antigravity = root
    sys.modules[_PRIVATE_CAPABILITY_MODULE_ANTIGRAVITY] = private


_install_private_capability_antigravity()

_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))

from experiments.exp465_ecg200_c8_external_validation_antigravity.cli_antigravity import (
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
