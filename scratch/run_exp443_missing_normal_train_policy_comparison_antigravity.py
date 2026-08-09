#!/usr/bin/env python3
"""Run Exp443 from the fixed external environment and any current directory."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


OFFICIAL_PYTHON_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "exp443_minirocket_antigravity_py314/bin/python"
)
OFFICIAL_VERSION_ANTIGRAVITY = (3, 14, 6)


def _validate_interpreter_antigravity(*, executable: str | None = None) -> None:
    candidate = Path(sys.executable if executable is None else executable).absolute()
    expected = Path(OFFICIAL_PYTHON_ANTIGRAVITY).absolute()
    if candidate != expected:
        raise RuntimeError(
            f"Exp443 requires the official interpreter {expected}; got {candidate}"
        )
    if tuple(sys.version_info[:3]) != OFFICIAL_VERSION_ANTIGRAVITY:
        raise RuntimeError("Exp443 official interpreter version differs from Python 3.14.6")
    if Path(sys.prefix).absolute() != expected.parents[1]:
        raise RuntimeError("Exp443 official interpreter environment is not active")


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    _validate_interpreter_antigravity()
    repository = Path(__file__).resolve().parents[1]
    if not (repository / "experiments").is_dir():
        raise RuntimeError("Exp443 repository root is invalid")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from experiments.exp443_missing_normal_train_policy_comparison_antigravity.cli_antigravity import (
        main_antigravity as cli_main_antigravity,
    )
    return cli_main_antigravity(argv)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
