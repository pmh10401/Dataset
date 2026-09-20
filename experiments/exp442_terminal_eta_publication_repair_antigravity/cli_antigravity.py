"""The exact four-command Exp442 CLI."""
from __future__ import annotations

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.cli_antigravity import (
    main_for_profile_antigravity,
)

from .profile_antigravity import PROFILE_ANTIGRAVITY


def main_antigravity(argv: list[str] | None = None) -> int:
    return main_for_profile_antigravity(PROFILE_ANTIGRAVITY, argv)


__all__ = ["main_antigravity"]
