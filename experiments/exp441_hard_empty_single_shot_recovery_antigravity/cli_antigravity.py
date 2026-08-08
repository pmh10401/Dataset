"""The fixed four-command Exp441 lifecycle interface."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import math
from typing import Mapping

from .artifacts_antigravity import canonical_json_bytes_antigravity
from .readiness_antigravity import current_clean_head_antigravity
from .lifecycle_profile_antigravity import (
    EXP441_PROFILE_ANTIGRAVITY,
    RecoveryLifecycleProfileAntigravity,
)


def _build_parser_for_profile_antigravity(
    profile: RecoveryLifecycleProfileAntigravity,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=profile.cli_program)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("readiness", "new", "inspect", "evaluate"):
        commands.add_parser(command)
    return parser


def build_parser_antigravity() -> argparse.ArgumentParser:
    return _build_parser_for_profile_antigravity(EXP441_PROFILE_ANTIGRAVITY)


def _json_safe_antigravity(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("nonfinite CLI JSON float")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe_antigravity(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("CLI JSON mapping keys must be strings")
        return {key: _json_safe_antigravity(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_json_safe_antigravity(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_safe_antigravity(item) for item in value]
        return sorted(converted, key=canonical_json_bytes_antigravity)
    raise TypeError(f"unsupported CLI JSON value: {type(value).__name__}")


def canonical_cli_status_antigravity(command: str, result: object) -> str:
    """Return the sole stdout document emitted by a successful route."""
    return canonical_json_bytes_antigravity({"command": command, "result": _json_safe_antigravity(result)}).decode("utf-8")


def main_for_profile_antigravity(
    profile: RecoveryLifecycleProfileAntigravity,
    argv: list[str] | None = None,
) -> int:
    command = _build_parser_for_profile_antigravity(profile).parse_args(argv).command
    if command == "readiness":
        if profile is EXP441_PROFILE_ANTIGRAVITY:
            from .readiness_antigravity import run_readiness_antigravity
            result = run_readiness_antigravity(current_clean_head_antigravity())
        else:
            from .readiness_antigravity import _run_readiness_for_profile_antigravity
            result = _run_readiness_for_profile_antigravity(profile, current_clean_head_antigravity())
    elif command == "new":
        if profile is EXP441_PROFILE_ANTIGRAVITY:
            from .readiness_antigravity import validate_readiness_for_run_antigravity
            binding = validate_readiness_for_run_antigravity(current_clean_head_antigravity())
        else:
            from .readiness_antigravity import _validate_readiness_for_profile_antigravity
            binding = _validate_readiness_for_profile_antigravity(profile, current_clean_head_antigravity())
        if current_clean_head_antigravity() != binding.reviewed_commit:
            raise ValueError("HEAD changed after readiness validation")
        if profile is EXP441_PROFILE_ANTIGRAVITY:
            from .runtime_antigravity import run_exp441_new_antigravity
            attempt = run_exp441_new_antigravity(reviewed_commit=binding.reviewed_commit)
            from .reporting_antigravity import publish_exp441_antigravity
            result = publish_exp441_antigravity(attempt)
        else:
            from .runtime_antigravity import _run_new_for_profile_antigravity
            attempt = _run_new_for_profile_antigravity(profile, reviewed_commit=binding.reviewed_commit)
            from .reporting_antigravity import _publish_for_profile_antigravity
            result = _publish_for_profile_antigravity(profile, attempt)
    elif command == "inspect":
        if profile is EXP441_PROFILE_ANTIGRAVITY:
            from .reporting_antigravity import inspect_exp441_publication_antigravity
            result = inspect_exp441_publication_antigravity()
        else:
            from .reporting_antigravity import _inspect_for_profile_antigravity
            result = _inspect_for_profile_antigravity(profile)
    else:
        if profile is EXP441_PROFILE_ANTIGRAVITY:
            from .evaluation_antigravity import evaluate_exp441_antigravity
            result = evaluate_exp441_antigravity()
        else:
            from .evaluation_antigravity import _evaluate_for_profile_antigravity
            result = _evaluate_for_profile_antigravity(profile)
    print(canonical_cli_status_antigravity(command, result), end="")
    return 0


def main_antigravity(argv: list[str] | None = None) -> int:
    return main_for_profile_antigravity(EXP441_PROFILE_ANTIGRAVITY, argv)


__all__ = [
    "build_parser_antigravity", "main_antigravity", "main_for_profile_antigravity",
    "canonical_cli_status_antigravity",
]
