"""The fixed four-command Exp441 lifecycle interface."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from typing import Mapping

from .artifacts_antigravity import canonical_json_bytes_antigravity
from .readiness_antigravity import current_clean_head_antigravity


def build_parser_antigravity() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_exp441_hard_empty_single_shot_recovery_antigravity")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("readiness", "new", "inspect", "evaluate"):
        commands.add_parser(command)
    return parser


def _json_safe_antigravity(value: object) -> object:
    if is_dataclass(value):
        return _json_safe_antigravity(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe_antigravity(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_antigravity(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_antigravity(item) for item in value]
    return value


def canonical_cli_status_antigravity(command: str, result: object) -> str:
    """Return the sole stdout document emitted by a successful route."""
    return canonical_json_bytes_antigravity({"command": command, "result": _json_safe_antigravity(result)}).decode("utf-8")


def main_antigravity(argv: list[str] | None = None) -> int:
    command = build_parser_antigravity().parse_args(argv).command
    if command == "readiness":
        from .readiness_antigravity import run_readiness_antigravity
        result = run_readiness_antigravity(current_clean_head_antigravity())
    elif command == "new":
        from .readiness_antigravity import validate_readiness_for_run_antigravity
        binding = validate_readiness_for_run_antigravity(current_clean_head_antigravity())
        if current_clean_head_antigravity() != binding.reviewed_commit:
            raise ValueError("HEAD changed after readiness validation")
        from .runtime_antigravity import run_exp441_new_antigravity
        attempt = run_exp441_new_antigravity(reviewed_commit=binding.reviewed_commit)
        from .reporting_antigravity import publish_exp441_antigravity
        result = publish_exp441_antigravity(attempt)
    elif command == "inspect":
        from .reporting_antigravity import inspect_exp441_publication_antigravity
        result = inspect_exp441_publication_antigravity()
    else:
        from .evaluation_antigravity import evaluate_exp441_antigravity
        result = evaluate_exp441_antigravity()
    print(canonical_cli_status_antigravity(command, result), end="")
    return 0


__all__ = ["build_parser_antigravity", "main_antigravity", "canonical_cli_status_antigravity"]
