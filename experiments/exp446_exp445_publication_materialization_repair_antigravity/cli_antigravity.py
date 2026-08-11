"""Closed three-command, source-free Exp446 operator interface."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Mapping


_REVIEWED_COMMIT_ANTIGRAVITY = re.compile(r"[0-9a-f]{40}$")


def build_parser_antigravity() -> argparse.ArgumentParser:
    """Expose exactly readiness, new, and frozen-leaf inspect."""
    parser = argparse.ArgumentParser(prog="exp446-antigravity")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "new"):
        command = commands.add_parser(name)
        command.add_argument("--reviewed-commit", required=True)
    commands.add_parser("inspect")
    return parser


def _validate_reviewed_commit_antigravity(reviewed_commit: str) -> str:
    from experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity import current_clean_head_antigravity

    if type(reviewed_commit) is not str or _REVIEWED_COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("Exp446 reviewed commit is invalid")
    if current_clean_head_antigravity() != reviewed_commit:
        raise ValueError("Exp446 reviewed commit is not the current clean HEAD")
    return reviewed_commit


def inspect_prediction_antigravity():
    """Delay lifecycle/reporting imports until frozen inspection is explicitly dispatched."""
    from .lifecycle_antigravity import inspect_prediction_antigravity as inspect

    return inspect()


def _dispatch_antigravity(arguments: argparse.Namespace) -> object:
    if arguments.command == "readiness":
        from .lifecycle_antigravity import run_readiness_antigravity

        return run_readiness_antigravity(_validate_reviewed_commit_antigravity(arguments.reviewed_commit))
    if arguments.command == "new":
        from .lifecycle_antigravity import publish_prediction_antigravity, run_new_antigravity

        outcome = run_new_antigravity(_validate_reviewed_commit_antigravity(arguments.reviewed_commit))
        return {"runtime": outcome, "publication": publish_prediction_antigravity()}
    if arguments.command == "inspect":
        return inspect_prediction_antigravity()
    raise ValueError("Exp446 CLI command is invalid")


def _json_safe_antigravity(value: object) -> object:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("CLI mapping key is invalid")
        return {key: _json_safe_antigravity(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_safe_antigravity(item) for item in value]
    payload = getattr(value, "to_payload_antigravity", None)
    if callable(payload):
        return _json_safe_antigravity(payload())
    if is_dataclass(value):
        return _json_safe_antigravity(asdict(value))
    raise TypeError(f"CLI output is unsupported: {type(value).__name__}")


def main_antigravity(argv: tuple[str, ...] | None = None) -> int:
    try:
        arguments = build_parser_antigravity().parse_args(argv)
        result = _dispatch_antigravity(arguments)
    except SystemExit:
        raise
    except BaseException as error:
        print(json.dumps({"error": str(error) or type(error).__name__}, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps({"command": arguments.command, "result": _json_safe_antigravity(result)}, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = ["build_parser_antigravity", "main_antigravity"]
