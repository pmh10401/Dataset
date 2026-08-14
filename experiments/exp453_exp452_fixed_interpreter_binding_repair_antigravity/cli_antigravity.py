"""Exact lazy command boundary for the Exp453 lifecycle."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
import json
import re


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")


def _reviewed_commit_antigravity(value: str) -> str:
    if _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("reviewed commit must be lowercase 40-hex")
    return value


def build_parser_antigravity() -> argparse.ArgumentParser:
    """Expose exactly readiness/evaluate with a commit and bare inspect."""
    parser = argparse.ArgumentParser(
        prog="run_exp453_exp452_fixed_interpreter_binding_repair_antigravity", allow_abbrev=False
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "evaluate"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--reviewed-commit", type=_reviewed_commit_antigravity, required=True)
    commands.add_parser("inspect", allow_abbrev=False)
    return parser


def _load_readiness_antigravity():
    from .readiness_antigravity import readiness_antigravity
    return readiness_antigravity


def _validate_reviewed_map_antigravity(reviewed_commit: str) -> None:
    """Require the clean reviewed map before evaluator import can occur."""
    from .readiness_antigravity import _validate_reviewed_map_antigravity as validate
    validate(reviewed_commit)


def _load_evaluate_antigravity():
    from .evaluation_antigravity import evaluate_antigravity
    return evaluate_antigravity


def _load_inspect_antigravity():
    from .replay_antigravity import inspect_antigravity
    return inspect_antigravity


def _json_safe_antigravity(value: object) -> object:
    if value is None or type(value) in (str, int, bool, float):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_safe_antigravity(item) for key, item in asdict(value).items()}
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_safe_antigravity(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_json_safe_antigravity(item) for item in value]
    raise ValueError("CLI result is not JSON-safe")


def _print_result_antigravity(result: object) -> None:
    print(json.dumps(_json_safe_antigravity(result), sort_keys=True, separators=(",", ":"), allow_nan=False))


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    """Dispatch the selected phase without importing any unrelated phase."""
    arguments = build_parser_antigravity().parse_args(None if argv is None else list(argv))
    if arguments.command == "readiness":
        _print_result_antigravity(_load_readiness_antigravity()(arguments.reviewed_commit))
        return 0
    if arguments.command == "evaluate":
        _validate_reviewed_map_antigravity(arguments.reviewed_commit)
        _print_result_antigravity(_load_evaluate_antigravity()(arguments.reviewed_commit))
        return 0
    if arguments.command == "inspect":
        _print_result_antigravity(_load_inspect_antigravity()())
        return 0
    raise ValueError("unknown closed Exp453 command")


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
