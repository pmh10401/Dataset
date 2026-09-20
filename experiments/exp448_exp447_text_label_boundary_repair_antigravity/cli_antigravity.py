"""Closed, lazy command boundary for the Exp448 TEXT-label repair."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import re


_REVIEWED_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")


def _reviewed_commit_antigravity(value: str) -> str:
    if _REVIEWED_COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("reviewed commit must be a lowercase 40-hex SHA")
    return value


def build_parser_antigravity() -> argparse.ArgumentParser:
    """Build the deliberately small command surface without runtime imports."""
    parser = argparse.ArgumentParser(
        prog="run_exp448_exp447_text_label_boundary_repair_antigravity",
        description="Exp448 controlled TEXT-label evaluation boundary.",
        allow_abbrev=False,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "evaluate"):
        command = subcommands.add_parser(name, allow_abbrev=False)
        command.add_argument("--reviewed-commit", required=True, type=_reviewed_commit_antigravity)
    subcommands.add_parser("inspect", allow_abbrev=False)
    return parser


def _load_readiness_antigravity():
    from .readiness_antigravity import readiness_antigravity

    return readiness_antigravity


def _load_clean_review_antigravity():
    from .readiness_antigravity import _require_clean_reviewed_head_antigravity

    return _require_clean_reviewed_head_antigravity


def _load_evaluate_antigravity():
    from .evaluation_antigravity import evaluate_antigravity

    return evaluate_antigravity


def _load_inspect_antigravity():
    """Load only the pure replay primitive; it cannot reach TEST labels."""
    from .replay_antigravity import inspect_replay_with_dependencies_antigravity

    return inspect_replay_with_dependencies_antigravity


def _inspect_dependencies_antigravity():
    """Build the identity-only adapter after selecting source-free replay."""
    from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
    from .publication_identity_antigravity import capture_exp446_publication_antigravity

    class _InspectDependenciesAntigravity:
        root = Path(FROZEN_CONFIG_ANTIGRAVITY.run_root)

        @staticmethod
        def capture_after():
            return capture_exp446_publication_antigravity()

    return _InspectDependenciesAntigravity()


def _json_safe_antigravity(value: object) -> object:
    """Recursively serialize public result values without private authorities."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("CLI result contains a non-finite float")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_safe_antigravity(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("CLI result mapping keys must be strings")
        return {key: _json_safe_antigravity(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_antigravity(item) for item in value]
    raise ValueError("CLI result contains an unsupported value")


def _print_result_antigravity(result: object) -> None:
    print(json.dumps(_json_safe_antigravity(result), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    """Dispatch one fixed command while retaining the pre-label import gate."""
    arguments = build_parser_antigravity().parse_args(None if argv is None else list(argv))
    if arguments.command == "readiness":
        _print_result_antigravity(_load_readiness_antigravity()(arguments.reviewed_commit))
        return 0
    if arguments.command == "evaluate":
        # This check happens before importing evaluation, which is the only CLI
        # path permitted to reach the TEST-label-capable evaluator.
        _load_clean_review_antigravity()(arguments.reviewed_commit)
        _print_result_antigravity(_load_evaluate_antigravity()(arguments.reviewed_commit))
        return 0
    if arguments.command == "inspect":
        _print_result_antigravity(_load_inspect_antigravity()(_inspect_dependencies_antigravity()))
        return 0
    raise ValueError("unknown closed Exp448 command")


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
