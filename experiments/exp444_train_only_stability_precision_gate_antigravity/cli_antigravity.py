"""Fixed five-command Exp444 operator interface."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path
import re
from typing import Mapping

from .config_antigravity import FROZEN_CONFIG
from .readiness_antigravity import current_clean_head_antigravity, run_readiness_antigravity, validate_readiness_for_run_antigravity
from .reporting_antigravity import inspect_publication_antigravity, publish_prediction_antigravity
from .runtime_antigravity import run_new_antigravity, run_resume_antigravity


_ATTEMPT_ANTIGRAVITY = re.compile(r"attempt_[0-9]{4}_antigravity$")


def build_parser_antigravity() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exp444-antigravity")
    commands = parser.add_subparsers(dest="command", required=True)
    readiness = commands.add_parser("readiness")
    readiness.add_argument("--reviewed-commit", required=True)
    new = commands.add_parser("new")
    new.add_argument("--reviewed-commit", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--reviewed-commit", required=True)
    resume.add_argument("--attempt", required=True, type=_attempt_name_antigravity)
    commands.add_parser("inspect")
    commands.add_parser("evaluate")
    return parser


def _attempt_name_antigravity(value: str) -> str:
    if _ATTEMPT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("attempt must be attempt_####_antigravity")
    return value


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


def _publication_root_antigravity() -> Path:
    return Path(FROZEN_CONFIG.run_root) / "attempts_antigravity" / "attempt_0000_antigravity" / "publication_antigravity"


def _dispatch_antigravity(arguments: argparse.Namespace) -> object:
    if arguments.command == "readiness":
        return run_readiness_antigravity(arguments.reviewed_commit)
    if arguments.command == "new":
        validate_readiness_for_run_antigravity(arguments.reviewed_commit)
        outcome = run_new_antigravity(arguments.reviewed_commit)
        return {"runtime": outcome, "publication": publish_prediction_antigravity(outcome.attempt_path)}
    if arguments.command == "resume":
        outcome = run_resume_antigravity(arguments.reviewed_commit, arguments.attempt)
        return {"runtime": outcome, "publication": publish_prediction_antigravity(outcome.attempt_path)}
    if arguments.command == "inspect":
        return inspect_publication_antigravity(_publication_root_antigravity())
    if arguments.command == "evaluate":
        # This independent full replay deliberately precedes the only import
        # capable of reaching TEST labels.
        inspect_publication_antigravity(_publication_root_antigravity())
        evaluator = importlib.import_module("experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity")
        return evaluator.evaluate_frozen_publication_antigravity()
    raise ValueError("Exp444 CLI command is invalid")


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
