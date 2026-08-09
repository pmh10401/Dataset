"""Fixed five-command interface for the reviewed Exp443 lifecycle."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
import importlib
import json
import math
from pathlib import Path
import re
import sys
from typing import TextIO

from .config_antigravity import FROZEN_CONFIG
from .readiness_antigravity import run_readiness_antigravity
from .reporting_antigravity import inspect_publication_antigravity
from .runtime_antigravity import resume_exp443_antigravity, run_exp443_new_antigravity


EVALUATION_MODULE_ANTIGRAVITY = (
    "experiments.exp443_missing_normal_train_policy_comparison_antigravity."
    "evaluation_antigravity"
)
_ATTEMPT_BASENAME_ANTIGRAVITY = re.compile(r"attempt_[0-9]{4}_antigravity")
_COMMIT_ANTIGRAVITY = re.compile(r"[0-9a-f]{40}")


class _CliUsageErrorAntigravity(ValueError):
    pass


class _ArgumentParserAntigravity(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CliUsageErrorAntigravity(message)


class CliDependenciesAntigravity:
    """Small injectable command seams; evaluator import remains explicit."""

    __slots__ = ("readiness", "new", "resume", "inspect", "import_module")

    def __init__(
        self,
        *,
        readiness: Callable[[str], object],
        new: Callable[[str], object],
        resume: Callable[[str, str], object],
        inspect: Callable[[Path], object],
        import_module: Callable[[str], object],
    ) -> None:
        values = {
            "readiness": readiness,
            "new": new,
            "resume": resume,
            "inspect": inspect,
            "import_module": import_module,
        }
        if any(not callable(value) for value in values.values()):
            raise ValueError("Exp443 CLI dependency is invalid")
        for name, value in values.items():
            object.__setattr__(self, name, value)


def _default_dependencies_antigravity() -> CliDependenciesAntigravity:
    return CliDependenciesAntigravity(
        readiness=lambda commit: run_readiness_antigravity(reviewed_commit=commit),
        new=lambda commit: run_exp443_new_antigravity(reviewed_commit=commit),
        resume=lambda commit, attempt: resume_exp443_antigravity(
            reviewed_commit=commit, attempt_name=attempt,
        ),
        inspect=inspect_publication_antigravity,
        import_module=importlib.import_module,
    )


def build_parser_antigravity() -> argparse.ArgumentParser:
    parser = _ArgumentParserAntigravity(
        prog="run_exp443_missing_normal_train_policy_comparison_antigravity.py",
        description="Run the fixed reviewed Exp443 lifecycle.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "new"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--reviewed-commit", required=True)
    resume = commands.add_parser("resume", allow_abbrev=False)
    resume.add_argument("--reviewed-commit", required=True)
    resume.add_argument("--attempt", required=True)
    commands.add_parser("inspect", allow_abbrev=False)
    commands.add_parser("evaluate", allow_abbrev=False)
    return parser


def _reviewed_commit_antigravity(value: object) -> str:
    if type(value) is not str or _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise _CliUsageErrorAntigravity("--reviewed-commit must be one lowercase 40-character Git commit")
    return value


def _attempt_name_antigravity(value: object) -> str:
    if type(value) is not str or _ATTEMPT_BASENAME_ANTIGRAVITY.fullmatch(value) is None:
        raise _CliUsageErrorAntigravity(
            "--attempt must match attempt_[0-9]{4}_antigravity and contain no path"
        )
    return value


def _attempt_path_antigravity() -> Path:
    return (
        Path(FROZEN_CONFIG.run_root)
        / "attempts_antigravity"
        / "attempt_0000_antigravity"
    )


def _json_safe_antigravity(value: object) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Exp443 CLI result contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_antigravity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe_antigravity(item) for item in value]
    serializer = getattr(value, "to_payload_antigravity", None)
    if callable(serializer):
        return _json_safe_antigravity(serializer())
    serializer = getattr(value, "to_payload", None)
    if callable(serializer):
        return _json_safe_antigravity(serializer())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe_antigravity(getattr(value, field.name))
            for field in fields(value)
        }
    raise ValueError(f"Exp443 CLI result is not JSON-safe: {type(value).__name__}")


def _write_json_antigravity(stream: TextIO, payload: Mapping[str, object]) -> None:
    stream.write(json.dumps(
        _json_safe_antigravity(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n")
    stream.flush()


def _run_command_antigravity(args: argparse.Namespace, dependencies: CliDependenciesAntigravity) -> object:
    if args.command == "readiness":
        return dependencies.readiness(_reviewed_commit_antigravity(args.reviewed_commit))
    if args.command == "new":
        return dependencies.new(_reviewed_commit_antigravity(args.reviewed_commit))
    if args.command == "resume":
        return dependencies.resume(
            _reviewed_commit_antigravity(args.reviewed_commit),
            _attempt_name_antigravity(args.attempt),
        )
    attempt = _attempt_path_antigravity()
    if args.command == "inspect":
        return dependencies.inspect(attempt)
    if args.command == "evaluate":
        # This inspection must complete before the only label-capable module is
        # imported. The evaluator independently repeats the inspection.
        dependencies.inspect(attempt)
        module = dependencies.import_module(EVALUATION_MODULE_ANTIGRAVITY)
        evaluator = getattr(module, "evaluate_publication_antigravity", None)
        if not callable(evaluator):
            raise ValueError("Exp443 evaluator entrypoint is unavailable")
        return evaluator(attempt)
    raise _CliUsageErrorAntigravity("Exp443 command is invalid")


def main_antigravity(
    argv: Sequence[str] | None = None,
    *,
    dependencies: CliDependenciesAntigravity | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    command: str | None = None
    try:
        try:
            args = build_parser_antigravity().parse_args(None if argv is None else list(argv))
        except SystemExit as exit_signal:
            return int(exit_signal.code or 0)
        command = args.command
        result = _run_command_antigravity(args, dependencies or _default_dependencies_antigravity())
        _write_json_antigravity(output, {
            "schema": "exp443.cli-status.v1",
            "command": command,
            "status": "success",
            "result": result,
        })
        return 0
    except KeyboardInterrupt as error:
        _write_json_antigravity(errors, {
            "schema": "exp443.cli-status.v1", "command": command,
            "status": "error", "error_type": type(error).__name__,
            "error": str(error) or "KeyboardInterrupt",
        })
        return 130
    except Exception as error:
        _write_json_antigravity(errors, {
            "schema": "exp443.cli-status.v1", "command": command,
            "status": "error", "error_type": type(error).__name__,
            "error": str(error) or type(error).__name__,
        })
        return 2 if isinstance(error, _CliUsageErrorAntigravity) else 1


__all__ = [
    "CliDependenciesAntigravity", "EVALUATION_MODULE_ANTIGRAVITY",
    "build_parser_antigravity", "main_antigravity",
]
