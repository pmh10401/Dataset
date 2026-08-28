"""Closed, parameter-free Exp472 command surface.

Runtime construction remains deliberately outside the command parser.  The
scientific scorer and TEST-label loader are injected only by the controller
after its independent authority gates, so read-only commands cannot discover
or open them by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import TextIO

from .config_antigravity import (
    EXP471_REUSE_ROOT_ANTIGRAVITY,
    EXP472_EVALUATION_ROOT_ANTIGRAVITY,
    EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
    EXP472_REPORT_ROOT_ANTIGRAVITY,
    EXP472_ROOT_ANTIGRAVITY,
    EXP472_SMOKE_ROOT_ANTIGRAVITY,
    FROZEN_EXP472_CONFIG,
    assess_exp472_backend_antigravity,
)
from .contracts_antigravity import canonical_json_bytes_exp472_antigravity
from .reporting_antigravity import write_exp472_report_antigravity


_COMMANDS_ANTIGRAVITY = ("readiness", "smoke", "run", "status", "verify", "evaluate", "report")


def _emit_antigravity(stream: TextIO, value: object) -> None:
    stream.write(canonical_json_bytes_exp472_antigravity(value).decode("utf-8"))
    stream.flush()


def _paths_antigravity() -> dict[str, str]:
    return {
        "root": str(EXP472_ROOT_ANTIGRAVITY),
        "smoke": str(EXP472_SMOKE_ROOT_ANTIGRAVITY),
        "official": str(EXP472_OFFICIAL_ROOT_ANTIGRAVITY),
        "evaluation": str(EXP472_EVALUATION_ROOT_ANTIGRAVITY),
        "report": str(EXP472_REPORT_ROOT_ANTIGRAVITY),
        "progress": str(EXP472_OFFICIAL_ROOT_ANTIGRAVITY / "progress_antigravity.jsonl"),
    }


def _status_antigravity() -> dict[str, object]:
    """Produce a values-only observation without creating or importing data paths."""
    progress = EXP472_OFFICIAL_ROOT_ANTIGRAVITY / "progress_antigravity.jsonl"
    completed = failed = remaining = None
    eta_seconds = None
    if progress.is_file() and not progress.is_symlink():
        try:
            lines = progress.read_text(encoding="utf-8").splitlines()
            if lines:
                last = json.loads(lines[-1])
                if isinstance(last, dict):
                    completed = last.get("completed"); failed = last.get("failed")
                    remaining = last.get("remaining")
                    elapsed = last.get("elapsed_seconds")
                    if all(type(value) in (int, float) and value >= 0 for value in (completed, remaining, elapsed)) and completed:
                        eta_seconds = float(elapsed) / float(completed) * float(remaining)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            # Status reports a non-authoritative unreadable log rather than
            # attempting repair or a hidden restart.
            pass
    return {"experiment": "Exp472", "paths": _paths_antigravity(), "scored": completed, "failed": failed, "remaining": remaining, "eta_seconds": eta_seconds, "read_only": True}


def _call_runtime_antigravity(runtime: Mapping[str, object] | None, name: str) -> object:
    callback = runtime.get(name) if isinstance(runtime, Mapping) else None
    if not callable(callback):
        raise ValueError(f"Exp472 {name} requires the controller-bound runtime")
    return callback()


def _verified_authorities_antigravity(runtime: Mapping[str, object] | None) -> tuple[object, object]:
    """Open real controller-bound verifiers in the label-safe Exp471→E order.

    A boolean completion flag is intentionally not accepted as scientific
    evidence. The Task 5 public readers require their frozen Path/context/
    catalog inputs, so the controller binds zero-argument verifier closures.
    """
    if not isinstance(runtime, Mapping):
        raise ValueError("Exp472 evaluation requires all 1,117 verified E score shards")
    exp471_reader = runtime.get("verify_exp471")
    e_reader = runtime.get("verify_e")
    if not callable(exp471_reader):
        raise ValueError("Exp472 evaluation requires verified frozen Exp471 authority")
    if not callable(e_reader):
        raise ValueError("Exp472 evaluation requires all 1,117 verified E score shards")
    exp471 = exp471_reader()
    e = e_reader()
    dataset_order = tuple(getattr(e, "dataset_order", ()))
    if len(dataset_order) != 1117 or len(set(dataset_order)) != 1117:
        raise ValueError("Exp472 evaluation requires all 1,117 verified E score shards")
    return exp471, e


def _readiness_antigravity() -> dict[str, object]:
    backend = assess_exp472_backend_antigravity().artifact_record()
    return {"experiment": "Exp472", "database_sha256": FROZEN_EXP472_CONFIG.database_sha256, "dataset_count": 1117, "test_row_count": 195900, "paths": _paths_antigravity(), "backend": backend, "labels_opened": False}


def main_antigravity(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, runtime: Mapping[str, object] | None = None) -> int:
    """Run exactly one closed command; no root/parameter override is accepted."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1 or arguments[0] not in _COMMANDS_ANTIGRAVITY:
        raise ValueError("Exp472 command must be one of readiness, smoke, run, status, verify, evaluate, report")
    stream = sys.stdout if stdout is None else stdout
    command = arguments[0]
    if command == "status":
        _emit_antigravity(stream, _status_antigravity()); return 0
    if command == "readiness":
        _emit_antigravity(stream, _readiness_antigravity()); return 0
    if command == "smoke":
        result = _call_runtime_antigravity(runtime, "run_smoke")
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "smoke", "result": result}); return 0
    if command == "run":
        result = _call_runtime_antigravity(runtime, "run_full")
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "run", "result": result}); return 0
    if command == "verify":
        exp471, e = _verified_authorities_antigravity(runtime)
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "verify", "exp471_authority": type(exp471).__name__, "e_authority": type(e).__name__, "read_only": True}); return 0
    if command == "evaluate":
        _verified_authorities_antigravity(runtime)
        evaluation = _call_runtime_antigravity(runtime, "evaluate")
        _emit_antigravity(stream, {"experiment": "Exp472", "command": "evaluate", "evaluation_sha256": getattr(evaluation, "artifact_record", lambda: {})().get("evaluation_sha256")}); return 0
    # report deliberately does not dispatch score generation or labels.  It can
    # only consume a completed controller evaluation and its frozen bindings.
    _verified_authorities_antigravity(runtime)
    evaluation = runtime.get("evaluation") if isinstance(runtime, Mapping) else None
    worker = runtime.get("worker_record") if isinstance(runtime, Mapping) else None
    e_manifest = runtime.get("e_manifest_sha256") if isinstance(runtime, Mapping) else None
    lock_digest = runtime.get("code_map_lock_sha256") if isinstance(runtime, Mapping) else None
    if evaluation is None or not isinstance(worker, Mapping) or type(e_manifest) is not str or type(lock_digest) is not str:
        raise ValueError("Exp472 report requires a completed controller evaluation; it never starts E scoring")
    authority = write_exp472_report_antigravity(evaluation, EXP472_REPORT_ROOT_ANTIGRAVITY, worker, e_manifest_sha256=e_manifest, code_map_lock_sha256=lock_digest)
    _emit_antigravity(stream, {"experiment": "Exp472", "command": "report", "report_manifest_sha256": authority.report_manifest_sha256}); return 0


__all__ = ["main_antigravity"]
