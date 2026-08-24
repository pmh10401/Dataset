"""Outside-CWD launcher for the fixed Exp468 Task 6 runtime.

The normal entry point forwards exactly one frozen ``evaluate`` or ``inspect``
command.  The sequence helper is intentionally explicit and is not called by
the launcher automatically; a separate user approval is required before any
private root is created.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-aeon140-py314/bin/python"
)
_FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-aeon140-py314"
)
_REEXEC_MARKER_ANTIGRAVITY = "EXP468_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
_RUN1_ROOT_ANTIGRAVITY = Path(
    "/private/tmp/exp468_task6_private_run1_antigravity",
)
_RUN2_ROOT_ANTIGRAVITY = Path(
    "/private/tmp/exp468_task6_private_run2_antigravity",
)
_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_SEQUENCE_SCHEMA_ANTIGRAVITY = "exp468_task6_private_sequence_v1"
_SEQUENCE_STAGES_ANTIGRAVITY = (
    "run1_evaluated",
    "run1_inspected",
    "run2_evaluated",
    "run2_inspected",
)
_SEQUENCE_STATE_PATH_ANTIGRAVITY = Path(
    "/private/tmp/exp468_task6_private_sequence_state_antigravity.json",
)
_SEQUENCE_NEXT_PATH_ANTIGRAVITY = Path(
    "/private/tmp/exp468_task6_private_sequence_next_antigravity.json",
)
_LEAF_NAMES_ANTIGRAVITY = {
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
}


def _ensure_fixed_runtime_antigravity() -> None:
    if (
        sys.executable == _FIXED_EXECUTABLE_ANTIGRAVITY
        and sys.prefix == _FIXED_PREFIX_ANTIGRAVITY
    ):
        return
    if os.environ.get(_REEXEC_MARKER_ANTIGRAVITY) == "1":
        raise RuntimeError("Exp468 fixed runtime re-exec failed")
    environment = dict(os.environ)
    environment[_REEXEC_MARKER_ANTIGRAVITY] = "1"
    os.execve(
        _FIXED_EXECUTABLE_ANTIGRAVITY,
        [_FIXED_EXECUTABLE_ANTIGRAVITY, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def _command_antigravity(argv: tuple[str, ...]) -> list[str]:
    if (
        len(argv) == 5
        and argv[0] == "evaluate"
        and argv[1] == "--output"
        and Path(argv[2]) in (_RUN1_ROOT_ANTIGRAVITY, _RUN2_ROOT_ANTIGRAVITY)
        and argv[3] == "--reviewed-commit"
        and _COMMIT_ANTIGRAVITY.fullmatch(argv[4]) is not None
    ):
        return [_FIXED_EXECUTABLE_ANTIGRAVITY, str(Path(__file__).resolve()), *argv]
    if (
        len(argv) == 3
        and argv[0] == "inspect"
        and argv[1] == "--root"
        and Path(argv[2]) in (_RUN1_ROOT_ANTIGRAVITY, _RUN2_ROOT_ANTIGRAVITY)
    ):
        return [_FIXED_EXECUTABLE_ANTIGRAVITY, str(Path(__file__).resolve()), *argv]
    raise ValueError("launcher command differs from the frozen Exp468 CLI surface")


def _is_absent_antigravity(path: Path) -> bool:
    return not path.exists() and not path.is_symlink()


def _has_exact_inventory_antigravity(root: Path) -> bool:
    if not root.is_dir() or root.is_symlink():
        return False
    entries = tuple(root.iterdir())
    return (
        {entry.name for entry in entries} == _LEAF_NAMES_ANTIGRAVITY
        and all(entry.is_file() and not entry.is_symlink() for entry in entries)
    )


def _read_sequence_state_antigravity() -> dict[str, str] | None:
    if _is_absent_antigravity(_SEQUENCE_STATE_PATH_ANTIGRAVITY):
        return None
    path = _SEQUENCE_STATE_PATH_ANTIGRAVITY
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("private sequence state must be a real file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("private sequence state is invalid") from error
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "stage", "reviewed_commit"}
        or payload.get("schema") != _SEQUENCE_SCHEMA_ANTIGRAVITY
        or payload.get("stage") not in _SEQUENCE_STAGES_ANTIGRAVITY
        or type(payload.get("reviewed_commit")) is not str
        or _COMMIT_ANTIGRAVITY.fullmatch(payload["reviewed_commit"]) is None
    ):
        raise RuntimeError("private sequence state is invalid")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if path.read_text(encoding="utf-8") != canonical:
        raise RuntimeError("private sequence state is not canonical")
    return payload


def _write_sequence_state_antigravity(stage: str, reviewed_commit: str) -> None:
    if stage not in _SEQUENCE_STAGES_ANTIGRAVITY or _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise RuntimeError("private sequence transition is invalid")
    if not _is_absent_antigravity(_SEQUENCE_NEXT_PATH_ANTIGRAVITY):
        raise RuntimeError("private sequence next-state path must be absent")
    payload = {
        "schema": _SEQUENCE_SCHEMA_ANTIGRAVITY,
        "stage": stage,
        "reviewed_commit": reviewed_commit,
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _SEQUENCE_NEXT_PATH_ANTIGRAVITY.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        _SEQUENCE_NEXT_PATH_ANTIGRAVITY,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    os.replace(_SEQUENCE_NEXT_PATH_ANTIGRAVITY, _SEQUENCE_STATE_PATH_ANTIGRAVITY)


def _validate_private_sequence_antigravity(
    selector: str,
    command: tuple[str, ...],
    reviewed_commit: str,
) -> None:
    from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.cli_antigravity import (
        _validate_clean_current_commit_antigravity,
    )

    _validate_clean_current_commit_antigravity(reviewed_commit)
    state = _read_sequence_state_antigravity()
    actual_stage = None if state is None else state["stage"]
    expected_stage = {
        ("run1", "evaluate"): None,
        ("run1", "inspect"): "run1_evaluated",
        ("run2", "evaluate"): "run1_inspected",
        ("run2", "inspect"): "run2_evaluated",
    }[(selector, command[0])]
    if actual_stage != expected_stage:
        raise RuntimeError(f"expected stage {expected_stage}, got {actual_stage}")
    if state is not None and state["reviewed_commit"] != reviewed_commit:
        raise RuntimeError("all private commands require the same reviewed commit")
    if selector == "run1" and command[0] == "evaluate":
        if not (
            _is_absent_antigravity(_RUN1_ROOT_ANTIGRAVITY)
            and _is_absent_antigravity(_RUN2_ROOT_ANTIGRAVITY)
            and _is_absent_antigravity(_SEQUENCE_STATE_PATH_ANTIGRAVITY)
        ):
            raise RuntimeError("run1 evaluate requires new private evidence paths")
    elif selector == "run1":
        if not (
            _has_exact_inventory_antigravity(_RUN1_ROOT_ANTIGRAVITY)
            and _is_absent_antigravity(_RUN2_ROOT_ANTIGRAVITY)
        ):
            raise RuntimeError("run1 inspect requires only the exact run1 result root")
    elif command[0] == "evaluate":
        if not (
            _has_exact_inventory_antigravity(_RUN1_ROOT_ANTIGRAVITY)
            and _is_absent_antigravity(_RUN2_ROOT_ANTIGRAVITY)
        ):
            raise RuntimeError("run2 evaluate requires an inspected run1 root")
    elif not (
        _has_exact_inventory_antigravity(_RUN1_ROOT_ANTIGRAVITY)
        and _has_exact_inventory_antigravity(_RUN2_ROOT_ANTIGRAVITY)
    ):
        raise RuntimeError("run2 inspect requires both exact private result roots")


def _advance_sequence_state_antigravity(
    selector: str,
    command: tuple[str, ...],
    reviewed_commit: str,
) -> None:
    next_stage = {
        ("run1", "evaluate"): "run1_evaluated",
        ("run1", "inspect"): "run1_inspected",
        ("run2", "evaluate"): "run2_evaluated",
        ("run2", "inspect"): "run2_inspected",
    }[(selector, command[0])]
    _write_sequence_state_antigravity(next_stage, reviewed_commit)


def run_private_sequence_antigravity(reviewed_commit: str) -> None:
    """Run the approved four-command order, stopping on the first failure."""

    if _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed commit must be lowercase 40-hex")
    commands = (
        (
            "evaluate",
            "--output",
            str(_RUN1_ROOT_ANTIGRAVITY),
            "--reviewed-commit",
            reviewed_commit,
        ),
        ("inspect", "--root", str(_RUN1_ROOT_ANTIGRAVITY)),
        (
            "evaluate",
            "--output",
            str(_RUN2_ROOT_ANTIGRAVITY),
            "--reviewed-commit",
            reviewed_commit,
        ),
        ("inspect", "--root", str(_RUN2_ROOT_ANTIGRAVITY)),
    )
    for index, argv in enumerate(commands):
        selector = "run1" if index < 2 else "run2"
        _validate_private_sequence_antigravity(selector, argv, reviewed_commit)
        command = _command_antigravity(argv)
        completed = subprocess.run(command, cwd=_REPOSITORY_ANTIGRAVITY, check=True)
        if getattr(completed, "returncode", 0) != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        _advance_sequence_state_antigravity(selector, argv, reviewed_commit)


def _run_antigravity() -> int:
    _ensure_fixed_runtime_antigravity()
    repository = str(_REPOSITORY_ANTIGRAVITY)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.cli_antigravity import (
        main_antigravity,
    )

    return main_antigravity()


if __name__ == "__main__":
    raise SystemExit(_run_antigravity())
