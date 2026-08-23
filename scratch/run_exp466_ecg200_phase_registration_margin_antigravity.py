"""Outside-CWD launcher that binds the frozen Exp466 interpreter and run order."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types


_FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-aeon140-py314/bin/python"
)
_FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/"
    "detection-coverage-antigravity-aeon140-py314"
)
_REEXEC_MARKER_ANTIGRAVITY = "EXP466_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"
_PRIVATE_CAPABILITY_ENV_ANTIGRAVITY = "EXP466_TASK6_PRIVATE_CAPABILITY_ANTIGRAVITY"
_PRIVATE_CAPABILITY_MODULE_ANTIGRAVITY = "_exp466_cli_private_test_antigravity"
_PRIVATE_CAPABILITY_VALUE_ANTIGRAVITY = "exp466_task6_private_capability_v1"
_PRIVATE_ROOTS_ANTIGRAVITY = {
    "run1": Path("/private/tmp/exp466_task6_private_run1_antigravity"),
    "run2": Path("/private/tmp/exp466_task6_private_run2_antigravity"),
}
_SEQUENCE_STATE_PATH_ANTIGRAVITY = Path(
    "/private/tmp/exp466_task6_private_sequence_state_antigravity.json",
)
_SEQUENCE_NEXT_PATH_ANTIGRAVITY = Path(
    "/private/tmp/exp466_task6_private_sequence_next_antigravity.json",
)
_SEQUENCE_SCHEMA_ANTIGRAVITY = "exp466_task6_private_sequence_antigravity_v1"
_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
_LEAVES_ANTIGRAVITY = {
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
}
_STAGES_ANTIGRAVITY = (
    "run1_evaluated",
    "run1_inspected",
    "run2_evaluated",
    "run2_inspected",
)


def _ensure_fixed_runtime_antigravity() -> None:
    if (
        sys.executable == _FIXED_EXECUTABLE_ANTIGRAVITY
        and sys.prefix == _FIXED_PREFIX_ANTIGRAVITY
    ):
        return
    if os.environ.get(_REEXEC_MARKER_ANTIGRAVITY) == "1":
        raise RuntimeError("Exp466 fixed runtime re-exec failed")
    environment = dict(os.environ)
    environment[_REEXEC_MARKER_ANTIGRAVITY] = "1"
    os.execve(
        _FIXED_EXECUTABLE_ANTIGRAVITY,
        [_FIXED_EXECUTABLE_ANTIGRAVITY, str(Path(__file__)), *sys.argv[1:]],
        environment,
    )


def _is_absent_antigravity(root: Path) -> bool:
    return not root.exists() and not root.is_symlink()


def _has_exact_inventory_antigravity(root: Path) -> bool:
    if not root.is_dir() or root.is_symlink():
        return False
    entries = tuple(root.iterdir())
    return (
        {entry.name for entry in entries} == _LEAVES_ANTIGRAVITY
        and all(entry.is_file() and not entry.is_symlink() for entry in entries)
    )


def _load_sequence_state_antigravity() -> dict[str, str] | None:
    path = _SEQUENCE_STATE_PATH_ANTIGRAVITY
    if _is_absent_antigravity(path):
        return None
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
        or payload.get("stage") not in _STAGES_ANTIGRAVITY
        or type(payload.get("reviewed_commit")) is not str
        or _COMMIT_ANTIGRAVITY.fullmatch(payload["reviewed_commit"]) is None
    ):
        raise RuntimeError("private sequence state is invalid")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    if path.read_text(encoding="utf-8") != canonical:
        raise RuntimeError("private sequence state is not canonical")
    return payload


def _write_sequence_state_antigravity(stage: str, reviewed_commit: str) -> None:
    if stage not in _STAGES_ANTIGRAVITY or _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise RuntimeError("private sequence transition is invalid")
    if not _is_absent_antigravity(_SEQUENCE_NEXT_PATH_ANTIGRAVITY):
        raise RuntimeError("private sequence next-state path must be absent")
    payload = {
        "schema": _SEQUENCE_SCHEMA_ANTIGRAVITY,
        "stage": stage,
        "reviewed_commit": reviewed_commit,
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        _SEQUENCE_NEXT_PATH_ANTIGRAVITY,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    os.replace(
        _SEQUENCE_NEXT_PATH_ANTIGRAVITY,
        _SEQUENCE_STATE_PATH_ANTIGRAVITY,
    )


def _current_clean_commit_antigravity() -> str:
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPOSITORY_ANTIGRAVITY,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain"),
        cwd=_REPOSITORY_ANTIGRAVITY,
        text=True,
    )
    if _COMMIT_ANTIGRAVITY.fullmatch(head) is None or dirty:
        raise RuntimeError("private sequence requires one clean reviewed commit")
    return head


def _command_antigravity(argv: tuple[str, ...]) -> str:
    if argv == ("inspect",):
        return "inspect"
    if (
        len(argv) == 3
        and argv[:2] == ("evaluate", "--reviewed-commit")
        and _COMMIT_ANTIGRAVITY.fullmatch(argv[2]) is not None
    ):
        return "evaluate"
    raise RuntimeError("launcher command differs from the frozen CLI surface")


def _expected_stage_antigravity(selector: str, command: str) -> str | None:
    return {
        ("run1", "evaluate"): None,
        ("run1", "inspect"): "run1_evaluated",
        ("run2", "evaluate"): "run1_inspected",
        ("run2", "inspect"): "run2_evaluated",
    }[(selector, command)]


def _validate_private_sequence_antigravity(
    selector: str,
    argv: tuple[str, ...],
) -> None:
    command = _command_antigravity(argv)
    state = _load_sequence_state_antigravity()
    expected_stage = _expected_stage_antigravity(selector, command)
    actual_stage = None if state is None else state["stage"]
    if actual_stage != expected_stage:
        if selector == "run2" and actual_stage in (None, "run1_evaluated"):
            raise RuntimeError("run1 inspect must complete before run2")
        raise RuntimeError(f"expected stage {expected_stage}, got {actual_stage}")

    run1 = _PRIVATE_ROOTS_ANTIGRAVITY["run1"]
    run2 = _PRIVATE_ROOTS_ANTIGRAVITY["run2"]
    if selector == "run1" and command == "evaluate":
        if not (
            _is_absent_antigravity(run1)
            and _is_absent_antigravity(run2)
            and _is_absent_antigravity(_SEQUENCE_STATE_PATH_ANTIGRAVITY)
        ):
            raise RuntimeError("run1 evaluate requires new private evidence paths")
        return

    if state is None:
        raise RuntimeError("private sequence state is required")
    reviewed_commit = state["reviewed_commit"]
    if _current_clean_commit_antigravity() != reviewed_commit:
        raise RuntimeError("all private commands require the same reviewed commit")
    if command == "evaluate" and argv[2] != reviewed_commit:
        raise RuntimeError("both private evaluations require the same reviewed commit")
    if selector == "run1":
        valid_roots = _has_exact_inventory_antigravity(run1) and _is_absent_antigravity(run2)
    elif command == "evaluate":
        valid_roots = _has_exact_inventory_antigravity(run1) and _is_absent_antigravity(run2)
    else:
        valid_roots = _has_exact_inventory_antigravity(run1) and _has_exact_inventory_antigravity(run2)
    if not valid_roots:
        raise RuntimeError("private result roots differ from the expected sequence stage")


def _advance_private_sequence_antigravity(
    selector: str,
    argv: tuple[str, ...],
) -> None:
    command = _command_antigravity(argv)
    state = _load_sequence_state_antigravity()
    expected_stage = _expected_stage_antigravity(selector, command)
    actual_stage = None if state is None else state["stage"]
    if actual_stage != expected_stage:
        raise RuntimeError("private sequence changed while a command was running")
    if selector == "run1" and command == "evaluate":
        reviewed_commit = argv[2]
        next_stage = "run1_evaluated"
    else:
        if state is None:
            raise RuntimeError("private sequence state is required")
        reviewed_commit = state["reviewed_commit"]
        next_stage = {
            ("run1", "inspect"): "run1_inspected",
            ("run2", "evaluate"): "run2_evaluated",
            ("run2", "inspect"): "run2_inspected",
        }[(selector, command)]
    _write_sequence_state_antigravity(next_stage, reviewed_commit)


def _install_private_capability_antigravity(
    argv: tuple[str, ...],
) -> str | None:
    selector = os.environ.get(_PRIVATE_CAPABILITY_ENV_ANTIGRAVITY)
    if selector is None:
        return None
    root = _PRIVATE_ROOTS_ANTIGRAVITY.get(selector)
    if root is None:
        raise RuntimeError("private capability is not preregistered")
    _validate_private_sequence_antigravity(selector, argv)
    private = types.ModuleType(_PRIVATE_CAPABILITY_MODULE_ANTIGRAVITY)
    private.CAPABILITY_ANTIGRAVITY = _PRIVATE_CAPABILITY_VALUE_ANTIGRAVITY
    private.root_antigravity = root
    sys.modules[_PRIVATE_CAPABILITY_MODULE_ANTIGRAVITY] = private
    return selector


def _load_main_antigravity():
    if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))
    from experiments.exp466_ecg200_phase_registration_margin_antigravity.cli_antigravity import (
        main_antigravity,
    )

    return main_antigravity


def _run_antigravity() -> int:
    _ensure_fixed_runtime_antigravity()
    argv = tuple(sys.argv[1:])
    selector = _install_private_capability_antigravity(argv)
    code = _load_main_antigravity()()
    if code == 0 and selector is not None:
        _advance_private_sequence_antigravity(selector, argv)
    return code


if __name__ == "__main__":
    raise SystemExit(_run_antigravity())
