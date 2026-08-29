#!/usr/bin/env python3
"""Approved-private-run helper for the closed Exp471 CLI; importing it performs no execution."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import LEAF_NAMES_ANTIGRAVITY
from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.cli_antigravity import main_antigravity


PRIVATE_RUN1_ANTIGRAVITY = Path("/private/tmp/exp471_task6_private_run1_antigravity")
PRIVATE_RUN2_ANTIGRAVITY = Path("/private/tmp/exp471_task6_private_run2_antigravity")
_APPROVAL_TOKEN_ANTIGRAVITY = "TASK_6_PRIVATE_RUN_1_2_APPROVED"
_COMMIT_RE_ANTIGRAVITY = re.compile(r"[0-9a-f]{40}\Z")
_ENTRYPOINT_ANTIGRAVITY = Path(__file__).resolve()


def _require_clean_reviewed_commit_antigravity(reviewed_commit: str) -> None:
    if _COMMIT_RE_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed commit must be exactly 40 lowercase hexadecimal characters")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != reviewed_commit:
        raise ValueError("reviewed commit is not the current HEAD")
    status = subprocess.run(
        ("git", "status", "--porcelain"), cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise ValueError("reviewed commit requires a clean worktree")


def _require_private_roots_absent_antigravity() -> None:
    for root in (PRIVATE_RUN1_ANTIGRAVITY, PRIVATE_RUN2_ANTIGRAVITY):
        if root.exists():
            raise ValueError("private roots must be absent before the approved sequence")


def _strict_json_object_antigravity(raw: bytes) -> dict[str, object]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("CLI stdout must be one strict JSON object") from error
    if type(value) is not dict or not decoded.endswith("\n"):
        raise ValueError("CLI stdout must be one strict JSON object")
    expected = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    if decoded != expected:
        raise ValueError("CLI stdout must be one canonical JSON object")
    return value


def _run_cli_antigravity(command: str, root: Path, reviewed_commit: str | None = None) -> dict[str, object]:
    args = [sys.executable, str(_ENTRYPOINT_ANTIGRAVITY), command, "--root", str(root)]
    if reviewed_commit is not None:
        args.extend(("--commit", reviewed_commit))
    completed = subprocess.run(args, cwd=_REPOSITORY_ROOT_ANTIGRAVITY, check=False, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{command} failed with exit code {completed.returncode}")
    return _strict_json_object_antigravity(completed.stdout)


def _evaluate_then_inspect_antigravity(root: Path, reviewed_commit: str) -> Path:
    if root.exists():
        raise ValueError("private evaluation root must be absent")
    _run_cli_antigravity("evaluate", root, reviewed_commit)
    _run_cli_antigravity("inspect", root)
    return root


def _require_byte_identical_antigravity(first: Path, second: Path) -> None:
    first_leaves = tuple(path.name for path in sorted(first.iterdir(), key=lambda path: path.name))
    second_leaves = tuple(path.name for path in sorted(second.iterdir(), key=lambda path: path.name))
    if first_leaves != LEAF_NAMES_ANTIGRAVITY or second_leaves != LEAF_NAMES_ANTIGRAVITY:
        raise ValueError("private results must contain exactly every publication leaf")
    for name in LEAF_NAMES_ANTIGRAVITY:
        first_leaf = first / name
        second_leaf = second / name
        if not first_leaf.is_file() or not second_leaf.is_file():
            raise ValueError("private results must contain every exact publication leaf")
        if first_leaf.read_bytes() != second_leaf.read_bytes():
            raise ValueError("private publication leaf bytes differ")


def run_private_sequence_antigravity(reviewed_commit: str, approval_token: str | None) -> tuple[Path, Path]:
    if approval_token != _APPROVAL_TOKEN_ANTIGRAVITY:
        raise PermissionError("private checkpoint approval is required")
    _require_clean_reviewed_commit_antigravity(reviewed_commit)
    _require_private_roots_absent_antigravity()
    first = _evaluate_then_inspect_antigravity(PRIVATE_RUN1_ANTIGRAVITY, reviewed_commit)
    _require_clean_reviewed_commit_antigravity(reviewed_commit)
    second = _evaluate_then_inspect_antigravity(PRIVATE_RUN2_ANTIGRAVITY, reviewed_commit)
    _require_byte_identical_antigravity(first, second)
    return first, second


if __name__ == "__main__":
    raise SystemExit(main_antigravity(tuple(sys.argv[1:])))
