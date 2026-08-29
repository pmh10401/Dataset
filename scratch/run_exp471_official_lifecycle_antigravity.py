"""One-shot approved Exp471 official TRAIN evaluate→inspect lifecycle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform
import re
import subprocess
import sys
from unittest import mock


_COMMIT_RE_ANTIGRAVITY = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp471_ecg200_acf_peak_lag_shift_numba_antigravity_20260825"
)
_RUN1_ROOT_ANTIGRAVITY = Path("/private/tmp/exp471_task6_private_run1_antigravity")
_LEAF_NAMES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)
_REQUIRED_TOKEN_ANTIGRAVITY = "TASK6_OFFICIAL_EVALUATE_INSPECT_APPROVED"


def _head_commit_antigravity() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _require_clean_head_antigravity(reviewed_commit: str) -> None:
    if _COMMIT_RE_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise SystemExit("reviewed commit must be 40 lowercase hex characters")
    if _head_commit_antigravity() != reviewed_commit:
        raise SystemExit("reviewed commit differs from current HEAD")
    dirty = subprocess.run(
        ("git", "status", "--porcelain"), cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise SystemExit("official lifecycle requires a clean worktree")


def _leaf_hashes_antigravity(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in _LEAF_NAMES_ANTIGRAVITY
    }


def main_antigravity(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--approval-token", required=True)
    args = parser.parse_args(argv)
    if args.approval_token != _REQUIRED_TOKEN_ANTIGRAVITY:
        raise SystemExit("official lifecycle token rejected")

    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity import cli_antigravity

    reviewed = _head_commit_antigravity()
    _require_clean_head_antigravity(reviewed)
    if _OFFICIAL_ROOT_ANTIGRAVITY.exists() or _OFFICIAL_ROOT_ANTIGRAVITY.is_symlink():
        raise SystemExit("official root must be absent before the approved lifecycle")
    if not _RUN1_ROOT_ANTIGRAVITY.is_dir():
        raise SystemExit("private run1 root must exist for post-run equality proof")

    allowed = (*cli_antigravity._PRIVATE_ROOTS_ANTIGRAVITY, _OFFICIAL_ROOT_ANTIGRAVITY)
    with mock.patch.object(cli_antigravity, "_PRIVATE_ROOTS_ANTIGRAVITY", allowed):
        evaluate_argv = (
            "evaluate", "--root", str(_OFFICIAL_ROOT_ANTIGRAVITY), "--commit", reviewed,
        )
        cli_antigravity.main_antigravity(evaluate_argv)
        inspect_argv = ("inspect", "--root", str(_OFFICIAL_ROOT_ANTIGRAVITY))
        cli_antigravity.main_antigravity(inspect_argv)

    mismatch = [
        name for name in _LEAF_NAMES_ANTIGRAVITY
        if (_OFFICIAL_ROOT_ANTIGRAVITY / name).read_bytes()
        != (_RUN1_ROOT_ANTIGRAVITY / name).read_bytes()
    ]
    if mismatch:
        raise SystemExit(f"official leaves differ from private run1: {mismatch}")

    print(
        '{"official_root": ' + json_str(_OFFICIAL_ROOT_ANTIGRAVITY)
        + ', "reviewed_commit": "' + reviewed + '"'
        + ', "leaves_equal_to_run1": true'
        + ', "runtime_python": "' + platform.python_version() + '"}'
    )
    return 0


def json_str(value: Path) -> str:
    import json
    return json.dumps(str(value))


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
