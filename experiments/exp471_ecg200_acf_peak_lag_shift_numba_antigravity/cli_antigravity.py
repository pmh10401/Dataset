"""Closed Exp471 synthetic evaluation and inspection CLI; no private or official execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

from .artifacts_antigravity import publish_result_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .evaluation_antigravity import evaluate_repair_gates_antigravity
from .interpolation_antigravity import interpolate_to_140_antigravity
from .predecessor_antigravity import capture_exp465_authority_antigravity
from .qualification_antigravity import blocked_fold_indices_antigravity, qualify_records_antigravity
from .replay_antigravity import inspect_result_antigravity
from .scoring_antigravity import score_fold_methods_antigravity
from .sources_antigravity import load_ecg200_normal_train_antigravity


_COMMIT_RE_ANTIGRAVITY = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/ECG200_TRAIN_antigravity.ts"
)
PRIVATE_RUN1_ANTIGRAVITY = Path("/private/tmp/exp471_task6_private_run1_antigravity")
PRIVATE_RUN2_ANTIGRAVITY = Path("/private/tmp/exp471_task6_private_run2_antigravity")
_PRIVATE_ROOTS_ANTIGRAVITY = (PRIVATE_RUN1_ANTIGRAVITY, PRIVATE_RUN2_ANTIGRAVITY)


def _commit_antigravity(value: str) -> str:
    if _COMMIT_RE_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("commit must be exactly 40 lowercase hexadecimal characters")
    return value


def _path_antigravity(value: str) -> Path:
    if not value:
        raise argparse.ArgumentTypeError("root path must not be empty")
    path = Path(value)
    if path not in _PRIVATE_ROOTS_ANTIGRAVITY:
        raise argparse.ArgumentTypeError("root must be one of the two fixed Task 6 private roots")
    return path


def parser_antigravity() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exp471", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate.add_argument("--root", required=True, type=_path_antigravity)
    evaluate.add_argument("--commit", required=True, type=_commit_antigravity)
    inspect = commands.add_parser("inspect", allow_abbrev=False)
    inspect.add_argument("--root", required=True, type=_path_antigravity)
    return parser


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


def _require_create_only_root_antigravity(root: Path) -> None:
    if root.is_symlink():
        raise ValueError("evaluation root must not be a symlink")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("evaluation root must be absent or an empty directory")


def _load_frozen_train_antigravity() -> object:
    return load_ecg200_normal_train_antigravity(_TRAIN_PATH_ANTIGRAVITY)


def _decision_payload_antigravity(decision: object) -> dict[str, object]:
    return {
        "eligible_for_private_reproducibility": bool(decision.eligible_for_private_reproducibility),
        "failure_reasons": list(decision.failure_reasons),
        "passed": bool(decision.passed),
    }


def _inspected_payload_antigravity(root: Path, inspected: object) -> dict[str, object]:
    return {
        "decision": _decision_payload_antigravity(inspected.decision),
        "hashes": dict(inspected.hashes),
        "root": str(root),
    }


def evaluate_pipeline_antigravity(
    root: Path,
    reviewed_commit: str,
    *,
    loader: Callable[[], object] = _load_frozen_train_antigravity,
    interpolate: Callable[[object], object] = interpolate_to_140_antigravity,
    authority_factory: Callable[[], object] = capture_exp465_authority_antigravity,
    scorer: Callable[..., object] = score_fold_methods_antigravity,
    qualifier: Callable[[object], object] = qualify_records_antigravity,
    decider: Callable[[object], object] = evaluate_repair_gates_antigravity,
    publisher: Callable[[Path, object, object, object], object] = publish_result_antigravity,
    inspector: Callable[[Path], object] = inspect_result_antigravity,
) -> dict[str, object]:
    """Run the frozen normal-TRAIN pipeline once and publish only its seven canonical leaves."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if _COMMIT_RE_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed commit must be exactly 40 lowercase hexadecimal characters")
    if root not in _PRIVATE_ROOTS_ANTIGRAVITY:
        raise ValueError("evaluation root must be one of the two fixed Task 6 private roots")
    _require_create_only_root_antigravity(root)
    source = loader()
    train140 = interpolate(source)
    authority = authority_factory()
    records: dict[tuple[int, int], object] = {}
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        for fold_index, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
            records[(seed, fold_index)] = scorer(
                train140, fit, query, seed=seed, fold_index=fold_index, authority=authority
            )
    qualification = qualifier(records)
    decision = decider(qualification)
    published = publisher(root, records, qualification, decision)
    inspected = inspector(root)
    payload = {
        "decision": _decision_payload_antigravity(decision),
        "hashes": dict(published.hashes),
        "root": str(root),
    }
    if inspected.decision != decision or dict(inspected.hashes) != dict(published.hashes):
        raise ValueError("semantic replay disagrees with the published result")
    return payload


def main_antigravity(argv: tuple[str, ...] | None = None) -> int:
    args = parser_antigravity().parse_args(argv)
    if args.command == "evaluate":
        _require_clean_reviewed_commit_antigravity(args.commit)
        payload = evaluate_pipeline_antigravity(args.root, args.commit)
    else:
        payload = _inspected_payload_antigravity(args.root, inspect_result_antigravity(args.root))
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
