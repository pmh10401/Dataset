"""Closed private-only command boundary for Exp465 Task 5."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/"
    "ECG200_TRAIN_antigravity.ts"
)
OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp465_ecg200_c8_external_validation_antigravity_20260822"
)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp465_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp465_cli_private_test_antigravity_v1"
_LAUNCHER_NAME_ANTIGRAVITY = (
    "run_exp465_ecg200_c8_external_validation_antigravity.py"
)
_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "method_fold_metrics_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


class _ExactSurfaceParserAntigravity(argparse.ArgumentParser):
    """Reject every spelling outside the two preregistered commands."""

    def parse_args(self, args: Sequence[str] | None = None, namespace=None):
        raw = list(sys.argv[1:] if args is None else args)
        if raw == ["inspect"] or (
            len(raw) == 3
            and raw[:2] == ["evaluate", "--reviewed-commit"]
            and _COMMIT_ANTIGRAVITY.fullmatch(raw[2]) is not None
        ):
            return super().parse_args(raw, namespace)
        self.error("only bare inspect or evaluate --reviewed-commit LOWER40 is accepted")


def _reviewed_commit_antigravity(value: str) -> str:
    if _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("reviewed commit must be lowercase 40-hex")
    return value


def parser_antigravity() -> argparse.ArgumentParser:
    parser = _ExactSurfaceParserAntigravity(
        prog="run_exp465_ecg200_c8_external_validation_antigravity",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate.add_argument(
        "--reviewed-commit",
        required=True,
        type=_reviewed_commit_antigravity,
    )
    commands.add_parser("inspect", allow_abbrev=False)
    return parser


def _private_test_root_antigravity() -> Path:
    private = sys.modules.get(_PRIVATE_TEST_MODULE_ANTIGRAVITY)
    if (
        private is None
        or getattr(private, "CAPABILITY_ANTIGRAVITY", None)
        != _PRIVATE_TEST_CAPABILITY_ANTIGRAVITY
    ):
        raise ValueError("private Task 5 root capability is required")
    extra = sorted(
        name
        for name in vars(private)
        if not name.startswith("__")
        and name not in {"CAPABILITY_ANTIGRAVITY", "root_antigravity"}
    )
    if extra:
        raise ValueError("private test seam may inject only root_antigravity")
    root = getattr(private, "root_antigravity", None)
    if (
        not isinstance(root, Path)
        or root.resolve(strict=False) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False)
    ):
        raise ValueError("private test root must be a non-official Path")
    if root.is_symlink():
        raise ValueError("private test root must be a non-symlink Path")
    return root


def _validate_create_only_root_antigravity(root: Path) -> None:
    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("result root must be a non-symlink Path")
    if root.resolve(strict=False) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False):
        raise ValueError("private test root must be a non-official Path")
    if root.exists():
        raise ValueError("publication destination must be a new create-only path")


def _launcher_token_matches_antigravity(command: str) -> bool:
    return any(
        Path(token.strip("\"'")).name == _LAUNCHER_NAME_ANTIGRAVITY
        for token in command.split()
    )


def _validate_no_duplicate_process_antigravity() -> None:
    output = subprocess.check_output(
        ["ps", "-Ao", "pid=,ppid=,command="],
        text=True,
    )
    rows: list[tuple[int, int, str, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2], line))
        except ValueError:
            continue
    parents = {pid: ppid for pid, ppid, _command, _line in rows}
    ancestors: set[int] = set()
    current = os.getpid()
    while current > 0 and current not in ancestors:
        ancestors.add(current)
        current = parents.get(current, 0)
    active = [
        line
        for pid, _ppid, command, line in rows
        if pid not in ancestors
        and _launcher_token_matches_antigravity(command)
        and "test_cli_antigravity" not in command
        and " -m unittest " not in command
        and "compileall" not in command
        and "build_codemap_exp465_antigravity.py" not in command
    ]
    if active:
        raise ValueError("duplicate Exp465 process is already active")


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
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
    if reviewed_commit != head or dirty:
        raise ValueError("reviewed commit must be the exact clean current commit")
    if OFFICIAL_ROOT_ANTIGRAVITY.exists() or OFFICIAL_ROOT_ANTIGRAVITY.is_symlink():
        raise ValueError("official result root must be absent during Task 5")
    _validate_no_duplicate_process_antigravity()


def _json_leaf_antigravity(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_leaf_antigravity(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _result_payload_from_root_antigravity(root: Path, replayed: object) -> dict[str, object]:
    decision = getattr(replayed, "decision", None)
    if getattr(decision, "label_query_count", None) != 0:
        raise ValueError("inspection opened labels")
    leaf_sha256 = getattr(replayed, "leaf_sha256", None)
    artifact_map_sha256 = getattr(replayed, "artifact_map_sha256", None)
    if type(leaf_sha256) is not tuple or type(artifact_map_sha256) is not str:
        raise ValueError("inspector must return the Exp465 replay payload")
    return {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "kind": FROZEN_CONFIG_ANTIGRAVITY.kind,
        "context": _json_leaf_antigravity(root / "context_antigravity.json"),
        "method_fold_metrics": _jsonl_leaf_antigravity(root / "method_fold_metrics_antigravity.jsonl"),
        "failed_folds": _jsonl_leaf_antigravity(root / "failed_folds_antigravity.jsonl"),
        "method_summaries": _jsonl_leaf_antigravity(root / "method_summary_antigravity.jsonl"),
        "decision": _json_leaf_antigravity(root / "decision_antigravity.json"),
        "dashboard_snapshot": _json_leaf_antigravity(root / "dashboard_snapshot_antigravity.json"),
        "leaf_sha256": dict(leaf_sha256),
        "artifact_map_sha256": artifact_map_sha256,
        "artifact_sha256": _json_leaf_antigravity(root / "artifact_sha256_antigravity.json"),
    }


def _blocked_fold_indices_antigravity() -> tuple[tuple[object, object], ...]:
    import numpy as np

    all_indices = np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64)
    folds = tuple(np.asarray(values, dtype=np.int64) for values in np.array_split(all_indices, FROZEN_CONFIG_ANTIGRAVITY.folds))
    if tuple(int(values.size) for values in folds) != (14, 14, 14, 14, 13):
        raise ValueError("frozen blocked folds do not cover 69 normal rows")
    result: list[tuple[object, object]] = []
    for query in folds:
        fit = all_indices[~np.isin(all_indices, query)]
        if int(fit.size) + int(query.size) != FROZEN_CONFIG_ANTIGRAVITY.normal_rows:
            raise ValueError("blocked fold indices are invalid")
        result.append((fit, query))
    return tuple(result)


def _evaluate_at_root_antigravity(root: Path) -> dict[str, object]:
    """Run the frozen Task 2→4 flow once into a new private result root."""
    import numpy as np

    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
        capture_exp463_authority_antigravity,
    )

    from .artifacts_antigravity import PublishedQualificationAntigravity, publish_result_antigravity
    from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_qualification_antigravity
    from .interpolation_antigravity import interpolate_to_140_antigravity
    from .qualification_antigravity import Exp465QualificationResultAntigravity, qualify_methods_antigravity
    from .replay_antigravity import inspect_replay_at_antigravity
    from .scoring_antigravity import score_fold_methods_antigravity
    from .sources_antigravity import load_ecg200_normal_train_antigravity

    _validate_create_only_root_antigravity(root)
    normal96 = load_ecg200_normal_train_antigravity(_TRAIN_PATH_ANTIGRAVITY)
    train140 = interpolate_to_140_antigravity(normal96)
    if train140.shape != (
        FROZEN_CONFIG_ANTIGRAVITY.normal_rows,
        FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
    ) or not np.isfinite(train140).all():
        raise ValueError("frozen ECG200 TRAIN interpolation output is invalid")
    authority = capture_exp463_authority_antigravity()
    records: list[tuple[int, int, object]] = []
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        for fold_index, (fit_indices, query_indices) in enumerate(_blocked_fold_indices_antigravity()):
            records.append((
                seed,
                fold_index,
                score_fold_methods_antigravity(
                    train140,
                    fit_indices,
                    query_indices,
                    seed=seed,
                    fold_index=fold_index,
                    authority=authority,
                ),
            ))
    qualification = qualify_methods_antigravity(tuple(records))
    if type(qualification) is not Exp465QualificationResultAntigravity:
        raise ValueError("qualifier must return the exact Exp465 qualification")
    decision = evaluate_qualification_antigravity(qualification)
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("evaluator must return the exact Exp465 decision")
    publication = publish_result_antigravity(root, qualification, decision)
    if (
        type(publication) is not PublishedQualificationAntigravity
        or publication.root != root
        or len(publication.leaf_sha256) != len(_LEAVES_ANTIGRAVITY)
    ):
        raise ValueError("publisher must create the exact seven-leaf result")
    payload = _result_payload_from_root_antigravity(root, inspect_replay_at_antigravity(root))
    if payload["decision"]["label_query_count"] != 0:  # type: ignore[index]
        raise ValueError("published result opened labels")
    return payload


def _inspect_at_root_antigravity(root: Path) -> dict[str, object]:
    from .replay_antigravity import inspect_replay_at_antigravity

    return _result_payload_from_root_antigravity(root, inspect_replay_at_antigravity(root))


def _load_evaluate_antigravity():
    return _evaluate_at_root_antigravity


def _load_inspect_antigravity():
    return _inspect_at_root_antigravity


def _emit_payload_antigravity(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    root = _private_test_root_antigravity()
    if args.command == "evaluate":
        _validate_create_only_root_antigravity(root)
        _validate_clean_current_commit_antigravity(args.reviewed_commit)
        payload = _load_evaluate_antigravity()(root)
    else:
        payload = _load_inspect_antigravity()(root)
    _emit_payload_antigravity(payload)
    return 0


__all__ = ("OFFICIAL_ROOT_ANTIGRAVITY", "main_antigravity", "parser_antigravity")
