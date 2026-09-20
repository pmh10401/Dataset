"""Closed Task 6 CLI for the Exp468 TRAIN-only qualification boundary.

The public command surface is intentionally small:

``evaluate --output PATH --reviewed-commit LOWER40HEX``
``inspect --root PATH``

Only the two preregistered private roots are accepted.  Evaluation imports the
scientific pipeline lazily and creates one new seven-leaf publication.  Inspect
imports only the retained-input replay boundary and therefore never loads the
ECG200 source, interpolation adapter, or a private pipeline.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import fields
import json
from pathlib import Path
import re
import subprocess
import sys


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/"
    "ECG200_TRAIN_antigravity.ts",
)
OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp468_ecg200_local_stft_spectral_redistribution_antigravity_20260824",
)
_PRIVATE_ROOTS_ANTIGRAVITY = (
    Path("/private/tmp/exp468_task6_private_run1_antigravity"),
    Path("/private/tmp/exp468_task6_private_run2_antigravity"),
)
_LEAF_NAMES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)


class _ExactSurfaceParserAntigravity(argparse.ArgumentParser):
    """Reject every command or option not present in the frozen contract."""

    def parse_args(self, args: Sequence[str] | None = None, namespace=None):
        raw = list(sys.argv[1:] if args is None else args)
        valid_evaluate = (
            len(raw) == 5
            and raw[0] == "evaluate"
            and raw[1] == "--output"
            and raw[3] == "--reviewed-commit"
            and _COMMIT_ANTIGRAVITY.fullmatch(raw[4]) is not None
        )
        valid_inspect = (
            len(raw) == 3
            and raw[0] == "inspect"
            and raw[1] == "--root"
        )
        if valid_evaluate or valid_inspect:
            return super().parse_args(raw, namespace)
        self.error(
            "only evaluate --output PATH --reviewed-commit LOWER40HEX or "
            "inspect --root PATH is accepted",
        )


def _reviewed_commit_antigravity(value: str) -> str:
    if _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("reviewed commit must be lowercase 40-hex")
    return value


def parser_antigravity() -> argparse.ArgumentParser:
    parser = _ExactSurfaceParserAntigravity(
        prog="run_exp468_ecg200_local_stft_spectral_redistribution_antigravity",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--reviewed-commit",
        required=True,
        type=_reviewed_commit_antigravity,
    )
    inspect = commands.add_parser("inspect", allow_abbrev=False)
    inspect.add_argument("--root", required=True)
    return parser


def _validate_output_root_antigravity(root: Path) -> None:
    if isinstance(root, Path) and root.is_symlink():
        raise ValueError("result root must not be a symlink")
    if not isinstance(root, Path) or root not in _PRIVATE_ROOTS_ANTIGRAVITY:
        raise ValueError("output must be one of the fixed Task 6 private roots")
    if root.exists():
        raise ValueError("publication destination must be a new create-only path")


def _validate_inspect_root_antigravity(root: Path) -> None:
    if isinstance(root, Path) and root.is_symlink():
        raise ValueError("result root must not be a symlink")
    if not isinstance(root, Path) or root not in _PRIVATE_ROOTS_ANTIGRAVITY:
        raise ValueError("root must be one of the fixed Task 6 private roots")
    if not root.exists() or not root.is_dir():
        raise ValueError("inspect root must be an existing private result directory")


def _validate_official_root_absent_antigravity() -> None:
    if OFFICIAL_ROOT_ANTIGRAVITY.exists() or OFFICIAL_ROOT_ANTIGRAVITY.is_symlink():
        raise ValueError("official result root must be absent during Task 6")


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    if _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed commit must be lowercase 40-hex")
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


def _load_pipeline_dependencies_antigravity() -> tuple[object, ...]:
    """Import evaluation-only dependencies after all CLI/root checks."""

    import numpy as np

    from .artifacts_antigravity import publish_result_antigravity
    from .evaluation_antigravity import evaluate_qualification_antigravity
    from .interpolation_antigravity import interpolate_to_140_antigravity
    from .predecessor_antigravity import capture_exp465_authority_antigravity
    from .qualification_antigravity import (
        blocked_fold_indices_antigravity,
        qualify_antigravity,
    )
    from .replay_antigravity import inspect_result_antigravity
    from .scoring_antigravity import score_fold_methods_antigravity
    from .sources_antigravity import load_ecg200_normal_train_antigravity

    return (
        load_ecg200_normal_train_antigravity,
        interpolate_to_140_antigravity,
        capture_exp465_authority_antigravity,
        blocked_fold_indices_antigravity,
        np,
        score_fold_methods_antigravity,
        qualify_antigravity,
        evaluate_qualification_antigravity,
        publish_result_antigravity,
        inspect_result_antigravity,
    )


def _qualification_payload_antigravity(qualification: object) -> dict[str, object]:
    expected = (
        "fold_results",
        "normal_trial_counts",
        "normal_confirmations",
        "label_query_count",
        "artifact_map_sha256",
    )
    if qualification is None or any(
        not hasattr(qualification, name) for name in expected
    ):
        raise ValueError("inspector must return the exact Exp468 qualification")
    rows = []
    for row in qualification.fold_results:
        rows.append(
            {
                "all_scores_finite": row.all_scores_finite,
                "c0_per_type_auc": list(row.c0_per_type_auc),
                "c10_per_type_auc": list(row.c10_per_type_auc),
                "failure_reasons": list(row.failure_reasons),
                "fold_index": row.fold_index,
                "normal_confirmations": row.normal_confirmations,
                "passed": row.passed,
                "period_confirmations": row.period_confirmations,
                "seed": row.seed,
            },
        )
    return {
        "artifact_map_sha256": qualification.artifact_map_sha256,
        "fold_results": rows,
        "label_query_count": qualification.label_query_count,
        "normal_confirmations": {
            str(seed): qualification.normal_confirmations[seed]
            for seed in qualification.normal_confirmations
        },
        "normal_trial_counts": {
            str(seed): qualification.normal_trial_counts[seed]
            for seed in qualification.normal_trial_counts
        },
    }


def _decision_payload_antigravity(decision: object) -> dict[str, object]:
    names = tuple(field.name for field in fields(decision))
    if not names or "status" not in names:
        raise ValueError("inspector must return the exact Exp468 decision")
    payload = {name: getattr(decision, name) for name in names}
    if type(payload.get("failure_reasons")) is tuple:
        payload["failure_reasons"] = list(payload["failure_reasons"])
    if payload.get("label_query_count") != 0 or payload.get("test_labels_opened") is not False:
        raise ValueError("inspection crossed the label-free authority boundary")
    return payload


def _artifact_hashes_antigravity(root: Path) -> dict[str, str]:
    path = root / "artifact_sha256_antigravity.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError("artifact hash map must be an object")
    expected = set(_LEAF_NAMES_ANTIGRAVITY) - {"artifact_sha256_antigravity.json"}
    if set(payload) != expected:
        raise ValueError("artifact hash map must contain exactly six non-self leaves")
    return {name: payload[name] for name in sorted(payload)}


def _payload_from_inspected_antigravity(inspected: object) -> dict[str, object]:
    root = getattr(inspected, "root", None)
    qualification = getattr(inspected, "qualification", None)
    decision = getattr(inspected, "decision", None)
    dashboard = getattr(inspected, "dashboard_snapshot", None)
    if not isinstance(root, Path) or type(dashboard) is not dict and not hasattr(dashboard, "items"):
        raise ValueError("inspector must return the Exp468 result boundary")
    dashboard_payload = dict(dashboard)
    private = dashboard_payload.get("private_reproducibility")
    if type(private) is not dict and not hasattr(private, "items"):
        raise ValueError("dashboard snapshot is missing private status")
    private_payload = dict(private)
    if private_payload.get("comparison_status") != "not_yet_compared":
        raise ValueError("private comparison status must remain not_yet_compared")
    return {
        "artifact_sha256": _artifact_hashes_antigravity(root),
        "context": {
            "data_split": "TRAIN_only",
            "dataset": "ECG200",
            "folds": 5,
            "interpolated_width": 140,
            "method_order": ["C0", "S10", "C10"],
            "normal_rows": 69,
            "primary_candidate": "C10",
            "seeds": [20260721, 20261721],
            "train_md5": "808875aeac1c358a90c630fe369327c4",
        },
        "dashboard_snapshot": dashboard_payload,
        "decision": _decision_payload_antigravity(decision),
        "experiment": "Exp468",
        "kind": "ecg200_local_stft_train_qualification",
        "qualification": _qualification_payload_antigravity(qualification),
    }


def _evaluate_at_root_antigravity(root: Path) -> dict[str, object]:
    """Run the frozen TRAIN-only 2x5 flow once and publish seven leaves."""

    _validate_output_root_antigravity(root)
    (
        load_train,
        interpolate,
        capture_authority,
        blocked_folds,
        np,
        score_fold,
        qualify,
        evaluate,
        publish,
        inspect,
    ) = _load_pipeline_dependencies_antigravity()
    normal96 = load_train(_TRAIN_PATH_ANTIGRAVITY)
    if (
        type(normal96) is not np.ndarray
        or normal96.shape != (69, 96)
        or normal96.dtype != np.dtype(np.float64)
        or not np.isfinite(normal96).all()
    ):
        raise ValueError("ECG200 TRAIN normal input must be finite 69x96 float64")
    train140 = interpolate(normal96)
    if (
        type(train140) is not np.ndarray
        or train140.shape != (69, 140)
        or train140.dtype != np.dtype(np.float64)
        or not np.isfinite(train140).all()
    ):
        raise ValueError("ECG200 TRAIN interpolation output must be finite 69x140 float64")
    authority = capture_authority()
    folds = tuple(blocked_folds())
    if tuple(len(query) for _fit, query in folds) != (14, 14, 14, 14, 13):
        raise ValueError("frozen blocked query folds have drifted")
    if tuple(len(fit) for fit, _query in folds) != (55, 55, 55, 55, 56):
        raise ValueError("frozen blocked fit folds have drifted")
    records: dict[tuple[int, int], object] = {}
    for seed in (20260721, 20261721):
        for fold_index, (fit_indices, query_indices) in enumerate(folds):
            records[(seed, fold_index)] = score_fold(
                train140,
                fit_indices,
                query_indices,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    if tuple(records) != tuple(
        (seed, fold_index)
        for seed in (20260721, 20261721)
        for fold_index in range(5)
    ):
        raise ValueError("records must preserve seed-major 2x5 order")
    qualification = qualify(records)
    decision = evaluate(qualification)
    published = publish(root, records, qualification, decision)
    if getattr(published, "root", None) != root:
        raise ValueError("publisher returned a different result root")
    inspected = inspect(root)
    return _payload_from_inspected_antigravity(inspected)


def _inspect_at_root_antigravity(root: Path) -> dict[str, object]:
    """Inspect the seven-leaf root through Task 5 retained-input replay only."""

    _validate_inspect_root_antigravity(root)
    # This delayed import intentionally enters replay only.  Replay validates
    # retained raw inputs and scalar projections but never loads ECG200 TRAIN.
    inspected = _load_replay_inspector_antigravity()(root)
    return _payload_from_inspected_antigravity(inspected)


def _load_replay_inspector_antigravity() -> Callable[[Path], object]:
    """Load only the retained-input inspector at the inspect boundary."""

    from .replay_antigravity import inspect_result_antigravity

    return inspect_result_antigravity


def _load_evaluate_antigravity() -> Callable[[Path], dict[str, object]]:
    return _evaluate_at_root_antigravity


def _load_inspect_antigravity() -> Callable[[Path], dict[str, object]]:
    return _inspect_at_root_antigravity


def _emit_payload_antigravity(payload: dict[str, object]) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    _validate_official_root_absent_antigravity()
    if args.command == "evaluate":
        root = Path(args.output)
        _validate_output_root_antigravity(root)
        _validate_clean_current_commit_antigravity(args.reviewed_commit)
        payload = _load_evaluate_antigravity()(root)
    else:
        root = Path(args.root)
        _validate_inspect_root_antigravity(root)
        payload = _load_inspect_antigravity()(root)
    _emit_payload_antigravity(payload)
    return 0


__all__ = (
    "OFFICIAL_ROOT_ANTIGRAVITY",
    "main_antigravity",
    "parser_antigravity",
)
