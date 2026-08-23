"""Closed private-only command boundary for Exp466 Task 6."""

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
    "ECG200_TRAIN_antigravity.ts",
)
OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp466_ecg200_phase_registration_margin_antigravity_20260823",
)
_PRIVATE_ROOTS_ANTIGRAVITY = (
    Path("/private/tmp/exp466_task6_private_run1_antigravity"),
    Path("/private/tmp/exp466_task6_private_run2_antigravity"),
)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp466_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp466_task6_private_capability_v1"
_LAUNCHER_NAME_ANTIGRAVITY = (
    "run_exp466_ecg200_phase_registration_margin_antigravity.py"
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
        self.error(
            "only bare inspect or evaluate --reviewed-commit LOWER40 is accepted",
        )


def _reviewed_commit_antigravity(value: str) -> str:
    if _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("reviewed commit must be lowercase 40-hex")
    return value


def parser_antigravity() -> argparse.ArgumentParser:
    parser = _ExactSurfaceParserAntigravity(
        prog="run_exp466_ecg200_phase_registration_margin_antigravity",
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
        raise ValueError("fixed Task 6 private root capability is required")
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
        or root not in _PRIVATE_ROOTS_ANTIGRAVITY
        or root.is_symlink()
    ):
        raise ValueError("private test seam must select a fixed Task 6 private root")
    return root


def _validate_create_only_root_antigravity(root: Path) -> None:
    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("result root must be a non-symlink Path")
    if root.resolve(strict=False) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False):
        raise ValueError("result root must be a non-official Path")
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
    rows: list[tuple[int, int, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    parents = {pid: ppid for pid, ppid, _command in rows}
    ancestors: set[int] = set()
    current = os.getpid()
    while current > 0 and current not in ancestors:
        ancestors.add(current)
        current = parents.get(current, 0)
    active = [
        command
        for pid, _ppid, command in rows
        if pid not in ancestors
        and _launcher_token_matches_antigravity(command)
        and "test_cli_antigravity" not in command
        and "test_governance_antigravity" not in command
        and " -m unittest " not in command
        and "compileall" not in command
        and "build_codemap_exp466_antigravity.py" not in command
    ]
    if active:
        raise ValueError("duplicate Exp466 process is already active")


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
    _validate_official_root_absent_antigravity()
    _validate_no_duplicate_process_antigravity()


def _validate_official_root_absent_antigravity() -> None:
    if OFFICIAL_ROOT_ANTIGRAVITY.exists() or OFFICIAL_ROOT_ANTIGRAVITY.is_symlink():
        raise ValueError("official result root must be absent during Task 6")


def _blocked_fold_indices_antigravity() -> tuple[tuple[object, object], ...]:
    import numpy as np

    all_indices = np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64)
    folds = tuple(
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(all_indices, FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    if tuple(int(values.size) for values in folds) != (14, 14, 14, 14, 13):
        raise ValueError("frozen blocked folds do not cover 69 normal rows")
    result: list[tuple[object, object]] = []
    expected = set(range(FROZEN_CONFIG_ANTIGRAVITY.normal_rows))
    for query in folds:
        query_set = set(int(value) for value in query)
        fit = np.asarray(
            tuple(
                value
                for value in range(FROZEN_CONFIG_ANTIGRAVITY.normal_rows)
                if value not in query_set
            ),
            dtype=np.int64,
        )
        if set(int(value) for value in fit) | query_set != expected:
            raise ValueError("frozen blocked fold indices are invalid")
        result.append((fit, query))
    return tuple(result)


def _decision_payload_antigravity(decision: object) -> dict[str, object]:
    names = (
        "status",
        "primary_candidate",
        "eligible_for_exp467_design",
        "primary_passed_count",
        "primary_required_count",
        "failure_reasons",
        "routing_applied",
        "champion_modified",
        "pending_modified",
        "promotion_applied",
        "test_labels_opened",
        "label_query_count",
    )
    if any(not hasattr(decision, name) for name in names):
        raise ValueError("inspector must return the Exp466 decision")
    payload = {name: getattr(decision, name) for name in names}
    reasons = payload["failure_reasons"]
    if type(reasons) is not tuple:
        raise ValueError("inspector decision failure reasons must be immutable")
    payload["failure_reasons"] = list(reasons)
    if payload["label_query_count"] != 0 or payload["test_labels_opened"] is not False:
        raise ValueError("inspection opened labels")
    return payload


def _result_payload_from_inspected_antigravity(inspected: object) -> dict[str, object]:
    qualification = getattr(inspected, "qualification", None)
    decision = getattr(inspected, "decision", None)
    dashboard = getattr(inspected, "dashboard", None)
    leaf_sha256 = getattr(inspected, "leaf_sha256", None)
    artifact_map_sha256 = getattr(inspected, "artifact_map_sha256", None)
    if any(
        not hasattr(qualification, name)
        for name in (
            "normal_trial_counts",
            "normal_confirmations",
            "label_query_count",
            "official_evaluate_count",
            "official_inspect_count",
        )
    ):
        raise ValueError("inspector must return the Exp466 qualification")
    if type(dashboard) is not dict:
        raise ValueError("inspector must return the Exp466 dashboard")
    if type(leaf_sha256) is not tuple or len(leaf_sha256) != 7:
        raise ValueError("inspector must return the exact seven-leaf hashes")
    if type(artifact_map_sha256) is not str or re.fullmatch(
        r"[0-9a-f]{64}",
        artifact_map_sha256,
    ) is None:
        raise ValueError("inspector must return the artifact map hash")
    seeds = FROZEN_CONFIG_ANTIGRAVITY.seeds
    qualification_summary = {
        "normal_trial_counts": {
            str(seed): qualification.normal_trial_counts[seed] for seed in seeds
        },
        "normal_confirmations": {
            str(seed): qualification.normal_confirmations[seed] for seed in seeds
        },
        "label_query_count": qualification.label_query_count,
        "official_evaluate_count": qualification.official_evaluate_count,
        "official_inspect_count": qualification.official_inspect_count,
    }
    if (
        qualification_summary["label_query_count"] != 0
        or qualification_summary["official_evaluate_count"] != 0
        or qualification_summary["official_inspect_count"] != 0
    ):
        raise ValueError("inspection crossed the Task 6 authority boundary")
    leaf_hashes = dict(leaf_sha256)
    if len(leaf_hashes) != 7 or any(
        type(name) is not str
        or type(digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in leaf_hashes.items()
    ):
        raise ValueError("inspector returned invalid leaf hashes")
    return {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "kind": "ecg200_phase_registration_margin_train_qualification",
        "context": {
            "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
            "data_split": FROZEN_CONFIG_ANTIGRAVITY.data_split,
            "train_md5": FROZEN_CONFIG_ANTIGRAVITY.train_md5,
            "interpolated_width": FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
            "seeds": list(seeds),
            "folds": FROZEN_CONFIG_ANTIGRAVITY.folds,
            "method_order": list(FROZEN_CONFIG_ANTIGRAVITY.method_order),
            "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        },
        "qualification_summary": qualification_summary,
        "decision": _decision_payload_antigravity(decision),
        "dashboard_snapshot": dashboard,
        "leaf_sha256": leaf_hashes,
        "artifact_map_sha256": artifact_map_sha256,
    }


def _evaluate_at_root_antigravity(root: Path) -> dict[str, object]:
    """Run the frozen Task 2→5 flow once into a new private result root."""

    import numpy as np

    from .artifacts_antigravity import publish_result_antigravity
    from .evaluation_antigravity import evaluate_qualification_antigravity
    from .interpolation_antigravity import interpolate_to_140_antigravity
    from .predecessor_antigravity import capture_exp465_authority_antigravity
    from .qualification_antigravity import qualify_antigravity
    from .replay_antigravity import inspect_result_antigravity
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
    authority = capture_exp465_authority_antigravity()
    records: dict[tuple[int, int], object] = {}
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        for fold_index, (fit_indices, query_indices) in enumerate(
            _blocked_fold_indices_antigravity(),
        ):
            records[(seed, fold_index)] = score_fold_methods_antigravity(
                train140,
                fit_indices,
                query_indices,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    qualification = qualify_antigravity(records)
    decision = evaluate_qualification_antigravity(qualification)
    publication = publish_result_antigravity(root, records, qualification, decision)
    if (
        getattr(publication, "root", None) != root
        or len(getattr(publication, "leaf_sha256", ())) != 7
    ):
        raise ValueError("publisher must create the exact seven-leaf result")
    payload = _result_payload_from_inspected_antigravity(
        inspect_result_antigravity(root),
    )
    if payload["decision"]["label_query_count"] != 0:  # type: ignore[index]
        raise ValueError("published result opened labels")
    return payload


def _inspect_at_root_antigravity(root: Path) -> dict[str, object]:
    from .replay_antigravity import inspect_result_antigravity

    return _result_payload_from_inspected_antigravity(
        inspect_result_antigravity(root),
    )


def _load_evaluate_antigravity():
    return _evaluate_at_root_antigravity


def _load_inspect_antigravity():
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
    root = _private_test_root_antigravity()
    if args.command == "evaluate":
        _validate_create_only_root_antigravity(root)
        _validate_clean_current_commit_antigravity(args.reviewed_commit)
        payload = _load_evaluate_antigravity()(root)
    else:
        _validate_official_root_absent_antigravity()
        payload = _load_inspect_antigravity()(root)
    _emit_payload_antigravity(payload)
    return 0


__all__ = ("OFFICIAL_ROOT_ANTIGRAVITY", "main_antigravity", "parser_antigravity")
