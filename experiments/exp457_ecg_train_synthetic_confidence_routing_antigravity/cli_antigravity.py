"""Closed lazy command boundary for the reviewed Exp457 candidate."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import re
import subprocess
import sys


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp457_ecg_train_synthetic_confidence_routing_antigravity_20260818"
)


class _ExactSurfaceParserAntigravity(argparse.ArgumentParser):
    """Reject every token sequence outside the two frozen command forms."""

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
    """Expose exactly one reviewed evaluation form and one bare replay form."""
    parser = _ExactSurfaceParserAntigravity(
        prog="run_exp457_ecg_train_synthetic_confidence_routing_antigravity",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate.add_argument(
        "--reviewed-commit", required=True, type=_reviewed_commit_antigravity
    )
    commands.add_parser("inspect", allow_abbrev=False)
    return parser


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    """Bind the official write to the exact clean reviewed commit."""
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=_REPOSITORY_ANTIGRAVITY, text=True
    ).strip()
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=_REPOSITORY_ANTIGRAVITY, text=True
    )
    if reviewed_commit != head or dirty:
        raise ValueError("reviewed commit must be the exact clean current commit")


def emit_dashboard_json_antigravity(
    result: object,
    export_dashboard: Callable[[object], dict[str, object]],
) -> None:
    """Emit only the validated aggregate/TRAIN-gate dashboard payload."""
    print(
        json.dumps(
            export_dashboard(result), sort_keys=True, separators=(",", ":")
        )
    )


def _evaluate_with_dependencies_antigravity(
    reviewed_commit: str,
    validate_clean_commit: Callable[[str], None],
    qualify_all: Callable[[], tuple[object, ...]],
    capture_authority: Callable[[], object],
    route_ecg: Callable[[object, object], object],
    evaluate_routes: Callable[[object, tuple[object, ...]], object],
    publish_result: Callable[[Path, object], object],
    root: Path,
) -> tuple[object, object]:
    """Run the exact Gate 1 → route → aggregate → publication candidate flow."""
    from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
    from .contracts_antigravity import (
        Exp456ResultAuthorityAntigravity,
        PublicationEvidenceAntigravity,
        TrainQualificationAntigravity,
    )

    validate_clean_commit(reviewed_commit)
    qualifications = qualify_all()
    if (
        type(qualifications) is not tuple
        or len(qualifications) != len(FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets)
        or any(type(item) is not TrainQualificationAntigravity for item in qualifications)
        or tuple(item.dataset for item in qualifications)
        != FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets
    ):
        raise ValueError("qualification must contain the exact ordered five ECG records")
    authority = capture_authority()
    if (
        type(authority) is not Exp456ResultAuthorityAntigravity
        or authority.label_query_count != 0
    ):
        raise ValueError("Exp456 authority must be exact and label-query-free")
    by_dataset = {feature.dataset: feature for feature in authority.features}
    if tuple(dataset for dataset in FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets if dataset in by_dataset) != FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets:
        raise ValueError("Exp456 authority lacks an ECG routing feature")
    features = tuple(by_dataset[item.dataset] for item in qualifications)
    decisions = tuple(
        route_ecg(feature, qualification)
        for feature, qualification in zip(features, qualifications, strict=True)
    )
    result = evaluate_routes(authority, decisions)
    evidence = PublicationEvidenceAntigravity(
        authority, qualifications, features, result
    )
    return result, publish_result(root, evidence)


def _evaluate_antigravity(reviewed_commit: str) -> int:
    from .artifacts_antigravity import publish_result_antigravity
    from .dashboard_export_antigravity import export_dashboard_antigravity
    from .evaluation_antigravity import (
        capture_exp456_result_authority_antigravity,
        evaluate_routes_antigravity,
    )
    from .qualification_antigravity import qualify_all_ecg_antigravity
    from .router_antigravity import route_ecg_antigravity

    result, _published = _evaluate_with_dependencies_antigravity(
        reviewed_commit,
        _validate_clean_current_commit_antigravity,
        qualify_all_ecg_antigravity,
        capture_exp456_result_authority_antigravity,
        route_ecg_antigravity,
        evaluate_routes_antigravity,
        publish_result_antigravity,
        _OFFICIAL_ROOT_ANTIGRAVITY,
    )
    emit_dashboard_json_antigravity(result, export_dashboard_antigravity)
    return 0


def _inspect_with_dependencies_antigravity(
    root: Path,
    inspect_result: Callable[[Path], object],
    export_dashboard: Callable[[object], dict[str, object]],
) -> int:
    """Replay the supplied result and emit its aggregate/TRAIN-gate view."""
    emit_dashboard_json_antigravity(inspect_result(root), export_dashboard)
    return 0


def _inspect_antigravity() -> int:
    from .dashboard_export_antigravity import export_dashboard_antigravity
    from .replay_antigravity import inspect_result_at_antigravity

    return _inspect_with_dependencies_antigravity(
        _OFFICIAL_ROOT_ANTIGRAVITY,
        inspect_result_at_antigravity,
        export_dashboard_antigravity,
    )


def _load_evaluate_antigravity():
    return _evaluate_antigravity


def _load_inspect_antigravity():
    return _inspect_antigravity


def main_antigravity(argv: Sequence[str] | None = None):
    """Parse first and load only the selected execution phase."""
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    if args.command == "evaluate":
        return _load_evaluate_antigravity()(args.reviewed_commit)
    return _load_inspect_antigravity()()


__all__ = ("main_antigravity", "parser_antigravity")
