"""Closed lazy command boundary for the reviewed Exp456 candidate evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import re
import subprocess


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]


class _OneReviewedCommitActionAntigravity(argparse.Action):
    """Reject repeated review identifiers rather than selecting an arbitrary one."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be specified exactly once")
        setattr(namespace, self.dest, values)


def _reviewed_commit_antigravity(value: str) -> str:
    if _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("reviewed commit must be lowercase 40-hex")
    return value


def parser_antigravity() -> argparse.ArgumentParser:
    """Expose only one reviewed evaluation form and one bare replay form."""
    parser = argparse.ArgumentParser(
        prog="run_exp456_exp455_dataset_key_order_repair_antigravity", allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate.add_argument(
        "--reviewed-commit", required=True, action=_OneReviewedCommitActionAntigravity,
        type=_reviewed_commit_antigravity,
    )
    commands.add_parser("inspect", allow_abbrev=False)
    return parser


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    """The one-shot command may write only for the exact clean reviewed commit."""
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=_ROOT_ANTIGRAVITY, text=True).strip()
    dirty = subprocess.check_output(("git", "status", "--porcelain"), cwd=_ROOT_ANTIGRAVITY, text=True)
    if reviewed_commit != head or dirty:
        raise ValueError("reviewed commit must be the exact clean current commit")


def emit_aggregate_json_antigravity(
    evaluation: object,
    export_aggregate: Callable[[object], dict[str, object]],
) -> None:
    """Emit only the public aggregate dashboard payload."""
    print(json.dumps(export_aggregate(evaluation), sort_keys=True, separators=(",", ":")))


def _evaluate_with_dependencies_antigravity(
    reviewed_commit: str,
    validate_clean_commit: Callable[[str], None],
    capture_authority: Callable[[], object],
    build_features: Callable[[object], tuple[object, ...]],
    leave_one_dataset_out: Callable[[tuple[object, ...], dict[str, tuple[object, object]]], tuple[object, ...]],
    evaluate_routing: Callable[[object, tuple[object, ...], tuple[object, ...]], object],
    publish_evaluation: Callable[[Path, object], object],
    root: Path,
) -> tuple[object, object]:
    """Production-shaped seam: authority once, features, LODO, evaluation, then one publication."""
    validate_clean_commit(reviewed_commit)
    authority = capture_authority()
    if getattr(authority, "label_query_count", None) != 0:
        raise ValueError("authority queried labels")
    features = build_features(authority)
    rows = getattr(authority, "rows", ())
    outcomes = {
        row.dataset: (row.baseline_confusion, row.candidate_confusion)
        for row in rows if row.candidate_confusion is not None
    }
    if len(rows) != 48 or len(features) != 8 or len(outcomes) != 8:
        raise ValueError("frozen Exp456 population differs")
    decisions = leave_one_dataset_out(features, outcomes)
    evaluation = evaluate_routing(authority, features, decisions)
    return evaluation, publish_evaluation(root, evaluation)


def _evaluate_antigravity(reviewed_commit: str) -> int:
    from .artifacts_antigravity import publish_evaluation_antigravity
    from .authority_antigravity import capture_routing_authority_antigravity
    from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
    from .dashboard_export_antigravity import export_aggregate_dashboard_antigravity
    from .evaluation_antigravity import evaluate_routing_antigravity
    from .features_antigravity import build_label_free_features_antigravity
    from .router_antigravity import leave_one_dataset_out_antigravity

    evaluation, _frozen = _evaluate_with_dependencies_antigravity(
        reviewed_commit,
        _validate_clean_current_commit_antigravity,
        capture_routing_authority_antigravity,
        build_label_free_features_antigravity,
        leave_one_dataset_out_antigravity,
        evaluate_routing_antigravity,
        publish_evaluation_antigravity,
        Path(FROZEN_CONFIG_ANTIGRAVITY.run_root),
    )
    emit_aggregate_json_antigravity(evaluation, export_aggregate_dashboard_antigravity)
    return 0


def _inspect_with_dependencies_antigravity(
    root: Path,
    inspect_replay: Callable[[Path], object],
    export_aggregate: Callable[[object], dict[str, object]],
) -> int:
    """Replay at the supplied root and emit its same aggregate-only result."""
    emit_aggregate_json_antigravity(inspect_replay(root), export_aggregate)
    return 0


def _inspect_antigravity() -> int:
    from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
    from .dashboard_export_antigravity import export_aggregate_dashboard_antigravity
    from .replay_antigravity import inspect_replay_at_antigravity

    return _inspect_with_dependencies_antigravity(
        Path(FROZEN_CONFIG_ANTIGRAVITY.run_root),
        inspect_replay_at_antigravity,
        export_aggregate_dashboard_antigravity,
    )


def _load_evaluate_antigravity():
    return _evaluate_antigravity


def _load_inspect_antigravity():
    return _inspect_antigravity


def main_antigravity(argv: Sequence[str] | None = None):
    """Load just the selected phase; parser help cannot import execution modules."""
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    if args.command == "evaluate":
        return _load_evaluate_antigravity()(args.reviewed_commit)
    return _load_inspect_antigravity()()


__all__ = ["main_antigravity", "parser_antigravity"]
