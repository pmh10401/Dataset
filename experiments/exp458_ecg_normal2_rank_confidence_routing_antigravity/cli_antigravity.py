"""Closed lazy command boundary for the reviewed Exp458 candidate."""

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
    "/Users/mac/Documents/Dataset/exp458_ecg_normal2_rank_confidence_routing_antigravity_20260818"
)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp458_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp458_cli_private_test_antigravity_v1"


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
        prog="run_exp458_ecg_normal2_rank_confidence_routing_antigravity",
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
    """Bind the public official write to the exact clean reviewed commit."""
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "validate_clean_current_commit_antigravity"):
        private.validate_clean_current_commit_antigravity(reviewed_commit)
        return
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=_REPOSITORY_ANTIGRAVITY, text=True
    ).strip()
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=_REPOSITORY_ANTIGRAVITY, text=True
    )
    if reviewed_commit != head or dirty:
        raise ValueError("reviewed commit must be the exact clean current commit")


def _private_test_seam_antigravity() -> object | None:
    private = sys.modules.get(_PRIVATE_TEST_MODULE_ANTIGRAVITY)
    if (
        private is not None
        and getattr(private, "CAPABILITY_ANTIGRAVITY", None)
        == _PRIVATE_TEST_CAPABILITY_ANTIGRAVITY
    ):
        return private
    return None


def _selected_root_antigravity(default: Path) -> Path:
    private = _private_test_seam_antigravity()
    if private is None:
        return default
    root = getattr(private, "root_antigravity", None)
    if not isinstance(root, Path) or root == _OFFICIAL_ROOT_ANTIGRAVITY:
        raise ValueError("private test root must be a non-official Path")
    return root


def emit_dashboard_json_antigravity(result: object) -> None:
    """Emit only the validated aggregate/TRAIN-gate dashboard payload."""
    from .dashboard_export_antigravity import export_dashboard_antigravity

    print(
        json.dumps(
            export_dashboard_antigravity(result), sort_keys=True, separators=(",", ":")
        )
    )


def observe_runtime_provenance_antigravity() -> object:
    """Validate exact interpreter/package provenance before any qualification writes."""
    from .contracts_antigravity import observe_runtime_provenance_antigravity as observe

    return observe()


def _evaluate_with_dependencies_antigravity(
    reviewed_commit: str,
    root: Path,
    qualifier: Callable[[], object],
    authority_capture: Callable[[], object],
    router: Callable[[object, object], object],
    evaluator: Callable[[object, object], object],
    publisher: Callable[[Path, object], object],
    inspector: Callable[[Path], object],
) -> int:
    """Run Gate 1, route, aggregate, publish, replay, and emit the dashboard."""
    from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
        DatasetRoutingFeaturesAntigravity,
    )

    from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
    from .contracts_antigravity import (
        EvaluationResultAntigravity,
        Exp456ResultAuthorityAntigravity,
        PublicationEvidenceAntigravity,
        PublishedResultAntigravity,
        RankQualificationAntigravity,
        RouteDecisionAntigravity,
    )

    if _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed commit must be lowercase 40-hex")
    observe_runtime_provenance_antigravity()
    qualification = qualifier()
    if type(qualification) is not RankQualificationAntigravity:
        raise ValueError("qualification must be the exact Exp458 rank record")
    authority = authority_capture()
    if type(authority) is not Exp456ResultAuthorityAntigravity or authority.label_query_count != 0:
        raise ValueError("Exp456 authority must be exact and label-query-free")
    by_dataset = {feature.dataset: feature for feature in authority.features}
    feature = by_dataset.get(FROZEN_CONFIG_ANTIGRAVITY.dataset)
    if type(feature) is not DatasetRoutingFeaturesAntigravity:
        raise ValueError("Exp456 authority lacks the normal_2 feature row")
    decision = router(feature, qualification)
    if type(decision) is not RouteDecisionAntigravity or decision.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
        raise ValueError("router must return the exact normal_2 decision")
    result = evaluator(authority, decision)
    if type(result) is not EvaluationResultAntigravity or tuple(result.decisions) != (decision,):
        raise ValueError("evaluator must return one normal_2 decision result")
    published = publisher(
        root,
        PublicationEvidenceAntigravity(authority, qualification, feature, result),
    )
    if type(published) is not PublishedResultAntigravity or published.root != root or len(published.leaf_sha256) != 9:
        raise ValueError("publisher must create the exact nine-leaf result")
    replayed = inspector(root)
    if type(replayed) is not EvaluationResultAntigravity or replayed != result:
        raise ValueError("published result does not replay to the evaluated payload")
    emit_dashboard_json_antigravity(replayed)
    return 0


def _evaluate_antigravity(reviewed_commit: str) -> int:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "evaluate_dependencies_antigravity"):
        dependencies = private.evaluate_dependencies_antigravity()
        if type(dependencies) is not tuple or len(dependencies) != 6:
            raise ValueError("private test dependencies are invalid")
        return _evaluate_with_dependencies_antigravity(
            reviewed_commit,
            _selected_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY),
            dependencies[0],
            dependencies[1],
            dependencies[2],
            dependencies[3],
            dependencies[4],
            dependencies[5],
        )

    from .artifacts_antigravity import publish_result_antigravity
    from .evaluation_antigravity import (
        capture_exp456_result_authority_antigravity,
        evaluate_routes_antigravity,
    )
    from .qualification_antigravity import qualify_normal2_antigravity
    from .replay_antigravity import inspect_result_at_antigravity
    from .router_antigravity import route_normal2_antigravity

    return _evaluate_with_dependencies_antigravity(
        reviewed_commit,
        _selected_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY),
        qualify_normal2_antigravity,
        capture_exp456_result_authority_antigravity,
        route_normal2_antigravity,
        evaluate_routes_antigravity,
        publish_result_antigravity,
        inspect_result_at_antigravity,
    )


def _inspect_with_dependencies_antigravity(
    root: Path, inspector: Callable[[Path], object]
) -> int:
    """Replay the supplied result and emit its aggregate/TRAIN-gate view."""
    emit_dashboard_json_antigravity(inspector(root))
    return 0


def _inspect_antigravity() -> int:
    from .replay_antigravity import inspect_result_at_antigravity

    return _inspect_with_dependencies_antigravity(
        _selected_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY),
        inspect_result_at_antigravity,
    )


def _load_evaluate_antigravity():
    return _evaluate_antigravity


def _load_inspect_antigravity():
    return _inspect_antigravity


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    """Parse first, validate public reviewed commit, then load only the selected phase."""
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    if args.command == "evaluate":
        _validate_clean_current_commit_antigravity(args.reviewed_commit)
        return _load_evaluate_antigravity()(args.reviewed_commit)
    return _load_inspect_antigravity()()


__all__ = (
    "_evaluate_with_dependencies_antigravity",
    "_inspect_with_dependencies_antigravity",
    "main_antigravity",
    "observe_runtime_provenance_antigravity",
    "parser_antigravity",
)
