"""Closed, lazy command boundary for Exp464 Task 5."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


if TYPE_CHECKING:
    from .artifacts_antigravity import PublishedQualificationAntigravity
    from .contracts_antigravity import Exp463AuthorityAntigravity
    from .evaluation_antigravity import QualificationDecisionAntigravity
    from .qualification_antigravity import Exp464QualificationContractAntigravity
    from .replay_antigravity import QualificationReplayAntigravity


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT_ANTIGRAVITY = Path(FROZEN_CONFIG_ANTIGRAVITY.official_result_root)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp464_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp464_cli_private_test_antigravity_v1"
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
        prog="run_exp464_ecg_residual_centered_ramp_confirmation_antigravity",
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
    if (
        not isinstance(root, Path)
        or root.resolve(strict=False) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False)
    ):
        raise ValueError("private test root must be a non-official Path")
    return root


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "validate_clean_current_commit_antigravity"):
        private.validate_clean_current_commit_antigravity(reviewed_commit)
    else:
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
        raise ValueError("official result root must be absent before Task 6")
    if private is not None and hasattr(private, "validate_no_duplicate_process_antigravity"):
        private.validate_no_duplicate_process_antigravity()


def _json_leaf_antigravity(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_leaf_antigravity(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _result_payload_from_root_antigravity(
    root: Path,
    replayed: QualificationReplayAntigravity,
) -> dict[str, object]:
    return {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "kind": FROZEN_CONFIG_ANTIGRAVITY.kind,
        "context": _json_leaf_antigravity(root / "context_antigravity.json"),
        "method_fold_metrics": _jsonl_leaf_antigravity(root / "method_fold_metrics_antigravity.jsonl"),
        "failed_folds": _jsonl_leaf_antigravity(root / "failed_folds_antigravity.jsonl"),
        "method_summaries": _jsonl_leaf_antigravity(root / "method_summary_antigravity.jsonl"),
        "decision": _json_leaf_antigravity(root / "decision_antigravity.json"),
        "dashboard_snapshot": _json_leaf_antigravity(root / "dashboard_snapshot_antigravity.json"),
        "leaf_sha256": dict(replayed.leaf_sha256),
        "artifact_map_sha256": replayed.artifact_map_sha256,
        "artifact_sha256": _json_leaf_antigravity(root / "artifact_sha256_antigravity.json"),
    }


def _inspect_result_payload_antigravity(
    root: Path,
    inspector: Callable[[Path], object],
) -> dict[str, object]:
    replayed = inspector(root)
    if isinstance(replayed, dict):
        decision = replayed.get("decision")
        if (
            not isinstance(decision, dict)
            or decision.get("label_query_count")
            != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain
        ):
            raise ValueError("inspection opened labels")
        return replayed
    from .replay_antigravity import QualificationReplayAntigravity

    if type(replayed) is not QualificationReplayAntigravity:
        raise ValueError("inspector must return an Exp464 replay payload")
    if replayed.decision.label_query_count != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
        raise ValueError("inspection opened labels")
    return _result_payload_from_root_antigravity(root, replayed)


def _evaluate_with_dependencies_antigravity(
    root: Path,
    authority_loader: Callable[[], Exp463AuthorityAntigravity],
    qualifier: Callable[[], Exp464QualificationContractAntigravity],
    evaluator: Callable[
        [Exp464QualificationContractAntigravity, Exp463AuthorityAntigravity],
        QualificationDecisionAntigravity,
    ],
    publisher: Callable[
        [
            Path,
            Exp464QualificationContractAntigravity,
            QualificationDecisionAntigravity,
            Exp463AuthorityAntigravity,
        ],
        PublishedQualificationAntigravity,
    ],
    inspector: Callable[[Path], object],
) -> dict[str, object]:
    """Evaluate, publish, replay, and return one canonical result JSON."""

    from .artifacts_antigravity import PublishedQualificationAntigravity
    from .contracts_antigravity import (
        Exp463AuthorityAntigravity,
        validate_exp463_authority_antigravity,
    )
    from .evaluation_antigravity import QualificationDecisionAntigravity
    from .qualification_antigravity import Exp464QualificationContractAntigravity

    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("private result root must be a non-symlink Path")
    authority = authority_loader()
    validate_exp463_authority_antigravity(authority)
    if type(authority) is not Exp463AuthorityAntigravity:
        raise ValueError("authority loader must return the exact Exp463 authority")
    qualification = qualifier()
    if type(qualification) is not Exp464QualificationContractAntigravity:
        raise ValueError("qualifier must return the exact Exp464 qualification")
    decision = evaluator(qualification, authority)
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("evaluator must return the exact Exp464 decision")
    publication = publisher(root, qualification, decision, authority)
    if (
        type(publication) is not PublishedQualificationAntigravity
        or publication.root != root
        or len(publication.leaf_sha256) != len(_LEAVES_ANTIGRAVITY)
    ):
        raise ValueError("publisher must create the exact seven-leaf result")
    payload = _inspect_result_payload_antigravity(root, inspector)
    if payload["decision"] != {
        "status": decision.status,
        "primary_candidate": decision.primary_candidate,
        "eligible_for_routing_research": decision.eligible_for_routing_research,
        "primary_passed_count": decision.primary_passed_count,
        "primary_required_count": decision.primary_required_count,
        "failure_reasons": list(decision.failure_reasons),
        "routing_applied": decision.routing_applied,
        "test_labels_opened": decision.test_labels_opened,
        "label_query_count": decision.label_query_count,
    }:
        raise ValueError("published result does not replay to the evaluated payload")
    return payload


def _inspect_with_dependencies_antigravity(
    root: Path,
    inspector: Callable[[Path], object],
) -> dict[str, object]:
    return _inspect_result_payload_antigravity(root, inspector)


def _evaluate_antigravity(_reviewed_commit: str) -> dict[str, object]:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "evaluate_dependencies_antigravity"):
        authority_loader, qualifier, evaluator, publisher = private.evaluate_dependencies_antigravity()
    else:
        from .artifacts_antigravity import publish_result_antigravity
        from .evaluation_antigravity import evaluate_qualification_antigravity
        from .predecessor_antigravity import capture_exp463_authority_antigravity
        from .qualification_antigravity import qualify_methods_antigravity

        authority_loader = capture_exp463_authority_antigravity
        qualifier = qualify_methods_antigravity
        evaluator = evaluate_qualification_antigravity
        publisher = publish_result_antigravity
    if private is not None and hasattr(private, "inspect_antigravity"):
        inspector = private.inspect_antigravity
    else:
        from .replay_antigravity import inspect_replay_at_antigravity

        inspector = inspect_replay_at_antigravity
    return _evaluate_with_dependencies_antigravity(
        _selected_root_antigravity(OFFICIAL_ROOT_ANTIGRAVITY),
        authority_loader,
        qualifier,
        evaluator,
        publisher,
        inspector,
    )


def _inspect_antigravity() -> dict[str, object]:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "inspect_antigravity"):
        inspector = private.inspect_antigravity
    else:
        from .replay_antigravity import inspect_replay_at_antigravity

        inspector = inspect_replay_at_antigravity
    return _inspect_with_dependencies_antigravity(
        _selected_root_antigravity(OFFICIAL_ROOT_ANTIGRAVITY),
        inspector,
    )


def _load_evaluate_antigravity():
    return _evaluate_antigravity


def _load_inspect_antigravity():
    return _inspect_antigravity


def _emit_payload_antigravity(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    if args.command == "evaluate":
        _validate_clean_current_commit_antigravity(args.reviewed_commit)
        payload = _load_evaluate_antigravity()(args.reviewed_commit)
    else:
        payload = _load_inspect_antigravity()()
    _emit_payload_antigravity(payload)
    return 0


__all__ = (
    "OFFICIAL_ROOT_ANTIGRAVITY",
    "_evaluate_with_dependencies_antigravity",
    "_inspect_with_dependencies_antigravity",
    "main_antigravity",
    "parser_antigravity",
)
