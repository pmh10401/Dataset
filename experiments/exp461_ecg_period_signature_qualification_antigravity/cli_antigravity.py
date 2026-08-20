"""Closed lazy command boundary for Exp461 Task 5."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


if TYPE_CHECKING:
    from .artifacts_antigravity import PublishedQualificationAntigravity
    from .contracts_antigravity import Exp460PeriodAuthorityAntigravity
    from .evaluation_antigravity import QualificationDecisionAntigravity
    from .qualification_antigravity import MethodQualificationAntigravity
    from .replay_antigravity import QualificationReplayAntigravity


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT_ANTIGRAVITY = Path(FROZEN_CONFIG_ANTIGRAVITY.official_result_root)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp461_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp461_cli_private_test_antigravity_v1"


class _ExactSurfaceParserAntigravity(argparse.ArgumentParser):
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
        prog="run_exp461_ecg_period_signature_qualification_antigravity",
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
    if not isinstance(root, Path) or root.resolve(strict=False) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False):
        raise ValueError("private test root must be a non-official Path")
    return root


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "validate_clean_current_commit_antigravity"):
        private.validate_clean_current_commit_antigravity(reviewed_commit)
        return
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


def _evaluate_with_dependencies_antigravity(
    root: Path,
    authority_loader: Callable[[], Exp460PeriodAuthorityAntigravity],
    qualifier: Callable[[], MethodQualificationAntigravity],
    evaluator: Callable[
        [Exp460PeriodAuthorityAntigravity, MethodQualificationAntigravity],
        QualificationDecisionAntigravity,
    ],
    publisher: Callable[
        [Path, Exp460PeriodAuthorityAntigravity, MethodQualificationAntigravity, QualificationDecisionAntigravity],
        PublishedQualificationAntigravity,
    ],
) -> dict[str, object]:
    from .artifacts_antigravity import PublishedQualificationAntigravity
    from .contracts_antigravity import (
        Exp460PeriodAuthorityAntigravity,
        validate_exp460_period_authority_antigravity,
    )
    from .dashboard_export_antigravity import export_dashboard_antigravity
    from .evaluation_antigravity import QualificationDecisionAntigravity
    from .qualification_antigravity import MethodQualificationAntigravity
    from .replay_antigravity import QualificationReplayAntigravity, inspect_replay_at_antigravity

    authority = validate_exp460_period_authority_antigravity(authority_loader())
    if type(authority) is not Exp460PeriodAuthorityAntigravity:
        raise ValueError("authority loader must return the exact Exp461 authority")
    qualification = qualifier()
    if type(qualification) is not MethodQualificationAntigravity:
        raise ValueError("qualifier must return the exact Exp461 qualification")
    decision = evaluator(authority, qualification)
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("evaluator must return the exact Exp461 decision")
    publication = publisher(root, authority, qualification, decision)
    if (
        type(publication) is not PublishedQualificationAntigravity
        or publication.root != root
        or len(publication.leaf_sha256) != 7
    ):
        raise ValueError("publisher must create the exact seven-leaf result")
    replayed = inspect_replay_at_antigravity(root)
    if type(replayed) is not QualificationReplayAntigravity or replayed.decision != decision:
        raise ValueError("published result does not replay to the evaluated payload")
    return export_dashboard_antigravity(authority, replayed.qualification, replayed.decision)


def _inspect_with_dependencies_antigravity(
    root: Path,
    inspector: Callable[[Path], object],
) -> dict[str, object]:
    replayed = inspector(root)
    if isinstance(replayed, dict):
        return replayed
    dashboard = getattr(replayed, "dashboard", None)
    if not isinstance(dashboard, dict):
        raise ValueError("inspector must return an Exp461 replay payload")
    return dashboard


def _evaluate_antigravity(_reviewed_commit: str) -> dict[str, object]:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "evaluate_dependencies_antigravity"):
        authority_loader, qualifier, evaluator, publisher = private.evaluate_dependencies_antigravity()
    else:
        from .artifacts_antigravity import publish_result_antigravity
        from .evaluation_antigravity import evaluate_qualification_antigravity
        from .predecessor_antigravity import capture_exp460_period_authority_antigravity
        from .qualification_antigravity import qualify_methods_antigravity

        authority_loader = capture_exp460_period_authority_antigravity
        qualifier = qualify_methods_antigravity
        evaluator = evaluate_qualification_antigravity
        publisher = publish_result_antigravity
    return _evaluate_with_dependencies_antigravity(
        _selected_root_antigravity(OFFICIAL_ROOT_ANTIGRAVITY),
        authority_loader,
        qualifier,
        evaluator,
        publisher,
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


def _emit_dashboard_antigravity(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    if args.command == "evaluate":
        _validate_clean_current_commit_antigravity(args.reviewed_commit)
        payload = _load_evaluate_antigravity()(args.reviewed_commit)
    else:
        payload = _load_inspect_antigravity()()
    _emit_dashboard_antigravity(payload)
    return 0


__all__ = (
    "OFFICIAL_ROOT_ANTIGRAVITY",
    "_evaluate_with_dependencies_antigravity",
    "_inspect_with_dependencies_antigravity",
    "main_antigravity",
    "parser_antigravity",
)
