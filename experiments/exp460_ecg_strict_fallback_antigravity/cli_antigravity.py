"""Closed lazy command boundary for Exp460 Task 5."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .artifacts_antigravity import FrozenFallbackPublicationAntigravity
    from .contracts_antigravity import (
        Exp459FallbackAuthorityAntigravity,
        FallbackEvaluationAntigravity,
    )


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp460_ecg_strict_fallback_antigravity_20260820"
)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp460_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp460_cli_private_test_antigravity_v1"


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
        self.error(
            "only bare inspect or evaluate --reviewed-commit LOWER40 is accepted"
        )


def _reviewed_commit_antigravity(value: str) -> str:
    if _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "reviewed commit must be lowercase 40-hex"
        )
    return value


def parser_antigravity() -> argparse.ArgumentParser:
    """Expose exactly one reviewed evaluation form and one bare replay form."""
    parser = _ExactSurfaceParserAntigravity(
        prog="run_exp460_ecg_strict_fallback_antigravity",
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
    if not isinstance(root, Path) or root.resolve(
        strict=False
    ) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False):
        raise ValueError("private test root must be a non-official Path")
    return root


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    """Bind a future official write to the exact clean reviewed commit."""
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(
        private, "validate_clean_current_commit_antigravity"
    ):
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
        raise ValueError(
            "reviewed commit must be the exact clean current commit"
        )


def _evaluate_with_dependencies_antigravity(
    root: Path,
    authority_loader: Callable[[], Exp459FallbackAuthorityAntigravity],
    evaluator: Callable[
        [Exp459FallbackAuthorityAntigravity], FallbackEvaluationAntigravity
    ],
    publisher: Callable[..., FrozenFallbackPublicationAntigravity],
) -> dict[str, object]:
    """Capture authority, evaluate, publish six leaves, replay, and export."""
    from .artifacts_antigravity import FrozenFallbackPublicationAntigravity
    from .contracts_antigravity import FallbackEvaluationAntigravity
    from .dashboard_export_antigravity import export_dashboard_antigravity
    from .replay_antigravity import inspect_replay_at_antigravity

    authority = authority_loader()
    result = evaluator(authority)
    if type(result) is not FallbackEvaluationAntigravity:
        raise ValueError("evaluator must return the exact Exp460 evaluation")
    publication = publisher(root, authority, result)
    if (
        type(publication) is not FrozenFallbackPublicationAntigravity
        or publication.root != root
        or len(publication.leaf_sha256) != 6
    ):
        raise ValueError("publisher must create the exact six-leaf result")
    replayed = inspect_replay_at_antigravity(root, authority)
    if type(replayed) is not FallbackEvaluationAntigravity or replayed != result:
        raise ValueError("published result does not replay to the evaluated payload")
    return export_dashboard_antigravity(authority, replayed)


def _inspect_with_dependencies_antigravity(
    root: Path,
    authority_loader: Callable[[], Exp459FallbackAuthorityAntigravity],
    inspector: Callable[
        [Path, Exp459FallbackAuthorityAntigravity],
        FallbackEvaluationAntigravity,
    ],
) -> dict[str, object]:
    """Replay an existing six-leaf root against authority and export it."""
    from .contracts_antigravity import FallbackEvaluationAntigravity
    from .dashboard_export_antigravity import export_dashboard_antigravity

    authority = authority_loader()
    replayed = inspector(root, authority)
    if type(replayed) is not FallbackEvaluationAntigravity:
        raise ValueError("inspector must return the exact Exp460 evaluation")
    return export_dashboard_antigravity(authority, replayed)


def _evaluate_antigravity(
    _reviewed_commit: str,
) -> dict[str, object]:
    from .artifacts_antigravity import publish_result_antigravity
    from .evaluation_antigravity import compare_fallback_rules_antigravity
    from .predecessor_antigravity import (
        capture_exp459_fallback_authority_antigravity,
    )

    return _evaluate_with_dependencies_antigravity(
        _selected_root_antigravity(OFFICIAL_ROOT_ANTIGRAVITY),
        capture_exp459_fallback_authority_antigravity,
        compare_fallback_rules_antigravity,
        publish_result_antigravity,
    )


def _inspect_antigravity() -> dict[str, object]:
    from .predecessor_antigravity import (
        capture_exp459_fallback_authority_antigravity,
    )
    from .replay_antigravity import inspect_replay_at_antigravity

    return _inspect_with_dependencies_antigravity(
        _selected_root_antigravity(OFFICIAL_ROOT_ANTIGRAVITY),
        capture_exp459_fallback_authority_antigravity,
        inspect_replay_at_antigravity,
    )


def _load_evaluate_antigravity():
    return _evaluate_antigravity


def _load_inspect_antigravity():
    return _inspect_antigravity


def _emit_dashboard_antigravity(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    """Parse first, then load and execute only the selected phase."""
    args = parser_antigravity().parse_args(
        None if argv is None else list(argv)
    )
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
