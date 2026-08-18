"""Closed lazy command boundary for Exp459 Task 5."""

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
    "/Users/mac/Documents/Dataset/exp459_ecg_normal2_blocked_validation_antigravity_20260818"
)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp459_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp459_cli_private_test_antigravity_v1"


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
        prog="run_exp459_ecg_normal2_blocked_validation_antigravity",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate.add_argument(
        "--reviewed-commit", required=True, type=_reviewed_commit_antigravity
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


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    """Bind a future official write to the exact clean reviewed commit."""
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


def _selected_root_antigravity(default: Path) -> Path:
    private = _private_test_seam_antigravity()
    if private is None:
        return default
    root = getattr(private, "root_antigravity", None)
    if not isinstance(root, Path) or root == _OFFICIAL_ROOT_ANTIGRAVITY:
        raise ValueError("private test root must be a non-official Path")
    return root


def observe_runtime_provenance_antigravity() -> dict[str, str]:
    """Validate exact interpreter/package provenance before qualification."""
    import platform

    import aeon
    import numpy as np

    from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY

    actual = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "aeon_version": aeon.__version__,
    }
    expected = {
        "executable": FROZEN_CONFIG_ANTIGRAVITY.runtime_executable,
        "prefix": FROZEN_CONFIG_ANTIGRAVITY.runtime_prefix,
        "python_version": FROZEN_CONFIG_ANTIGRAVITY.runtime_python_version,
        "numpy_version": FROZEN_CONFIG_ANTIGRAVITY.runtime_numpy_version,
        "aeon_version": FROZEN_CONFIG_ANTIGRAVITY.runtime_aeon_version,
    }
    for field, wanted in expected.items():
        if actual[field] != wanted:
            raise ValueError(f"{field} differs from frozen runtime")
    return actual


def _load_frozen_validation_authority_antigravity() -> object:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "authority_antigravity"):
        return private.authority_antigravity()
    try:
        from .frozen_validation_authority_antigravity import (
            FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY,
        )
    except ImportError as error:
        raise ValueError("trusted frozen validation authority is required") from error
    return FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY


def emit_dashboard_json_antigravity(authority: object) -> None:
    """Emit only the validated aggregate/TRAIN dashboard payload."""
    from .dashboard_export_antigravity import export_dashboard_from_authority_antigravity

    print(
        json.dumps(
            export_dashboard_from_authority_antigravity(authority),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _same_decision_scalars_antigravity(left: object, right: object) -> bool:
    return (
        getattr(left, "dataset", None) == getattr(right, "dataset", None)
        and getattr(left, "status", None) == getattr(right, "status", None)
        and getattr(left, "passed", None) is getattr(right, "passed", None)
    )


def _evaluate_with_dependencies_antigravity(
    reviewed_commit: str,
    root: Path,
    authority_loader: Callable[[], object],
    qualifier: Callable[[], object],
    evaluator: Callable[[object], object],
    publisher: Callable[[Path, object, object], object],
    inspector: Callable[[Path, object], object],
) -> int:
    """Run blocked qualification, exact authority comparison, publish, replay, and emit."""
    from .artifacts_antigravity import PublishedResultAntigravity
    from .contracts_antigravity import (
        BlockedQualificationAntigravity,
        ValidationDecisionAntigravity,
    )
    from .evaluation_antigravity import (
        validate_decision_matches_authority_antigravity,
        validate_frozen_validation_authority_antigravity,
    )

    if _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed commit must be lowercase 40-hex")
    observe_runtime_provenance_antigravity()
    authority = validate_frozen_validation_authority_antigravity(authority_loader())
    qualification = qualifier()
    if type(qualification) is not BlockedQualificationAntigravity:
        raise ValueError("qualification must be the exact Exp459 blocked qualification")
    decision = evaluator(qualification)
    if type(decision) is not ValidationDecisionAntigravity:
        raise ValueError("evaluator must return the exact Exp459 validation decision")
    trusted = validate_decision_matches_authority_antigravity(decision, authority)
    published = publisher(root, decision, trusted)
    if (
        type(published) is not PublishedResultAntigravity
        or published.root != root
        or len(published.leaf_sha256) != 7
    ):
        raise ValueError("publisher must create the exact seven-leaf result")
    replayed = inspector(root, trusted)
    if (
        type(replayed) is not ValidationDecisionAntigravity
        or not _same_decision_scalars_antigravity(replayed, trusted.decision)
    ):
        raise ValueError("published result does not replay to the authority payload")
    emit_dashboard_json_antigravity(trusted)
    return 0


def _evaluate_dependencies_antigravity() -> tuple[
    Callable[[], object],
    Callable[[object], object],
    Callable[[Path, object, object], object],
    Callable[[Path, object], object],
]:
    private = _private_test_seam_antigravity()
    if private is not None and hasattr(private, "evaluate_dependencies_antigravity"):
        dependencies = private.evaluate_dependencies_antigravity()
        if type(dependencies) is not tuple or len(dependencies) != 4:
            raise ValueError("private test dependencies are invalid")
        return dependencies

    def qualifier_antigravity() -> object:
        from .qualification_antigravity import qualify_blocked_normal2_antigravity

        return qualify_blocked_normal2_antigravity()

    def evaluator_antigravity(qualification: object) -> object:
        from .evaluation_antigravity import evaluate_blocked_qualification_antigravity

        return evaluate_blocked_qualification_antigravity(qualification)

    def publisher_antigravity(root: Path, decision: object, authority: object) -> object:
        from .artifacts_antigravity import publish_result_antigravity

        return publish_result_antigravity(root, decision, authority)

    def inspector_antigravity(root: Path, authority: object) -> object:
        from .replay_antigravity import inspect_replay_at_antigravity

        return inspect_replay_at_antigravity(root, authority)

    return (
        qualifier_antigravity,
        evaluator_antigravity,
        publisher_antigravity,
        inspector_antigravity,
    )


def _evaluate_antigravity(reviewed_commit: str) -> int:
    qualifier, evaluator, publisher, inspector = _evaluate_dependencies_antigravity()
    return _evaluate_with_dependencies_antigravity(
        reviewed_commit,
        _selected_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY),
        _load_frozen_validation_authority_antigravity,
        qualifier,
        evaluator,
        publisher,
        inspector,
    )


def _inspect_with_dependencies_antigravity(
    root: Path,
    authority_loader: Callable[[], object],
    inspector: Callable[[Path, object], object],
) -> int:
    """Replay the supplied result against trusted authority and emit aggregate TRAIN view."""
    from .contracts_antigravity import ValidationDecisionAntigravity
    from .evaluation_antigravity import validate_frozen_validation_authority_antigravity

    authority = validate_frozen_validation_authority_antigravity(authority_loader())
    replayed = inspector(root, authority)
    if (
        type(replayed) is not ValidationDecisionAntigravity
        or not _same_decision_scalars_antigravity(replayed, authority.decision)
    ):
        raise ValueError("published result does not replay to the authority payload")
    emit_dashboard_json_antigravity(authority)
    return 0


def _inspect_antigravity() -> int:
    from .replay_antigravity import inspect_replay_at_antigravity

    return _inspect_with_dependencies_antigravity(
        _selected_root_antigravity(_OFFICIAL_ROOT_ANTIGRAVITY),
        _load_frozen_validation_authority_antigravity,
        inspect_replay_at_antigravity,
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
