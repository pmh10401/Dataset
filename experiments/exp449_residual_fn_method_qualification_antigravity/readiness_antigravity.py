"""Closed, receipt-gated readiness for the Exp449 two-root lifecycle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Callable


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_MAP_TRIPLET_ANTIGRAVITY = frozenset((
    "docs/codemap/codemap_exp449_antigravity.html",
    "docs/codemap/codemap_exp449_antigravity.json",
    "docs/codemap/codemap_exp449_antigravity.lock",
))
_PREDICTION_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812")
_EVALUATION_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_evaluation_antigravity_20260812")
_RECEIPT_DIR_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_lifecycle_receipts_antigravity_20260812")
_AUDIT_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_failure_audit_antigravity.json")


def _git_antigravity(*arguments: str, repository: Path | None = None) -> str:
    """Return a strict Git result from the reviewed repository."""
    try:
        return subprocess.check_output(("git", "-C", str(repository or _REPOSITORY_ANTIGRAVITY), *arguments), text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("repository Git state cannot be verified") from error


def _strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("reviewed map JSON has duplicate keys")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("reviewed map is unavailable") from error


def fingerprint_tracked_paths_antigravity(repository: Path, paths: list[str]) -> dict[str, str]:
    """Hash every exact non-triplet tracked path without a directory-wide shortcut."""
    fingerprints: dict[str, str] = {}
    for relative in paths:
        if type(relative) is not str or relative in _MAP_TRIPLET_ANTIGRAVITY:
            continue
        candidate = repository / relative
        try:
            fingerprints[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError("reviewed tracked path is unavailable") from error
    return fingerprints


def validate_reviewed_map_antigravity(reviewed_commit: str, *, repository: Path = _REPOSITORY_ANTIGRAVITY) -> None:
    """Require one clean committed map that represents and fingerprints this HEAD."""
    if type(reviewed_commit) is not str or _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed_commit must be a lowercase 40-hex commit")
    if _git_antigravity("rev-parse", "HEAD", repository=repository) != reviewed_commit:
        raise ValueError("reviewed_commit does not equal current HEAD")
    if _git_antigravity("status", "--porcelain=v1", repository=repository):
        raise ValueError("working tree must be clean")
    parent = _git_antigravity("rev-parse", "HEAD^", repository=repository)
    map_dir = repository / "docs/codemap"
    graph = _strict_json_antigravity(map_dir / "codemap_exp449_antigravity.json")
    lock = _strict_json_antigravity(map_dir / "codemap_exp449_antigravity.lock")
    html = (map_dir / "codemap_exp449_antigravity.html").read_text(encoding="utf-8")
    if type(graph) is not dict or type(lock) is not dict or "<!doctype html" not in html.lower():
        raise ValueError("reviewed map triplet is invalid")
    if graph.get("source_parent_commit") != parent or lock.get("schema") != "exp449_codemap_same_commit_v1" or lock.get("source_parent_commit") != parent:
        raise ValueError("reviewed map source parent does not equal HEAD^")
    expected_paths = sorted(item for item in _git_antigravity("diff", "--name-only", "HEAD^", repository=repository).splitlines() if item)
    represented = lock.get("represented_paths")
    if type(represented) is not list or any(type(item) is not str for item in represented) or sorted(represented) != expected_paths:
        raise ValueError("reviewed map represented paths differ")
    if set(_MAP_TRIPLET_ANTIGRAVITY) - set(represented) or lock.get("excluded_paths") != sorted(_MAP_TRIPLET_ANTIGRAVITY):
        raise ValueError("reviewed map triplet scope differs")
    tracked = sorted(item for item in _git_antigravity("ls-files", "-z", repository=repository).split("\0") if item)
    fingerprints = lock.get("tracked_fingerprints")
    expected_fingerprints = fingerprint_tracked_paths_antigravity(repository, tracked)
    if fingerprints != expected_fingerprints:
        raise ValueError("reviewed map fingerprint drift")


def run_readiness_with_dependencies_antigravity(
    reviewed_commit: str,
    reviewed_map: Callable[[str], None],
    *,
    receipt_dir: Path = _RECEIPT_DIR_ANTIGRAVITY,
    audit_path: Path = _AUDIT_PATH_ANTIGRAVITY,
    prediction_root: Path = _PREDICTION_ROOT_ANTIGRAVITY,
    evaluation_root: Path = _EVALUATION_ROOT_ANTIGRAVITY,
) -> dict[str, object]:
    """Run first and final review checks inside the existing receipt/audit ordering."""
    if not callable(reviewed_map):
        raise ValueError("reviewed map validator is invalid")
    from .lifecycle_antigravity import run_phase_antigravity

    def final_review() -> dict[str, object]:
        reviewed_map(reviewed_commit)
        return {"experiment_id": "Exp449", "reviewed_commit": reviewed_commit, "label_access_count": 0,
                "prediction_root_absent": True, "evaluation_root_absent": True}

    return run_phase_antigravity(
        "readiness", reviewed_commit, final_review, receipt_dir=receipt_dir, audit_path=audit_path,
        prediction_root=prediction_root, evaluation_root=evaluation_root,
        path_precondition=lambda: reviewed_map(reviewed_commit),
    )


def readiness_antigravity(reviewed_commit: str) -> dict[str, object]:
    """Official no-label readiness entrypoint; it cannot create either publication root."""
    return run_readiness_with_dependencies_antigravity(reviewed_commit, validate_reviewed_map_antigravity)


__all__ = ["fingerprint_tracked_paths_antigravity", "readiness_antigravity", "run_readiness_with_dependencies_antigravity", "validate_reviewed_map_antigravity"]
