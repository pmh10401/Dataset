"""No-label, no-mutation readiness checks for Exp447's future evaluator."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .sources_antigravity import _verify_database_antigravity, capture_exp446_publication_antigravity, capture_hard_predictions_antigravity


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_DESIGN_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / "docs/superpowers/specs/2026-08-11-exp447-exp446-postfreeze-performance-evaluation-antigravity-design.md"
_PREREGISTRATION_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / "docs/EXP447_PREREGISTRATION_antigravity_KO.md"
_EXPECTED_DESIGN_SHA256_ANTIGRAVITY = "043009941619335b658af99911b1463f66ea73e93d600fe1cfb884391afea868"
_EXPECTED_PREREGISTRATION_SHA256_ANTIGRAVITY = "a3400c9222c5f3d8e5f5a15a8e38222d62471e0d5d2f2a73a43072df9cb0e6bd"


def _git_antigravity(*arguments: str) -> str:
    try:
        return subprocess.check_output(("git", "-C", str(_REPOSITORY_ANTIGRAVITY), *arguments), text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("repository Git state cannot be verified") from error


def _require_clean_reviewed_head_antigravity(reviewed_commit: str) -> None:
    if type(reviewed_commit) is not str or _COMMIT_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise ValueError("reviewed_commit must be a lowercase 40-hex commit")
    if _git_antigravity("rev-parse", "HEAD") != reviewed_commit:
        raise ValueError("reviewed_commit does not equal current HEAD")
    if _git_antigravity("status", "--porcelain=v1"):
        raise ValueError("working tree must be clean")


def _require_document_hashes_antigravity() -> None:
    for path, expected in ((_DESIGN_ANTIGRAVITY, _EXPECTED_DESIGN_SHA256_ANTIGRAVITY), (_PREREGISTRATION_ANTIGRAVITY, _EXPECTED_PREREGISTRATION_SHA256_ANTIGRAVITY)):
        try:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError("frozen Exp447 document is unavailable") from error
        if observed != expected:
            raise ValueError("frozen Exp447 document digest changed")


def _require_absent_run_root_antigravity() -> None:
    root = Path(FROZEN_CONFIG_ANTIGRAVITY.run_root)
    try:
        root.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("Exp447 root absence cannot be verified") from error
    raise ValueError("Exp447 root already exists or is unsafe")


def readiness_antigravity(reviewed_commit: str) -> dict[str, object]:
    """Validate source authority and operator state without opening TEST labels."""
    _require_clean_reviewed_head_antigravity(reviewed_commit)
    _require_document_hashes_antigravity()
    _require_absent_run_root_antigravity()
    database_descriptor, _database_identity = _verify_database_antigravity()
    os.close(database_descriptor)
    identity = capture_exp446_publication_antigravity()
    hard = capture_hard_predictions_antigravity(identity)
    if sys.version_info[:2] < (3, 14):
        raise ValueError("Exp447 requires the frozen Python 3.14 environment")
    return {
        "experiment_id": FROZEN_CONFIG_ANTIGRAVITY.experiment_id,
        "reviewed_commit": reviewed_commit,
        "dataset_count": len(identity.dataset_order),
        "hard_coordinate_count": len(hard.coordinates),
        "label_access_count": 0,
        "run_root_absent": True,
    }


__all__ = ["readiness_antigravity"]
