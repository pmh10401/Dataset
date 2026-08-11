"""No-label, no-mutation readiness checks for the Exp448 TEXT boundary."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .sources_antigravity import (
    _verify_database_antigravity,
    capture_exp446_publication_antigravity,
    capture_hard_predictions_antigravity,
    require_absent_exp447_root_antigravity,
    verify_exp447_failure_authority_antigravity,
)


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_DESIGN_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / "docs/superpowers/specs/2026-08-12-exp448-exp447-text-label-boundary-repair-antigravity-design.md"
_PREREGISTRATION_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / "docs/EXP448_PREREGISTRATION_antigravity_KO.md"
_EXPECTED_DESIGN_SHA256_ANTIGRAVITY = "ed2b6e5e8b8ee63ab41a15e54f679147f7e14aeca601f4d62f694ca104ed3ce4"
_EXPECTED_PREREGISTRATION_SHA256_ANTIGRAVITY = "af2b8b10ff25ea4e6e54a4af4757bc5e0432c8d76d17c69612fd10f3933c612f"
_FROZEN_PYTHON_ANTIGRAVITY = (3, 14, 6)
_FROZEN_NUMPY_ANTIGRAVITY = "2.4.6"


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
            raise ValueError("frozen Exp448 document is unavailable") from error
        if observed != expected:
            raise ValueError("frozen Exp448 document digest changed")


def _require_absent_run_root_antigravity() -> None:
    root = Path(FROZEN_CONFIG_ANTIGRAVITY.run_root)
    try:
        root.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("Exp448 root absence cannot be verified") from error
    raise ValueError("Exp448 root already exists or is unsafe")


def _runtime_versions_antigravity() -> tuple[tuple[int, int, int], str]:
    import numpy
    return tuple(sys.version_info[:3]), numpy.__version__


def _require_frozen_runtime_antigravity() -> None:
    python_version, numpy_version = _runtime_versions_antigravity()
    if python_version != _FROZEN_PYTHON_ANTIGRAVITY or numpy_version != _FROZEN_NUMPY_ANTIGRAVITY:
        raise ValueError("Exp448 requires exactly Python 3.14.6 and NumPy 2.4.6")


def verify_instances_schema_metadata_antigravity(descriptor: int) -> None:
    """Inspect only the two frozen SQLite column declarations, never label rows."""
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("database descriptor is invalid")
    connection = None
    try:
        import sqlite3
        connection = sqlite3.connect(f"file:/dev/fd/{descriptor}?mode=ro&immutable=1", uri=True, isolation_level=None)
        def authorizer(action: int, _first: str | None, _second: str | None, _database: str | None, _trigger: str | None) -> int:
            return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_PRAGMA else sqlite3.SQLITE_DENY
        connection.set_authorizer(authorizer)
        rows = tuple(connection.execute("PRAGMA table_info(instances)"))
        declarations = {(row[1], row[2]) for row in rows if len(row) >= 3 and row[1] in ("instance_index", "label")}
        if declarations != {("instance_index", "INTEGER"), ("label", "TEXT")}:
            raise ValueError("instances schema metadata differs from the TEXT boundary")
    finally:
        if connection is not None:
            connection.close()


def readiness_antigravity(reviewed_commit: str) -> dict[str, object]:
    """Verify state and metadata without constructing a TEST-label session or root."""
    _require_clean_reviewed_head_antigravity(reviewed_commit)
    _require_document_hashes_antigravity()
    verify_exp447_failure_authority_antigravity()
    require_absent_exp447_root_antigravity()
    _require_absent_run_root_antigravity()
    _require_frozen_runtime_antigravity()
    descriptor, _identity = _verify_database_antigravity()
    try:
        verify_instances_schema_metadata_antigravity(descriptor)
    finally:
        os.close(descriptor)
    identity = capture_exp446_publication_antigravity()
    hard = capture_hard_predictions_antigravity(identity)
    _require_clean_reviewed_head_antigravity(reviewed_commit)
    _require_absent_run_root_antigravity()
    return {
        "experiment_id": FROZEN_CONFIG_ANTIGRAVITY.experiment_id,
        "reviewed_commit": reviewed_commit,
        "dataset_count": len(identity.dataset_order),
        "hard_coordinate_count": len(hard.coordinates),
        "label_access_count": 0,
        "run_root_absent": True,
    }


__all__ = ["readiness_antigravity", "verify_instances_schema_metadata_antigravity"]
