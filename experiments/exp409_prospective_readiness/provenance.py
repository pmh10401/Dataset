"""Canonical component-level provenance for Exp409 publications."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp403_408.feature_registry import FEATURE_SOURCES
from experiments.exp403_408.synthetic_anomaly import MODEL_NAMES

from .config import FROZEN_CONFIG, Exp409FrozenConfig


_TOP_LEVEL_KEYS = frozenset(
    {
        "git",
        "config_sha256",
        "code_sha256",
        "database_snapshot_sha256",
        "lane_artifact_sha256",
        "source_artifact_sha256",
        "dataset_values_sha256",
        "source_output_sha256",
        "upstream_artifact_sha256",
        "candidate_sha256",
        "gates",
    }
)
_MAPPING_DIGEST_KEYS = (
    "code_sha256",
    "database_snapshot_sha256",
    "lane_artifact_sha256",
    "source_artifact_sha256",
    "dataset_values_sha256",
    "source_output_sha256",
    "upstream_artifact_sha256",
)
_PASSED_GATES = {
    "label_access": "passed",
    "lane_parity": "passed",
    "review_budget": "passed",
    "determinism": "passed",
}


def require_clean_git(
    repo_root: Path, *, git_executable: str | None = None
) -> str:
    """Return HEAD only when the exact worktree is clean."""
    executable = git_executable or shutil.which("git")
    if not executable:
        raise RuntimeError("git executable is unavailable")
    root = Path(repo_root).resolve()
    status = _git_status(root, executable)
    if status.strip():
        raise ValueError("Exp409 publication blocked by dirty Git worktree")
    commit = _git_commit(root, executable).strip()
    if not _is_git_commit(commit):
        raise ValueError("Exp409 Git commit identity is invalid")
    return commit


def _git_status(repo_root: Path, git_executable: str) -> str:
    return subprocess.run(
        [git_executable, "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def _git_commit(repo_root: Path, git_executable: str) -> str:
    return subprocess.run(
        [git_executable, "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def dataset_values_sha256(bundle: object) -> str:
    """Hash normalized float64 TRAIN/TEST values with field names and shapes."""
    digest = hashlib.sha256()
    for field_name in ("train_normal", "test_values"):
        values = np.ascontiguousarray(
            np.asarray(getattr(bundle, field_name), dtype=np.float64)
        )
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Exp409 dataset values are invalid: {field_name}"
            )
        digest.update(field_name.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def build_canonical_provenance(
    *,
    git_commit: str,
    code_sha256: Mapping[str, str],
    database_snapshot_sha256: Mapping[str, str],
    lane_artifact_sha256: Mapping[str, str],
    source_artifact_sha256: Mapping[str, str],
    dataset_values_sha256: Mapping[str, str],
    source_output_sha256: Mapping[str, str],
    upstream_artifact_sha256: Mapping[str, str],
    candidate_sha256: str,
    config: Exp409FrozenConfig = FROZEN_CONFIG,
) -> dict[str, object]:
    """Construct and validate the timestamp-free reproducibility closure."""
    if config != FROZEN_CONFIG:
        raise ValueError("Exp409 provenance configuration is not frozen")
    payload: dict[str, object] = {
        "git": {"commit": str(git_commit), "dirty": False},
        "config_sha256": hashlib.sha256(
            canonical_json_bytes(config.to_payload())
        ).hexdigest(),
        "code_sha256": dict(sorted(code_sha256.items())),
        "database_snapshot_sha256": dict(
            sorted(database_snapshot_sha256.items())
        ),
        "lane_artifact_sha256": dict(sorted(lane_artifact_sha256.items())),
        "source_artifact_sha256": dict(
            sorted(source_artifact_sha256.items())
        ),
        "dataset_values_sha256": dict(
            sorted(dataset_values_sha256.items())
        ),
        "source_output_sha256": dict(sorted(source_output_sha256.items())),
        "upstream_artifact_sha256": dict(
            sorted(upstream_artifact_sha256.items())
        ),
        "candidate_sha256": str(candidate_sha256),
        "gates": dict(_PASSED_GATES),
    }
    validate_canonical_provenance(payload)
    return payload


def validate_canonical_provenance(payload: Mapping[str, object]) -> None:
    """Fail closed unless every canonical provenance component is complete."""
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("Exp409 provenance top-level contract is invalid")

    git = payload["git"]
    if (
        not isinstance(git, Mapping)
        or set(git) != {"commit", "dirty"}
        or git.get("dirty") is not False
        or not _is_git_commit(git.get("commit"))
    ):
        raise ValueError("Exp409 Git provenance is invalid")

    if not _is_sha256(payload["config_sha256"]) or not _is_sha256(
        payload["candidate_sha256"]
    ):
        raise ValueError("Exp409 provenance digest is invalid")

    for key in _MAPPING_DIGEST_KEYS:
        values = payload[key]
        if (
            not isinstance(values, Mapping)
            or not values
            or any(
                not isinstance(name, str)
                or not name
                or not _is_sha256(digest)
                for name, digest in values.items()
            )
        ):
            raise ValueError(f"Exp409 provenance digest mapping is invalid: {key}")

    source_output = payload["source_output_sha256"]
    expected_sources = {source.name for source in FEATURE_SOURCES}
    expected_sources.add("tri_view")
    expected_sources.update(f"synthetic_{name}" for name in MODEL_NAMES)
    if set(source_output) != expected_sources:  # type: ignore[arg-type]
        raise ValueError("Exp409 source provenance coverage is invalid")

    if payload["gates"] != _PASSED_GATES:
        raise ValueError("Exp409 provenance gates are not all passed")


def provenance_closure_sha256(payload: Mapping[str, object]) -> str:
    """Hash only the canonical deterministic provenance payload."""
    validate_canonical_provenance(payload)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_runtime_envelope(run_id: str, started_at: str) -> dict[str, object]:
    """Record non-canonical execution context under its own content hash."""
    if not str(run_id).strip() or not str(started_at).strip():
        raise ValueError("Exp409 runtime identity is required")
    package_versions = {
        package: _package_version(package)
        for package in ("numpy", "scipy", "scikit-learn", "torch")
    }
    content: dict[str, object] = {
        "run_id": str(run_id),
        "started_at": str(started_at),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
    }
    return {
        **content,
        "runtime_envelope_sha256": hashlib.sha256(
            canonical_json_bytes(content)
        ).hexdigest(),
    }


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
