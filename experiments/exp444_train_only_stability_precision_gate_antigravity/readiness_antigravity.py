"""Label-free Exp444 readiness evidence for the fixed lifecycle."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Mapping

from .config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity
from .environment_antigravity import verify_environment_antigravity
from .provenance_antigravity import verify_exp444_provenance_antigravity
from .sources_antigravity import load_candidate_values_antigravity, verify_frozen_sources_antigravity
from .stability_antigravity import replay_original_candidates_antigravity


_HEX_ANTIGRAVITY = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class ReadinessEvidenceAntigravity:
    """Detached proof that the frozen inputs may be handed to a new runtime."""

    reviewed_commit: str
    source_hashes: Mapping[str, str]
    source_candidate_count: int
    control_count: int
    label_access_count: int
    backend: str
    workers: int
    environment_lock_sha256: str
    scientific_config_sha256: str

    def __post_init__(self) -> None:
        if type(self.reviewed_commit) is not str or len(self.reviewed_commit) != 40 or any(char not in _HEX_ANTIGRAVITY for char in self.reviewed_commit):
            raise ValueError("Exp444 readiness reviewed commit is invalid")
        if self.source_candidate_count != FROZEN_CONFIG.source_candidate_count or self.control_count != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("Exp444 readiness candidate/control count is invalid")
        if self.label_access_count != 0 or self.backend != "cpu" or self.workers != FROZEN_CONFIG.cpu_workers:
            raise ValueError("Exp444 readiness governance is invalid")
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "schema": "exp444.readiness.v1", "reviewed_commit": self.reviewed_commit,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "source_candidate_count": self.source_candidate_count, "control_count": self.control_count,
            "label_access_count": self.label_access_count, "backend": self.backend,
            "workers": self.workers, "environment_lock_sha256": self.environment_lock_sha256,
            "scientific_config_sha256": self.scientific_config_sha256,
        }


def _run_root_antigravity() -> Path:
    return Path(FROZEN_CONFIG.run_root)


def current_clean_head_antigravity() -> str:
    root = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True):
        raise ValueError("Exp444 readiness requires a clean working tree")
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if len(value) != 40 or any(char not in _HEX_ANTIGRAVITY for char in value):
        raise ValueError("Exp444 readiness clean HEAD is invalid")
    return value


def _materialize_values_antigravity() -> tuple[tuple[object, ...], Mapping[str, str], int]:
    snapshot = verify_frozen_sources_antigravity()
    session = snapshot.session
    try:
        inputs = tuple(load_candidate_values_antigravity(snapshot))
        labels = snapshot.label_access_count
        if labels != 0 or session is None or session.label_access_count != 0:
            raise ValueError("Exp444 readiness source accessed TEST labels")
        if len(inputs) != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("Exp444 readiness source candidate count is invalid")
        return inputs, dict(snapshot.source_hashes), labels
    finally:
        if session is not None:
            session.close()


def _readiness_payload_antigravity(
    reviewed_commit: str, *, source_hashes: Mapping[str, str], source_candidate_count: int,
    control_count: int, label_access_count: int,
) -> dict[str, object]:
    return ReadinessEvidenceAntigravity(
        reviewed_commit, source_hashes, source_candidate_count, control_count, label_access_count,
        "cpu", FROZEN_CONFIG.cpu_workers, FROZEN_CONFIG.environment_lock_sha256,
        scientific_config_sha256_antigravity(),
    ).to_payload_antigravity()


def validate_readiness_payload_antigravity(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "reviewed_commit", "source_hashes", "source_candidate_count", "control_count",
        "label_access_count", "backend", "workers", "environment_lock_sha256", "scientific_config_sha256",
    }:
        raise ValueError("Exp444 readiness payload schema is invalid")
    hashes = payload["source_hashes"]
    if not isinstance(hashes, Mapping) or not hashes or any(type(name) is not str or type(value) is not str or len(value) != 64 or any(char not in _HEX_ANTIGRAVITY for char in value) for name, value in hashes.items()):
        raise ValueError("Exp444 readiness source hashes are invalid")
    evidence = ReadinessEvidenceAntigravity(
        payload["reviewed_commit"], dict(hashes), payload["source_candidate_count"], payload["control_count"],
        payload["label_access_count"], payload["backend"], payload["workers"],
        payload["environment_lock_sha256"], payload["scientific_config_sha256"],
    )
    if evidence.environment_lock_sha256 != FROZEN_CONFIG.environment_lock_sha256 or evidence.scientific_config_sha256 != scientific_config_sha256_antigravity():
        raise ValueError("Exp444 readiness frozen identity changed")
    return evidence.to_payload_antigravity()


def validate_readiness_for_run_antigravity(reviewed_commit: str) -> ReadinessEvidenceAntigravity:
    """Recheck only immutable local identities; runtime re-materializes before mutation."""
    if current_clean_head_antigravity() != reviewed_commit:
        raise ValueError("Exp444 reviewed commit is not the current clean HEAD")
    root = _run_root_antigravity()
    if root.exists() or root.is_symlink():
        raise ValueError("Exp444 fixed new run root must be absent")
    return run_readiness_antigravity(reviewed_commit)


def run_readiness_antigravity(reviewed_commit: str) -> ReadinessEvidenceAntigravity:
    """Verify all immutable label-free authority while leaving the official root absent."""
    if current_clean_head_antigravity() != reviewed_commit:
        raise ValueError("Exp444 reviewed commit is not the current clean HEAD")
    root = _run_root_antigravity()
    if root.exists() or root.is_symlink():
        raise ValueError("Exp444 fixed run root must be absent")
    FROZEN_CONFIG.validate()
    environment = verify_environment_antigravity()
    if environment.inner_threads != 1 or environment.outer_workers != FROZEN_CONFIG.cpu_workers:
        raise ValueError("Exp444 readiness environment topology changed")
    provenance = verify_exp444_provenance_antigravity()
    if provenance.backend_decision.backend != "cpu" or provenance.backend_decision.workers != FROZEN_CONFIG.cpu_workers:
        raise ValueError("Exp444 readiness backend decision changed")
    inputs, source_hashes, labels = _materialize_values_antigravity()
    controls = replay_original_candidates_antigravity(inputs)  # type: ignore[arg-type]
    if len(controls) != FROZEN_CONFIG.source_candidate_count or any(not item.matches_source for item in controls):
        raise ValueError("Exp444 readiness original control replay failed")
    payload = _readiness_payload_antigravity(
        reviewed_commit, source_hashes=source_hashes, source_candidate_count=len(inputs),
        control_count=len(controls), label_access_count=labels,
    )
    validate_readiness_payload_antigravity(payload)
    return ReadinessEvidenceAntigravity(
        reviewed_commit, source_hashes, len(inputs), len(controls), labels, "cpu", FROZEN_CONFIG.cpu_workers,
        FROZEN_CONFIG.environment_lock_sha256, scientific_config_sha256_antigravity(),
    )


__all__ = ["ReadinessEvidenceAntigravity", "current_clean_head_antigravity", "run_readiness_antigravity", "validate_readiness_for_run_antigravity"]
