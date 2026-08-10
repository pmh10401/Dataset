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
from .provenance_antigravity import (
    validate_worker_parity_authority_payload_antigravity,
    verify_exp444_provenance_antigravity,
)
from .artifacts_antigravity import parse_canonical_json_bytes_antigravity
from .sources_antigravity import load_candidate_values_antigravity, verify_frozen_sources_antigravity
from .stability_antigravity import replay_original_candidates_antigravity


_HEX_ANTIGRAVITY = frozenset("0123456789abcdef")
_REVIEWED_DESIGN_PATH_ANTIGRAVITY = Path(__file__).resolve().parents[2] / "docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md"
_PREREGISTRATION_PATH_ANTIGRAVITY = Path(__file__).resolve().parents[2] / "docs/EXP444_PREREGISTRATION_antigravity_KO.md"
_WORKER_PARITY_AUTHORITY_PATH_ANTIGRAVITY = Path(__file__).with_name(FROZEN_CONFIG.worker_parity_authority_identifier)


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
    reviewed_design_sha256: str
    preregistration_sha256: str
    worker_parity_authority_sha256: str

    def __post_init__(self) -> None:
        if type(self.reviewed_commit) is not str or len(self.reviewed_commit) != 40 or any(char not in _HEX_ANTIGRAVITY for char in self.reviewed_commit):
            raise ValueError("Exp444 readiness reviewed commit is invalid")
        if self.source_candidate_count != FROZEN_CONFIG.source_candidate_count or self.control_count != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("Exp444 readiness candidate/control count is invalid")
        if self.label_access_count != 0 or self.backend != "cpu" or self.workers != FROZEN_CONFIG.cpu_workers:
            raise ValueError("Exp444 readiness governance is invalid")
        if (
            self.reviewed_design_sha256 != FROZEN_CONFIG.reviewed_design_sha256
            or self.preregistration_sha256 != FROZEN_CONFIG.preregistration_sha256
            or self.worker_parity_authority_sha256 != FROZEN_CONFIG.worker_parity_authority_sha256
        ):
            raise ValueError("Exp444 readiness reviewed evidence is invalid")
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))

    def to_payload_antigravity(self) -> dict[str, object]:
        return {
            "schema": "exp444.readiness.v1", "reviewed_commit": self.reviewed_commit,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "source_candidate_count": self.source_candidate_count, "control_count": self.control_count,
            "label_access_count": self.label_access_count, "backend": self.backend,
            "workers": self.workers, "environment_lock_sha256": self.environment_lock_sha256,
            "scientific_config_sha256": self.scientific_config_sha256,
            "reviewed_design_sha256": self.reviewed_design_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "worker_parity_authority_sha256": self.worker_parity_authority_sha256,
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


def _review_document_sha256_antigravity() -> dict[str, str]:
    """Open both reviewed source documents and require their frozen byte hashes."""
    entries = (
        ("reviewed_design_sha256", _REVIEWED_DESIGN_PATH_ANTIGRAVITY, FROZEN_CONFIG.reviewed_design_sha256, "design"),
        ("preregistration_sha256", _PREREGISTRATION_PATH_ANTIGRAVITY, FROZEN_CONFIG.preregistration_sha256, "preregistration"),
    )
    output: dict[str, str] = {}
    for key, path, expected, name in entries:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Exp444 reviewed {name} is absent or invalid")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Exp444 reviewed {name} SHA256 changed")
        output[key] = digest
    return output


def load_worker_parity_authority_antigravity() -> dict[str, object]:
    """Validate the exact tracked Task7 authority from independently pinned bytes."""
    path = _WORKER_PARITY_AUTHORITY_PATH_ANTIGRAVITY
    if path.is_symlink() or not path.is_file():
        raise ValueError("Task7 worker parity authority is absent or invalid")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != FROZEN_CONFIG.worker_parity_authority_sha256:
        raise ValueError("Task7 worker parity authority SHA256 changed")
    try:
        payload = parse_canonical_json_bytes_antigravity(raw)
    except ValueError as error:
        raise ValueError("Task7 worker parity authority is not canonical") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Task7 worker parity authority is invalid")
    return validate_worker_parity_authority_payload_antigravity(payload)


def validate_execution_identity_antigravity(reviewed_commit: str) -> str:
    """Validate the exact current commit plus documents and Task7 worker authority."""
    if type(reviewed_commit) is not str or len(reviewed_commit) != 40 or any(char not in _HEX_ANTIGRAVITY for char in reviewed_commit):
        raise ValueError("Exp444 reviewed commit is invalid")
    if current_clean_head_antigravity() != reviewed_commit:
        raise ValueError("Exp444 reviewed commit is not the current clean HEAD")
    _review_document_sha256_antigravity()
    load_worker_parity_authority_antigravity()
    return reviewed_commit


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
    documents = _review_document_sha256_antigravity()
    load_worker_parity_authority_antigravity()
    return ReadinessEvidenceAntigravity(
        reviewed_commit, source_hashes, source_candidate_count, control_count, label_access_count,
        "cpu", FROZEN_CONFIG.cpu_workers, FROZEN_CONFIG.environment_lock_sha256,
        scientific_config_sha256_antigravity(), documents["reviewed_design_sha256"],
        documents["preregistration_sha256"], FROZEN_CONFIG.worker_parity_authority_sha256,
    ).to_payload_antigravity()


def validate_readiness_payload_antigravity(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "reviewed_commit", "source_hashes", "source_candidate_count", "control_count",
        "label_access_count", "backend", "workers", "environment_lock_sha256", "scientific_config_sha256",
        "reviewed_design_sha256", "preregistration_sha256", "worker_parity_authority_sha256",
    }:
        raise ValueError("Exp444 readiness payload schema is invalid")
    hashes = payload["source_hashes"]
    if not isinstance(hashes, Mapping) or not hashes or any(type(name) is not str or type(value) is not str or len(value) != 64 or any(char not in _HEX_ANTIGRAVITY for char in value) for name, value in hashes.items()):
        raise ValueError("Exp444 readiness source hashes are invalid")
    evidence = ReadinessEvidenceAntigravity(
        payload["reviewed_commit"], dict(hashes), payload["source_candidate_count"], payload["control_count"],
        payload["label_access_count"], payload["backend"], payload["workers"],
        payload["environment_lock_sha256"], payload["scientific_config_sha256"],
        payload["reviewed_design_sha256"], payload["preregistration_sha256"], payload["worker_parity_authority_sha256"],
    )
    if evidence.environment_lock_sha256 != FROZEN_CONFIG.environment_lock_sha256 or evidence.scientific_config_sha256 != scientific_config_sha256_antigravity():
        raise ValueError("Exp444 readiness frozen identity changed")
    return evidence.to_payload_antigravity()


def validate_readiness_for_run_antigravity(reviewed_commit: str) -> ReadinessEvidenceAntigravity:
    """Recheck only immutable local identities; runtime re-materializes before mutation."""
    validate_execution_identity_antigravity(reviewed_commit)
    root = _run_root_antigravity()
    if root.exists() or root.is_symlink():
        raise ValueError("Exp444 fixed new run root must be absent")
    return run_readiness_antigravity(reviewed_commit)


def run_readiness_antigravity(reviewed_commit: str) -> ReadinessEvidenceAntigravity:
    """Verify all immutable label-free authority while leaving the official root absent."""
    validate_execution_identity_antigravity(reviewed_commit)
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
        FROZEN_CONFIG.reviewed_design_sha256, FROZEN_CONFIG.preregistration_sha256,
        FROZEN_CONFIG.worker_parity_authority_sha256,
    )


__all__ = [
    "ReadinessEvidenceAntigravity", "current_clean_head_antigravity", "run_readiness_antigravity",
    "validate_readiness_for_run_antigravity", "validate_execution_identity_antigravity",
    "load_worker_parity_authority_antigravity", "validate_worker_parity_authority_payload_antigravity",
]
