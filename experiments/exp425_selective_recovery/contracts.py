"""Immutable contracts for Exp425's TRAIN-only eligibility preflight."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import TypeAlias

from experiments.exp417_causal_multihorizon_shadow.contracts import ProcessedGroup


_EVIDENCE_STATUSES = frozenset(
    {"eligible", "oof_identity_insufficient", "preprocessing_abstained"}
)
_USABLE_PROCESSED_STATUSES = frozenset({"candidate_eligible", "diagnostic_only"})
_HEX_DIGITS = frozenset("0123456789abcdef")


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
        raise ValueError(f"{name} must be a 64-character lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class FoldIdentityCounts:
    """Identity cardinalities for one outer OOF fold and its early split."""

    oof_fit: int
    oof_validation: int
    early_stop_fit: int
    early_stop_validation: int

    def __post_init__(self) -> None:
        for name in (
            "oof_fit",
            "oof_validation",
            "early_stop_fit",
            "early_stop_validation",
        ):
            object.__setattr__(self, name, _nonnegative_integer(getattr(self, name), name))


@dataclass(frozen=True)
class GroupEligibilityEvidence:
    """Deterministic identity sufficiency evidence for one exact-length group."""

    dataset: str
    length: int
    processed_status: str
    train_identity_sha256: str
    fold_contract_sha256: str
    unique_identity_count: int
    folds: tuple[FoldIdentityCounts, ...]
    final_fit: int | None
    final_validation: int | None
    status: str
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _nonempty_string(self.dataset, "dataset"))
        object.__setattr__(self, "length", _positive_integer(self.length, "length"))
        object.__setattr__(
            self, "processed_status", _nonempty_string(self.processed_status, "processed_status")
        )
        object.__setattr__(
            self, "train_identity_sha256", _sha256_digest(self.train_identity_sha256, "train_identity_sha256")
        )
        object.__setattr__(
            self, "fold_contract_sha256", _sha256_digest(self.fold_contract_sha256, "fold_contract_sha256")
        )
        object.__setattr__(
            self, "unique_identity_count", _nonnegative_integer(self.unique_identity_count, "unique_identity_count")
        )
        if isinstance(self.folds, (str, bytes)) or not isinstance(self.folds, Sequence):
            raise ValueError("folds must be a sequence")
        folds = tuple(self.folds)
        if any(not isinstance(fold, FoldIdentityCounts) for fold in folds):
            raise ValueError("folds must contain FoldIdentityCounts")
        object.__setattr__(self, "folds", folds)
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError("invalid eligibility evidence status")
        if self.status == "preprocessing_abstained":
            if self.processed_status in _USABLE_PROCESSED_STATUSES:
                raise ValueError("preprocessing_abstained requires an abstained processed status")
            if folds or self.final_fit is not None or self.final_validation is not None:
                raise ValueError("preprocessing_abstained requires empty folds and unavailable final counts")
            if self.reason != self.processed_status:
                raise ValueError("preprocessing_abstained reason must preserve processed_status")
            return

        if self.processed_status not in _USABLE_PROCESSED_STATUSES:
            raise ValueError("OOF eligibility evidence requires a usable processed status")
        if len(folds) != 5:
            raise ValueError("OOF eligibility evidence requires exactly five folds")
        if self.processed_status == "candidate_eligible":
            final_fit = _nonnegative_integer(self.final_fit, "final_fit")
            final_validation = _nonnegative_integer(self.final_validation, "final_validation")
            object.__setattr__(self, "final_fit", final_fit)
            object.__setattr__(self, "final_validation", final_validation)
        elif self.final_fit is not None or self.final_validation is not None:
            raise ValueError("diagnostic_only evidence requires unavailable final counts")

        split_passes = all(
            fold.early_stop_fit >= 5 and fold.early_stop_validation >= 5 for fold in folds
        )
        if self.processed_status == "candidate_eligible":
            split_passes = bool(
                split_passes
                and self.final_fit is not None
                and self.final_validation is not None
                and self.final_fit >= 5
                and self.final_validation >= 5
            )
        if self.status == "eligible":
            if not split_passes:
                raise ValueError("eligible split requires at least five identities on both sides")
            if self.reason is not None:
                raise ValueError("eligible evidence reason must be None")
        else:
            if split_passes:
                raise ValueError("oof_identity_insufficient requires a failing split")
            if self.reason != "oof_identity_insufficient":
                raise ValueError("oof_identity_insufficient reason must be canonical")


@dataclass(frozen=True)
class EligibilityAssessment:
    """Groups after deterministic eligibility abstention plus evidence by length."""

    groups: tuple[ProcessedGroup, ...]
    evidence: Mapping[int, GroupEligibilityEvidence]

    def __post_init__(self) -> None:
        if isinstance(self.groups, (str, bytes)) or not isinstance(self.groups, Sequence):
            raise ValueError("groups must be a sequence")
        groups = tuple(self.groups)
        if any(not isinstance(group, ProcessedGroup) for group in groups):
            raise ValueError("groups must contain ProcessedGroup records")
        lengths = tuple(group.group.length for group in groups)
        if lengths != tuple(sorted(lengths)) or len(lengths) != len(set(lengths)):
            raise ValueError("group lengths must be sorted and unique")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")
        evidence = dict(self.evidence)
        if tuple(sorted(evidence)) != lengths:
            raise ValueError("evidence must cover the exact sorted group lengths")
        for length, item in evidence.items():
            if type(length) is not int or not isinstance(item, GroupEligibilityEvidence):
                raise ValueError("evidence must map group lengths to GroupEligibilityEvidence")
            if item.length != length:
                raise ValueError("evidence length must match its mapping key")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "evidence", MappingProxyType(evidence))


def _frozen_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _freeze_payload(value: object, name: str) -> object:
    """Detach a canonical JSON payload from all caller-owned containers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{name} contains a nonfinite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} mapping keys must be strings")
            frozen[key] = _freeze_payload(nested, name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(nested, name) for nested in value)
    raise ValueError(f"{name} contains a noncanonical payload value")


def _freeze_payload_mapping(value: object, name: str) -> Mapping[str, object]:
    frozen = _freeze_payload(value, name)
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return frozen


def thaw_payload(value: object) -> object:
    """Convert recursively-frozen canonical payload values back to JSON values."""
    if isinstance(value, Mapping):
        return {key: thaw_payload(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [thaw_payload(nested) for nested in value]
    return value


class _DeepImmutable:
    """Frozen payload contracts are safe to share and therefore safe to deepcopy."""

    def __copy__(self) -> object:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        memo[id(self)] = self
        return self


def _finite_elapsed(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class Exp425RunProvenance(_DeepImmutable):
    """Immutable identity for a compact Exp425 checkpoint root."""

    run_identity: str
    git_commit: str
    database_sha256: str
    lane_artifact_sha256: Mapping[str, str]
    frozen_config_sha256: str
    fold_contract_sha256: str
    model_contract_sha256: str
    backend_contract_sha256: str
    model_seeds: tuple[int, ...]
    perturbation_seeds: tuple[int, ...]
    runtime_versions: Mapping[str, str]
    expected_datasets: tuple[str, ...]
    predecessor_compatibility_sha256: str
    run_context_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_identity", "git_commit"):
            object.__setattr__(self, name, _nonempty_string(getattr(self, name), name))
        for name in ("database_sha256", "frozen_config_sha256", "fold_contract_sha256", "model_contract_sha256", "backend_contract_sha256", "predecessor_compatibility_sha256"):
            object.__setattr__(self, name, _sha256_digest(getattr(self, name), name))
        if self.run_context_sha256 is not None:
            object.__setattr__(self, "run_context_sha256", _sha256_digest(self.run_context_sha256, "run_context_sha256"))
        lanes = _frozen_mapping(self.lane_artifact_sha256, "lane_artifact_sha256")
        if set(lanes) != {"hard", "standard", "priority", "automatic_normal"}:
            raise ValueError("lane_artifact_sha256 must contain exactly four lanes")
        for name, digest in lanes.items():
            _sha256_digest(digest, f"lane_artifact_sha256[{name}]")
        object.__setattr__(self, "lane_artifact_sha256", lanes)
        for name in ("model_seeds", "perturbation_seeds"):
            values = tuple(getattr(self, name))
            if not values or any(type(item) is not int for item in values):
                raise ValueError(f"{name} must contain integers")
            object.__setattr__(self, name, values)
        versions = _frozen_mapping(self.runtime_versions, "runtime_versions")
        if not versions or any(not isinstance(key, str) or not isinstance(value, str) for key, value in versions.items()):
            raise ValueError("runtime_versions must contain strings")
        object.__setattr__(self, "runtime_versions", versions)
        datasets = tuple(self.expected_datasets)
        if not datasets or datasets != tuple(sorted(datasets)) or len(datasets) != len(set(datasets)):
            raise ValueError("expected_datasets must be sorted and unique")
        if any(not isinstance(dataset, str) or not dataset for dataset in datasets):
            raise ValueError("expected_datasets must contain names")
        object.__setattr__(self, "expected_datasets", datasets)

    def with_run_context_sha256(self, digest: str) -> "Exp425RunProvenance":
        return Exp425RunProvenance(self.run_identity, self.git_commit, self.database_sha256, self.lane_artifact_sha256, self.frozen_config_sha256, self.fold_contract_sha256, self.model_contract_sha256, self.backend_contract_sha256, self.model_seeds, self.perturbation_seeds, self.runtime_versions, self.expected_datasets, self.predecessor_compatibility_sha256, digest)

    def to_payload(self) -> dict[str, object]:
        if self.run_context_sha256 is None:
            raise ValueError("run_context_sha256 is required for checkpoint provenance")
        return {"schema": "exp425.checkpoint.v1", "run_identity": self.run_identity, "git_commit": self.git_commit, "database_sha256": self.database_sha256, "lane_artifact_sha256": dict(sorted(self.lane_artifact_sha256.items())), "frozen_config_sha256": self.frozen_config_sha256, "fold_contract_sha256": self.fold_contract_sha256, "model_contract_sha256": self.model_contract_sha256, "backend_contract_sha256": self.backend_contract_sha256, "model_seeds": list(self.model_seeds), "perturbation_seeds": list(self.perturbation_seeds), "runtime_versions": dict(sorted(self.runtime_versions.items())), "expected_datasets": list(self.expected_datasets), "predecessor_compatibility_sha256": self.predecessor_compatibility_sha256, "run_context_sha256": self.run_context_sha256}


@dataclass(frozen=True)
class SourceProvenGroupEvidence(_DeepImmutable):
    source_group_evidence: Mapping[str, object]
    source_group_evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_group_evidence", _freeze_payload_mapping(self.source_group_evidence, "source_group_evidence"))
        object.__setattr__(self, "source_group_evidence_sha256", _sha256_digest(self.source_group_evidence_sha256, "source_group_evidence_sha256"))


@dataclass(frozen=True)
class NativeRecordEnvelope(_DeepImmutable):
    dataset: str; result_payload: Mapping[str, object]; group_evidence: Mapping[str, object]
    train_identity_sha256: str; eligibility_sha256: str; processed_groups_sha256: str; model_evidence_sha256: str
    frozen_train_contract_sha256: str; scoring_evidence_sha256: str; result_evidence_sha256: str
    model_hashes: Mapping[str, object]; calibration_hashes: Mapping[str, object]; backend_audit_sha256: str
    trace_sha256: str; record_payload_sha256: str; elapsed_seconds: float; reason: str | None; origin: str = "exp425_native"; context: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _nonempty_string(self.dataset, "dataset"))
        if self.origin != "exp425_native": raise ValueError("native record origin is invalid")
        for name in ("train_identity_sha256", "eligibility_sha256", "processed_groups_sha256", "model_evidence_sha256", "frozen_train_contract_sha256", "scoring_evidence_sha256", "result_evidence_sha256", "backend_audit_sha256", "trace_sha256", "record_payload_sha256"):
            object.__setattr__(self, name, _sha256_digest(getattr(self, name), name))
        for name in ("result_payload", "group_evidence", "model_hashes", "calibration_hashes"):
            object.__setattr__(self, name, _freeze_payload_mapping(getattr(self, name), name))
        object.__setattr__(self, "elapsed_seconds", _finite_elapsed(self.elapsed_seconds))
        if self.reason is not None and not isinstance(self.reason, str): raise ValueError("reason must be a string or None")


@dataclass(frozen=True)
class ImportedRecordEnvelope(_DeepImmutable):
    dataset: str; result_payload: Mapping[str, object]; group_evidence: SourceProvenGroupEvidence
    source_root_sha256: str; source_progress_sha256: str; source_record_sha256: str; source_result_sha256: str
    compatibility_attestation_sha256: str; elapsed_seconds: float; reason: str | None; origin: str = "exp424_import"; context: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _nonempty_string(self.dataset, "dataset"))
        if self.origin != "exp424_import": raise ValueError("imported record origin is invalid")
        if not isinstance(self.group_evidence, SourceProvenGroupEvidence): raise ValueError("imported group evidence is invalid")
        object.__setattr__(self, "result_payload", _freeze_payload_mapping(self.result_payload, "result_payload"))
        for name in ("source_root_sha256", "source_progress_sha256", "source_record_sha256", "source_result_sha256", "compatibility_attestation_sha256"):
            object.__setattr__(self, name, _sha256_digest(getattr(self, name), name))
        object.__setattr__(self, "elapsed_seconds", _finite_elapsed(self.elapsed_seconds))
        if self.reason is not None and not isinstance(self.reason, str): raise ValueError("reason must be a string or None")


Exp425Record: TypeAlias = NativeRecordEnvelope | ImportedRecordEnvelope
