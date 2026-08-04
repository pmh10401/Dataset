"""Immutable contracts for Exp425's TRAIN-only eligibility preflight."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

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
