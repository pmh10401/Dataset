"""Strict immutable evidence contracts for the label-free Exp441 replay."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} is invalid")
    return value


def _exact_payload(payload: Mapping[str, object], keys: set[str]) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise ValueError("payload keys are invalid")
    return dict(payload)


def _immutable_int_map(value: Mapping[str, object], name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} key is invalid")
        copied[key] = _positive_int(item, name, allow_zero=True)
    return MappingProxyType(copied)


def _immutable_digest_map(value: Mapping[str, object]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("record digest map must be a mapping")
    copied: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or len(item) != 64:
            raise ValueError("record digest map is invalid")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True)
class CandidateEvidenceAntigravity:
    """One copied Exp440 provisional candidate under Exp441's rank policy."""

    dataset: str
    instance_index: int
    raw_length: int
    depths: tuple[int, ...]
    agreement_count: int
    second_excess_ratio: float
    second_empirical_rank: float

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("candidate dataset is invalid")
        _positive_int(self.instance_index, "instance index", allow_zero=True)
        _positive_int(self.raw_length, "raw length")
        depths = tuple(self.depths)
        if len(depths) < 2 or depths != tuple(sorted(set(depths))) or any(type(depth) is not int or depth < 1 for depth in depths):
            raise ValueError("candidate depths are invalid")
        if type(self.agreement_count) is not int or self.agreement_count < 2 or self.agreement_count != len(depths):
            raise ValueError("candidate agreement count is invalid")
        ratio = _finite(self.second_excess_ratio, "second excess ratio", nonnegative=True)
        rank = _finite(self.second_empirical_rank, "second empirical rank", nonnegative=True)
        if rank <= 0.0 or rank > 1.0:
            raise ValueError("second empirical rank is invalid")
        object.__setattr__(self, "depths", depths)
        object.__setattr__(self, "second_excess_ratio", ratio)
        object.__setattr__(self, "second_empirical_rank", rank)

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.dataset, self.instance_index

    @property
    def dataset_key(self) -> tuple[float, float, int]:
        return -self.second_excess_ratio, self.second_empirical_rank, self.instance_index

    @property
    def global_key(self) -> tuple[float, float, str, int]:
        return -self.second_excess_ratio, self.second_empirical_rank, self.dataset, self.instance_index

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset, "instance_index": self.instance_index,
            "raw_length": self.raw_length, "depths": list(self.depths),
            "agreement_count": self.agreement_count,
            "second_excess_ratio": self.second_excess_ratio,
            "second_empirical_rank": self.second_empirical_rank,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CandidateEvidenceAntigravity":
        raw = _exact_payload(payload, {
            "dataset", "instance_index", "raw_length", "depths", "agreement_count",
            "second_excess_ratio", "second_empirical_rank",
        })
        if not isinstance(raw["depths"], list):
            raise ValueError("candidate depths must be a list")
        raw["depths"] = tuple(raw["depths"])
        return cls(**raw)


@dataclass(frozen=True)
class RankedCandidateAntigravity:
    """A candidate with complete global and optional retained selection rank."""

    candidate: CandidateEvidenceAntigravity
    global_rank: int
    selected: bool
    selection_rank: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateEvidenceAntigravity):
            raise ValueError("ranked candidate is invalid")
        _positive_int(self.global_rank, "global rank")
        if type(self.selected) is not bool:
            raise ValueError("selected must be a bool")
        if self.selected:
            _positive_int(self.selection_rank, "selection rank")
        elif self.selection_rank is not None:
            raise ValueError("unselected candidate has a selection rank")

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.candidate.coordinate

    @property
    def dataset(self) -> str:
        return self.candidate.dataset

    @property
    def instance_index(self) -> int:
        return self.candidate.instance_index

    @property
    def global_key(self) -> tuple[float, float, str, int]:
        return self.candidate.global_key

    def to_payload(self) -> dict[str, object]:
        payload = self.candidate.to_payload()
        payload.update({
            "global_rank": self.global_rank, "selected": self.selected,
            "selection_rank": self.selection_rank,
        })
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RankedCandidateAntigravity":
        raw = _exact_payload(payload, {
            "dataset", "instance_index", "raw_length", "depths", "agreement_count",
            "second_excess_ratio", "second_empirical_rank", "global_rank", "selected",
            "selection_rank",
        })
        candidate_fields = {name: raw.pop(name) for name in tuple(CandidateEvidenceAntigravity.__dataclass_fields__)}
        return cls(CandidateEvidenceAntigravity.from_payload(candidate_fields), **raw)


@dataclass(frozen=True)
class ParsedRecordAntigravity:
    """One parsed source record, detached from its caller-owned candidate container."""

    dataset: str
    test_row_count: int
    candidates: tuple[CandidateEvidenceAntigravity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("record dataset is invalid")
        _positive_int(self.test_row_count, "test row count", allow_zero=True)
        candidates = tuple(self.candidates)
        if any(not isinstance(item, CandidateEvidenceAntigravity) or item.dataset != self.dataset for item in candidates):
            raise ValueError("record candidates are invalid")
        coordinates = [item.coordinate for item in candidates]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("record candidates are duplicated")
        if any(item.instance_index >= self.test_row_count for item in candidates):
            raise ValueError("record candidate index is outside TEST identity")
        object.__setattr__(self, "candidates", candidates)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset, "test_row_count": self.test_row_count,
            "candidates": [item.to_payload() for item in self.candidates],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ParsedRecordAntigravity":
        raw = _exact_payload(payload, {"dataset", "test_row_count", "candidates"})
        if not isinstance(raw["candidates"], list):
            raise ValueError("record candidates must be a list")
        raw["candidates"] = tuple(CandidateEvidenceAntigravity.from_payload(item) for item in raw["candidates"])
        return cls(**raw)


@dataclass(frozen=True)
class SelectionResultAntigravity:
    """All pure-selector populations, including sparse fixtures, under one invariant."""

    provisional_candidates: tuple[CandidateEvidenceAntigravity, ...]
    outside_hard_candidates: tuple[CandidateEvidenceAntigravity, ...]
    eligible_candidates: tuple[CandidateEvidenceAntigravity, ...]
    winners: tuple[RankedCandidateAntigravity, ...]
    selected: tuple[RankedCandidateAntigravity, ...]
    hard_coordinates: frozenset[tuple[str, int]]

    def __post_init__(self) -> None:
        provisional = tuple(self.provisional_candidates)
        outside = tuple(self.outside_hard_candidates)
        eligible = tuple(self.eligible_candidates)
        winners = tuple(self.winners)
        selected = tuple(self.selected)
        hard = frozenset(self.hard_coordinates)
        if any(not isinstance(item, CandidateEvidenceAntigravity) for item in provisional + outside + eligible):
            raise ValueError("selection candidate population is invalid")
        if any(not isinstance(item, RankedCandidateAntigravity) for item in winners + selected):
            raise ValueError("selection ranked population is invalid")
        if any(not isinstance(dataset, str) or not dataset or type(index) is not int or index < 0 for dataset, index in hard):
            raise ValueError("Hard coordinates are invalid")
        coordinates = [item.coordinate for item in provisional]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("provisional coordinates are duplicated")
        if set(outside) != {item for item in provisional if item.coordinate not in hard}:
            raise ValueError("outside-Hard population is invalid")
        hard_datasets = {dataset for dataset, _ in hard}
        if set(eligible) != {item for item in outside if item.dataset not in hard_datasets}:
            raise ValueError("Hard-empty eligibility population is invalid")
        winner_candidates = tuple(item.candidate for item in winners)
        if any(item.dataset in hard_datasets for item in winner_candidates + tuple(item.candidate for item in selected)):
            raise ValueError("Hard-active dataset was selected")
        if set(winner_candidates) != set(minimum for minimum in winner_candidates) or not set(winner_candidates).issubset(set(eligible)):
            raise ValueError("winner population is invalid")
        winner_datasets = [item.dataset for item in winners]
        if len(winner_datasets) != len(set(winner_datasets)):
            raise ValueError("more than one winner per dataset")
        if tuple(item.global_rank for item in winners) != tuple(range(1, len(winners) + 1)):
            raise ValueError("global ranks are invalid")
        if set(selected) - set(winners) or len(selected) > 30:
            raise ValueError("selected population is invalid")
        selected_coordinates = [item.coordinate for item in selected]
        if len(selected_coordinates) != len(set(selected_coordinates)) or len({item.dataset for item in selected}) != len(selected):
            raise ValueError("selected coordinates are duplicated")
        if tuple(item.selection_rank for item in selected) != tuple(range(1, len(selected) + 1)):
            raise ValueError("selection ranks are invalid")
        if any(not item.selected for item in selected):
            raise ValueError("selected records are not marked selected")
        selected_set = set(selected)
        if any(item.selected != (item in selected_set) for item in winners):
            raise ValueError("winner selected flags are invalid")
        if any((item.selection_rank is None) == item.selected for item in winners):
            raise ValueError("winner selection ranks are invalid")
        object.__setattr__(self, "provisional_candidates", provisional)
        object.__setattr__(self, "outside_hard_candidates", outside)
        object.__setattr__(self, "eligible_candidates", eligible)
        object.__setattr__(self, "winners", winners)
        object.__setattr__(self, "selected", selected)
        object.__setattr__(self, "hard_coordinates", hard)

    def to_payload(self) -> dict[str, object]:
        return {
            "provisional_candidates": [item.to_payload() for item in self.provisional_candidates],
            "outside_hard_candidates": [item.to_payload() for item in self.outside_hard_candidates],
            "eligible_candidates": [item.to_payload() for item in self.eligible_candidates],
            "winners": [item.to_payload() for item in self.winners],
            "selected": [item.to_payload() for item in self.selected],
            "hard_coordinates": [[dataset, index] for dataset, index in sorted(self.hard_coordinates)],
        }


def validate_production_selection_antigravity(result: SelectionResultAntigravity) -> SelectionResultAntigravity:
    """Require the one reviewed Exp441 population shape, never a padded fixture."""
    if not isinstance(result, SelectionResultAntigravity):
        raise ValueError("selection result is invalid")
    if (len(result.provisional_candidates), len(result.outside_hard_candidates), len(result.eligible_candidates), len(result.winners), len(result.selected)) != (840, 394, 73, 49, 30):
        raise ValueError("production selection structure is invalid")
    return result


@dataclass(frozen=True)
class FrozenSourceSnapshotAntigravity:
    """Fully copied, label-free source identity for source replay and selection."""

    context_datasets: tuple[str, ...]
    test_row_count_by_dataset: Mapping[str, int]
    record_digest_by_dataset: Mapping[str, str]
    hard_coordinates: frozenset[tuple[str, int]]
    provisional_candidates: tuple[CandidateEvidenceAntigravity, ...]
    context_sha256: str
    state_sha256: str
    retained_inventory_sha256: str
    publication_prediction_sha256: str
    publication_manifest_sha256: str
    publication_artifact_map_sha256: str
    hard_prediction_sha256: str
    hard_manifest_sha256: str
    hard_coordinate_identity_sha256: str
    label_access_count: int = 0

    def __post_init__(self) -> None:
        datasets = tuple(self.context_datasets)
        if not datasets or any(not isinstance(dataset, str) or not dataset for dataset in datasets) or datasets != tuple(sorted(set(datasets))):
            raise ValueError("context dataset order is invalid")
        rows = _immutable_int_map(self.test_row_count_by_dataset, "test row count")
        digests = _immutable_digest_map(self.record_digest_by_dataset)
        if set(rows) != set(datasets) or set(digests) != set(datasets):
            raise ValueError("source maps do not match context")
        hard = frozenset(self.hard_coordinates)
        candidates = tuple(self.provisional_candidates)
        if any(dataset not in rows or type(index) is not int or index < 0 or index >= rows[dataset] for dataset, index in hard):
            raise ValueError("Hard coordinate is outside source identity")
        if any(not isinstance(item, CandidateEvidenceAntigravity) or item.dataset not in rows or item.instance_index >= rows[item.dataset] for item in candidates):
            raise ValueError("provisional candidate is outside source identity")
        if len({item.coordinate for item in candidates}) != len(candidates):
            raise ValueError("provisional candidates are duplicated")
        hashes = (
            self.context_sha256, self.state_sha256, self.retained_inventory_sha256,
            self.publication_prediction_sha256, self.publication_manifest_sha256,
            self.publication_artifact_map_sha256, self.hard_prediction_sha256,
            self.hard_manifest_sha256, self.hard_coordinate_identity_sha256,
        )
        if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
            raise ValueError("source hash is invalid")
        if self.label_access_count != 0:
            raise ValueError("source snapshot accessed labels")
        object.__setattr__(self, "context_datasets", datasets)
        object.__setattr__(self, "test_row_count_by_dataset", rows)
        object.__setattr__(self, "record_digest_by_dataset", digests)
        object.__setattr__(self, "hard_coordinates", hard)
        object.__setattr__(self, "provisional_candidates", candidates)


@dataclass(frozen=True)
class BackendAssessmentAntigravity:
    canonical_backend: str
    workers: int
    tensor_eligible: bool
    acceleration_benchmark_status: str
    pytorch: str
    mlx: str
    mps: str
    environment: Mapping[str, object]

    def __post_init__(self) -> None:
        if (self.canonical_backend, self.acceleration_benchmark_status, self.pytorch, self.mlx, self.mps) != (
            "cpu", "not_run_inapplicable", "inapplicable_no_tensor_stage",
            "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage",
        ) or type(self.tensor_eligible) is not bool or self.tensor_eligible:
            raise ValueError("backend assessment is not frozen")
        _positive_int(self.workers, "workers")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@dataclass(frozen=True)
class ReadinessBindingAntigravity:
    reviewed_commit: str
    source_snapshot_sha256: str
    backend: BackendAssessmentAntigravity
    readiness_sha256: str

    def __post_init__(self) -> None:
        if (not isinstance(self.reviewed_commit, str) or len(self.reviewed_commit) != 40 or
                any(not isinstance(value, str) or len(value) != 64 for value in (self.source_snapshot_sha256, self.readiness_sha256))):
            raise ValueError("readiness identity is invalid")
        if not isinstance(self.backend, BackendAssessmentAntigravity):
            raise ValueError("readiness backend is invalid")


@dataclass(frozen=True)
class CompletedAttemptAntigravity:
    attempt_sequence: int
    reviewed_commit: str
    source_snapshot_sha256: str
    selection: SelectionResultAntigravity
    selection_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.attempt_sequence, "attempt sequence", allow_zero=True)
        if (not isinstance(self.reviewed_commit, str) or len(self.reviewed_commit) != 40 or
                any(not isinstance(value, str) or len(value) != 64 for value in (self.source_snapshot_sha256, self.selection_sha256))):
            raise ValueError("attempt identity is invalid")
        if not isinstance(self.selection, SelectionResultAntigravity):
            raise ValueError("attempt selection is invalid")


@dataclass(frozen=True)
class FrozenPublicationAntigravity:
    publication_sha256: str
    prediction_sha256: str
    ranking_sha256: str
    source_audit_sha256: str
    selection: SelectionResultAntigravity

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or len(value) != 64 for value in (self.publication_sha256, self.prediction_sha256, self.ranking_sha256, self.source_audit_sha256)):
            raise ValueError("publication hash is invalid")
        if not isinstance(self.selection, SelectionResultAntigravity):
            raise ValueError("publication selection is invalid")


@dataclass(frozen=True)
class BootstrapResultAntigravity:
    micro_ci: tuple[float, float]
    macro_ci: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("micro_ci", "macro_ci"):
            values = tuple(getattr(self, name))
            if len(values) != 2:
                raise ValueError(f"{name} is invalid")
            normalized = tuple(_finite(value, name) for value in values)
            if normalized[0] > normalized[1]:
                raise ValueError(f"{name} is not ordered")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class EvaluationFactsAntigravity:
    baseline_confusion: tuple[int, int, int, int]
    union_confusion: tuple[int, int, int, int]
    candidate_tp: int
    candidate_fp: int
    rescued_datasets: int
    regressed_datasets: int
    maximum_strong_loss: float
    baseline_f1: float
    union_f1: float
    micro_delta: float
    macro_delta: float

    def __post_init__(self) -> None:
        for name in ("baseline_confusion", "union_confusion"):
            values = tuple(getattr(self, name))
            if len(values) != 4 or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        for name in ("candidate_tp", "candidate_fp", "rescued_datasets", "regressed_datasets"):
            _positive_int(getattr(self, name), name, allow_zero=True)
        for name in ("maximum_strong_loss", "baseline_f1", "union_f1", "micro_delta", "macro_delta"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class FrozenEvaluationAntigravity:
    evaluation_sha256: str
    status: str
    facts: EvaluationFactsAntigravity
    bootstrap: BootstrapResultAntigravity

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_sha256, str) or len(self.evaluation_sha256) != 64:
            raise ValueError("evaluation hash is invalid")
        if self.status not in {
            "engineering_reproduction_failed", "retrospective_reproduction_discrepancy",
            "retrospective_reproduction_complete",
        }:
            raise ValueError("evaluation status is invalid")
        if not isinstance(self.facts, EvaluationFactsAntigravity) or not isinstance(self.bootstrap, BootstrapResultAntigravity):
            raise ValueError("evaluation evidence is invalid")


__all__ = [
    "CandidateEvidenceAntigravity", "RankedCandidateAntigravity", "ParsedRecordAntigravity",
    "SelectionResultAntigravity", "validate_production_selection_antigravity",
    "FrozenSourceSnapshotAntigravity", "BackendAssessmentAntigravity", "ReadinessBindingAntigravity",
    "CompletedAttemptAntigravity", "FrozenPublicationAntigravity", "BootstrapResultAntigravity",
    "EvaluationFactsAntigravity", "FrozenEvaluationAntigravity",
]
