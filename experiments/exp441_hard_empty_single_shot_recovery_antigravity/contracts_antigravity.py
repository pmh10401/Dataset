"""Strict immutable evidence contracts for the label-free Exp441 replay."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


_TERMINAL_ATTEMPT_AUTHORITY_FILENAMES_ANTIGRAVITY = (
    "source_snapshot_antigravity.json", "selection_antigravity.json", "progress_antigravity.json",
    "pid_antigravity", "events_antigravity.jsonl", "text_antigravity.log",
)


def terminal_attempt_authority_sha256_from_bytes_antigravity(authority: Mapping[str, bytes]) -> str:
    """Return the deterministic identity of Exp441's six terminal evidence files."""
    if not isinstance(authority, Mapping) or set(authority) != set(_TERMINAL_ATTEMPT_AUTHORITY_FILENAMES_ANTIGRAVITY):
        raise ValueError("terminal attempt authority is invalid")
    digest = sha256()
    for name in _TERMINAL_ATTEMPT_AUTHORITY_FILENAMES_ANTIGRAVITY:
        payload = authority[name]
        if not isinstance(payload, bytes):
            raise ValueError("terminal attempt authority is invalid")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


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


def _freeze_json_safe(value: object) -> object:
    """Detach recursively while retaining only canonical JSON data types."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return _finite(value, "environment float")
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("environment key is invalid")
            copied[key] = _freeze_json_safe(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_safe(item) for item in value)
    raise ValueError("environment value is not JSON-safe")


def _json_safe_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe_payload(item) for item in value]
    return value


def _require_unique_coordinates(items: Iterable[CandidateEvidenceAntigravity | RankedCandidateAntigravity], name: str) -> None:
    coordinates = [item.coordinate for item in items]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"{name} coordinates are duplicated")


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
        _require_unique_coordinates(provisional, "provisional")
        _require_unique_coordinates(outside, "outside-Hard")
        _require_unique_coordinates(eligible, "eligible")
        _require_unique_coordinates(winners, "winner")
        _require_unique_coordinates(selected, "selected")
        expected_outside = tuple(item for item in provisional if item.coordinate not in hard)
        if outside != expected_outside:
            raise ValueError("outside-Hard population is invalid")
        hard_datasets = {dataset for dataset, _ in hard}
        expected_eligible = tuple(item for item in outside if item.dataset not in hard_datasets)
        if eligible != expected_eligible:
            raise ValueError("Hard-empty eligibility population is invalid")
        winner_candidates = tuple(item.candidate for item in winners)
        if any(item.dataset in hard_datasets for item in winner_candidates + tuple(item.candidate for item in selected)):
            raise ValueError("Hard-active dataset was selected")
        grouped: dict[str, list[CandidateEvidenceAntigravity]] = {}
        for candidate in eligible:
            grouped.setdefault(candidate.dataset, []).append(candidate)
        expected_winners = tuple(sorted(
            (min(items, key=lambda item: item.dataset_key) for items in grouped.values()),
            key=lambda item: item.global_key,
        ))
        if winner_candidates != expected_winners:
            raise ValueError("winner population is invalid")
        winner_datasets = [item.dataset for item in winners]
        if len(winner_datasets) != len(set(winner_datasets)):
            raise ValueError("more than one winner per dataset")
        if tuple(item.global_rank for item in winners) != tuple(range(1, len(winners) + 1)):
            raise ValueError("global ranks are invalid")
        if len(selected) > 30 or selected != winners[:len(selected)]:
            raise ValueError("selected population is invalid")
        if len({item.dataset for item in selected}) != len(selected):
            raise ValueError("selected coordinates are duplicated")
        if tuple(item.selection_rank for item in selected) != tuple(range(1, len(selected) + 1)):
            raise ValueError("selection ranks are invalid")
        if any(not item.selected for item in selected):
            raise ValueError("selected records are not marked selected")
        if any(item.selected != (index <= len(selected)) for index, item in enumerate(winners, start=1)):
            raise ValueError("winner selected flags are invalid")
        if any(item.selection_rank != (index if item.selected else None) for index, item in enumerate(winners, start=1)):
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
        if not isinstance(self.environment, Mapping):
            raise ValueError("backend environment is invalid")
        object.__setattr__(self, "environment", _freeze_json_safe(self.environment))

    def to_payload(self) -> dict[str, object]:
        return {
            "canonical_backend": self.canonical_backend,
            "workers": self.workers,
            "tensor_eligible": self.tensor_eligible,
            "acceleration_benchmark_status": self.acceleration_benchmark_status,
            "pytorch": self.pytorch,
            "mlx": self.mlx,
            "mps": self.mps,
            "environment": _json_safe_payload(self.environment),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "BackendAssessmentAntigravity":
        raw = _exact_payload(payload, {
            "canonical_backend", "workers", "tensor_eligible", "acceleration_benchmark_status",
            "pytorch", "mlx", "mps", "environment",
        })
        return cls(**raw)


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
    terminal_authority_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.attempt_sequence, "attempt sequence", allow_zero=True)
        if (not isinstance(self.reviewed_commit, str) or len(self.reviewed_commit) != 40 or
                any(not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in (self.source_snapshot_sha256, self.selection_sha256, self.terminal_authority_sha256))):
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


def _finite_builtin_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


_RETROSPECTIVE_EXPECTED_BASELINE_CONFUSION_ANTIGRAVITY = (1915, 303, 2003, 191679)
_RETROSPECTIVE_EXPECTED_UNION_CONFUSION_ANTIGRAVITY = (1930, 318, 1988, 191664)
_RETROSPECTIVE_EXPECTED_CANDIDATE_COUNTS_ANTIGRAVITY = (15, 15)
_RETROSPECTIVE_EXPECTED_RESCUED_REGRESSED_ANTIGRAVITY = (15, 0)
_RETROSPECTIVE_EXPECTED_EXACT_FLOATS_ANTIGRAVITY = (0.0, 0.6241851368970013, 0.6260136230943887)
_RETROSPECTIVE_EXPECTED_TOLERANT_FLOATS_ANTIGRAVITY = (
    0.0018284861973874023, 0.012444046553267682,
    0.00005373870255999115, 0.003814109586073529,
    0.006535362578334826, 0.019158460161145925,
)


def _retrospective_project_f1_antigravity(confusion: tuple[int, int, int, int]) -> float:
    tp, fp, fn, _tn = (float(value) for value in confusion)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float((2.0 * precision * recall) / (precision + recall)) if precision + recall else 0.0


def retrospective_status_from_evidence_antigravity(
    facts: "EvaluationFactsAntigravity", bootstrap: "BootstrapResultAntigravity",
) -> str:
    """Classify immutable aggregate and bootstrap evidence without rerunning bootstrap."""
    if not isinstance(facts, EvaluationFactsAntigravity) or not isinstance(bootstrap, BootstrapResultAntigravity):
        return "engineering_reproduction_failed"
    exact = (
        facts.baseline_confusion == _RETROSPECTIVE_EXPECTED_BASELINE_CONFUSION_ANTIGRAVITY,
        facts.union_confusion == _RETROSPECTIVE_EXPECTED_UNION_CONFUSION_ANTIGRAVITY,
        (facts.candidate_tp, facts.candidate_fp) == _RETROSPECTIVE_EXPECTED_CANDIDATE_COUNTS_ANTIGRAVITY,
        (facts.rescued_datasets, facts.regressed_datasets) == _RETROSPECTIVE_EXPECTED_RESCUED_REGRESSED_ANTIGRAVITY,
        (facts.maximum_strong_loss, facts.baseline_f1, facts.union_f1) == _RETROSPECTIVE_EXPECTED_EXACT_FLOATS_ANTIGRAVITY,
    )
    values = (
        facts.maximum_strong_loss, facts.baseline_f1, facts.union_f1, facts.micro_delta, facts.macro_delta,
        *bootstrap.micro_ci, *bootstrap.macro_ci,
    )
    if not all(type(value) is float and math.isfinite(value) for value in values) or not all(exact):
        return "engineering_reproduction_failed"
    observed = (facts.micro_delta, facts.macro_delta, *bootstrap.micro_ci, *bootstrap.macro_ci)
    if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
           for actual, expected in zip(observed, _RETROSPECTIVE_EXPECTED_TOLERANT_FLOATS_ANTIGRAVITY, strict=True)):
        return "retrospective_reproduction_discrepancy"
    return "retrospective_reproduction_complete"


def _evaluation_confusion_antigravity(value: object, name: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} is invalid") from error
    if len(values) != 4 or any(type(item) is not int or item < 0 for item in values):
        raise ValueError(f"{name} is invalid")
    return values  # type: ignore[return-value]


@dataclass(frozen=True)
class DatasetEvaluationFactsAntigravity:
    """Detached retrospective facts for exactly one context dataset."""

    dataset: str
    baseline_confusion: tuple[int, int, int, int]
    union_confusion: tuple[int, int, int, int]
    baseline_f1: float
    union_f1: float
    delta: float
    candidate_tp: int
    candidate_fp: int

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("dataset evaluation dataset is invalid")
        object.__setattr__(self, "baseline_confusion", _evaluation_confusion_antigravity(self.baseline_confusion, "baseline confusion"))
        object.__setattr__(self, "union_confusion", _evaluation_confusion_antigravity(self.union_confusion, "union confusion"))
        for name in ("baseline_f1", "union_f1", "delta"):
            object.__setattr__(self, name, _finite_builtin_float_antigravity(getattr(self, name), name))
        for name in ("candidate_tp", "candidate_fp"):
            _positive_int(getattr(self, name), name, allow_zero=True)
        baseline_f1 = _retrospective_project_f1_antigravity(self.baseline_confusion)
        union_f1 = _retrospective_project_f1_antigravity(self.union_confusion)
        if (self.baseline_f1 != baseline_f1 or self.union_f1 != union_f1
                or self.delta != union_f1 - baseline_f1):
            raise ValueError("dataset evaluation metrics are inconsistent")


@dataclass(frozen=True)
class RetrospectiveEvaluationAntigravity:
    """The complete immutable in-memory Exp441 retrospective calculation."""

    dataset_facts: tuple[DatasetEvaluationFactsAntigravity, ...]
    rescued_datasets: tuple[str, ...]
    regressed_datasets: tuple[str, ...]
    bootstrap_rows: tuple[tuple[int, int, int, int, int, int, int, int, float], ...]
    facts: EvaluationFactsAntigravity
    bootstrap: BootstrapResultAntigravity
    status: str
    scientific_winner_selected: bool

    def __post_init__(self) -> None:
        dataset_facts = tuple(self.dataset_facts)
        if len(dataset_facts) != 1117 or any(type(item) is not DatasetEvaluationFactsAntigravity for item in dataset_facts):
            raise ValueError("dataset evaluation facts are invalid")
        datasets = tuple(item.dataset for item in dataset_facts)
        if len(datasets) != len(set(datasets)):
            raise ValueError("dataset evaluation facts are duplicated")
        rescued = tuple(self.rescued_datasets)
        regressed = tuple(self.regressed_datasets)
        if (any(type(item) is not str or not item for item in rescued + regressed)
                or len(rescued) != len(set(rescued)) or len(regressed) != len(set(regressed))
                or any(item not in datasets for item in rescued + regressed)):
            raise ValueError("retrospective dataset names are invalid")
        copied_rows: list[tuple[int, int, int, int, int, int, int, int, float]] = []
        try:
            rows = tuple(self.bootstrap_rows)
        except TypeError as error:
            raise ValueError("bootstrap rows are invalid") from error
        if len(rows) != 1117:
            raise ValueError("bootstrap rows are invalid")
        for row in rows:
            try:
                values = tuple(row)
            except TypeError as error:
                raise ValueError("bootstrap row is invalid") from error
            if (len(values) != 9
                    or any(type(value) is not int or value < 0 for value in values[:8])):
                raise ValueError("bootstrap row is invalid")
            copied_rows.append((*values[:8], _finite_builtin_float_antigravity(values[8], "bootstrap delta")))  # type: ignore[arg-type]
        if not isinstance(self.facts, EvaluationFactsAntigravity) or not isinstance(self.bootstrap, BootstrapResultAntigravity):
            raise ValueError("retrospective evidence is invalid")
        derived_rescued = tuple(item.dataset for item in dataset_facts
                                if item.baseline_confusion[0] == 0 and item.union_confusion[0] >= 1)
        derived_regressed = tuple(item.dataset for item in dataset_facts
                                  if item.union_f1 < item.baseline_f1)
        if rescued != derived_rescued or regressed != derived_regressed:
            raise ValueError("retrospective dataset classifications are inconsistent")
        expected_rows = tuple(
            (*item.baseline_confusion, *item.union_confusion, item.delta) for item in dataset_facts
        )
        if tuple(copied_rows) != expected_rows:
            raise ValueError("bootstrap rows are inconsistent")
        baseline = tuple(sum(item.baseline_confusion[index] for item in dataset_facts) for index in range(4))
        union = tuple(sum(item.union_confusion[index] for item in dataset_facts) for index in range(4))
        candidate_tp = sum(item.candidate_tp for item in dataset_facts)
        candidate_fp = sum(item.candidate_fp for item in dataset_facts)
        baseline_f1 = _retrospective_project_f1_antigravity(baseline)
        union_f1 = _retrospective_project_f1_antigravity(union)
        micro_delta = union_f1 - baseline_f1
        macro_delta = float(np.asarray([item.delta for item in dataset_facts], dtype=np.float64).mean(dtype=np.float64))
        maximum_strong_loss = max((
            item.baseline_f1 - item.union_f1
            for item in dataset_facts
            if item.union_f1 < item.baseline_f1 and item.baseline_f1 >= 0.8
        ), default=0.0)
        if (self.facts.baseline_confusion != baseline or self.facts.union_confusion != union
                or self.facts.candidate_tp != candidate_tp or self.facts.candidate_fp != candidate_fp
                or self.facts.rescued_datasets != len(rescued) or self.facts.regressed_datasets != len(regressed)
                or self.facts.maximum_strong_loss != maximum_strong_loss
                or self.facts.baseline_f1 != baseline_f1 or self.facts.union_f1 != union_f1
                or self.facts.micro_delta != micro_delta or self.facts.macro_delta != macro_delta):
            raise ValueError("retrospective aggregate is inconsistent")
        if self.status != retrospective_status_from_evidence_antigravity(self.facts, self.bootstrap):
            raise ValueError("retrospective status is inconsistent")
        if type(self.scientific_winner_selected) is not bool or self.scientific_winner_selected is not False:
            raise ValueError("scientific winner must remain false")
        object.__setattr__(self, "dataset_facts", dataset_facts)
        object.__setattr__(self, "rescued_datasets", rescued)
        object.__setattr__(self, "regressed_datasets", regressed)
        object.__setattr__(self, "bootstrap_rows", tuple(copied_rows))


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
    "terminal_attempt_authority_sha256_from_bytes_antigravity",
    "EvaluationFactsAntigravity", "DatasetEvaluationFactsAntigravity",
    "RetrospectiveEvaluationAntigravity", "FrozenEvaluationAntigravity",
]
