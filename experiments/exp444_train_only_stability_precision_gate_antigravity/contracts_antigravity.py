"""Immutable, label-free contracts for Exp444.

The prediction side of Exp444 only carries TRAIN-derived evidence and TEST
values/identities.  Recursive payload validation rejects labels and evaluation
fields before any artifact is serialized.  Every contract has strict
``to_payload``/``from_payload`` round trips so later tasks can use canonical
JSON without retaining mutable caller-owned containers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

import numpy as np

from .config_antigravity import (
    FROZEN_CONFIG,
    _validate_exp444_lifecycle_profile_antigravity,
    scientific_config_sha256_antigravity,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PREDICTION_TERMS = frozenset({
    "label", "labels", "labels_blob", "ground_truth", "truth", "target",
    "anomaly", "anomalies", "anomaly_position", "anomaly_positions", "actual",
    "evaluation", "confusion", "precision", "recall", "specificity",
    "false_positive", "false_negative", "fpr", "f1", "metric", "metrics",
    "family_test", "test_metric", "test_metrics", "tp", "fp", "fn", "tn",
})


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _name(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _tuple_ints(value: object, name: str, *, sorted_unique: bool = False) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_exact_int(item, name) for item in value)
    if sorted_unique and (result != tuple(sorted(result)) or len(result) != len(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _exact_payload(payload: Mapping[str, object], keys: set[str]) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise ValueError("payload keys are invalid")
    return dict(payload)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _freeze_json(value: object, *, prediction: bool = False) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return _finite(value, "JSON float")
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError("JSON key must be a non-empty string")
            if prediction:
                _validate_prediction_key(key)
            copied[key] = _freeze_json(item, prediction=prediction)
        return MappingProxyType(copied)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item, prediction=prediction) for item in value)
    raise ValueError("value is not deterministic JSON data")


def _validate_prediction_key(key: str) -> None:
    lowered = key.casefold().replace("-", "_")
    if any(term in lowered for term in _FORBIDDEN_PREDICTION_TERMS):
        raise ValueError(f"prediction payload contains forbidden key: {key}")


def validate_prediction_payload_antigravity(payload: object) -> None:
    """Recursively reject labels, metrics, and non-JSON values."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if type(key) is not str or not key:
                raise ValueError("prediction field name is invalid")
            _validate_prediction_key(key)
            validate_prediction_payload_antigravity(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            validate_prediction_payload_antigravity(value)
    elif payload is None or type(payload) in (str, int, bool):
        return
    elif type(payload) is float and math.isfinite(payload):
        return
    else:
        raise ValueError("prediction payload is not JSON-safe")


@dataclass(frozen=True)
class SourceCandidateAntigravity:
    dataset: str
    instance_index: int
    raw_length: int
    original_depths: tuple[int, ...]
    original_rank: int

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _exact_int(self.instance_index, "instance_index")
        _exact_int(self.raw_length, "raw_length", minimum=1)
        depths = _tuple_ints(self.original_depths, "original_depths", sorted_unique=True)
        if len(depths) < FROZEN_CONFIG.minimum_consensus_depths:
            raise ValueError("original_depths must contain at least two depths")
        _exact_int(self.original_rank, "original_rank", minimum=1)
        object.__setattr__(self, "original_depths", depths)

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.dataset, self.instance_index

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "instance_index": self.instance_index,
            "raw_length": self.raw_length,
            "original_depths": list(self.original_depths),
            "original_rank": self.original_rank,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SourceCandidateAntigravity":
        raw = _exact_payload(payload, {"dataset", "instance_index", "raw_length", "original_depths", "original_rank"})
        if not isinstance(raw["original_depths"], list):
            raise ValueError("original_depths must be a list")
        raw["original_depths"] = tuple(raw["original_depths"])
        return cls(**raw)


def _freeze_values_array_antigravity(value: object, name: str, *, ndim: int) -> np.ndarray:
    """Copy a numeric values matrix/vector onto immutable bytes."""
    array = np.asarray(value)
    if array.dtype != np.dtype("<f4") or array.ndim != ndim:
        raise ValueError(f"{name} must be a little-endian float32 array")
    if not np.isfinite(array).all() and name == "train_values":
        raise ValueError("train_values must contain only finite rows")
    contiguous = np.array(array, dtype="<f4", copy=True, order="C")
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f4").reshape(contiguous.shape)
    frozen.setflags(write=False)
    return frozen


def _selected_coordinate_identity_antigravity(candidates: Sequence[SourceCandidateAntigravity]) -> str:
    payload = b"".join(
        json.dumps(
            {"dataset": candidate.dataset, "instance_index": candidate.instance_index},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        for candidate in sorted(candidates, key=lambda item: item.coordinate)
    )
    return hashlib.sha256(payload).hexdigest()


def candidate_attestation_sha256_antigravity(
    candidates: Sequence[SourceCandidateAntigravity],
) -> str:
    """Attest the ordered, complete Exp442 candidate replay."""
    population = tuple(candidates)
    if any(not isinstance(item, SourceCandidateAntigravity) for item in population):
        raise ValueError("candidate attestation population is invalid")
    payload = b"".join(
        json.dumps(
            {
                "order": order,
                "dataset": candidate.dataset,
                "instance_index": candidate.instance_index,
                "raw_length": candidate.raw_length,
                "original_depths": list(candidate.original_depths),
                "original_rank": candidate.original_rank,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        for order, candidate in enumerate(population, start=1)
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CandidateInputAntigravity:
    """Copied, values-only input for one frozen Exp442 coordinate.

    ``train_values`` contains only finite normal TRAIN rows in the candidate's
    exact-length group.  ``test_values`` is the one bound TEST values row and
    intentionally carries no label or evaluation metadata.
    """

    candidate: SourceCandidateAntigravity
    train_indices: tuple[int, ...]
    train_values: np.ndarray
    test_values: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SourceCandidateAntigravity):
            raise ValueError("candidate input identity is invalid")
        indices = _tuple_ints(self.train_indices, "train_indices", sorted_unique=True)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("train_indices are invalid")
        train = _freeze_values_array_antigravity(self.train_values, "train_values", ndim=2)
        test = _freeze_values_array_antigravity(self.test_values, "test_values", ndim=1)
        if train.shape[0] != len(indices) or train.shape[1] != self.candidate.raw_length:
            raise ValueError("TRAIN values do not match candidate raw length")
        if test.shape[0] != self.candidate.raw_length:
            raise ValueError("TEST values do not match candidate raw length")
        object.__setattr__(self, "train_indices", indices)
        object.__setattr__(self, "train_values", train)
        object.__setattr__(self, "test_values", test)

    @property
    def dataset(self) -> str:
        return self.candidate.dataset

    @property
    def instance_index(self) -> int:
        return self.candidate.instance_index

    @property
    def raw_length(self) -> int:
        return self.candidate.raw_length

    @property
    def train_instance_indices(self) -> tuple[int, ...]:
        return self.train_indices


@dataclass(frozen=True)
class FrozenSourceSnapshotAntigravity:
    """Immutable source identity plus the parent-owned values session."""

    candidates: tuple[SourceCandidateAntigravity, ...]
    selected_coordinate_identity_sha256: str
    candidate_attestation_sha256: str
    source_hashes: Mapping[str, str]
    database_path: object
    label_access_count: int = 0
    session: object | None = None

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(item, SourceCandidateAntigravity) for item in candidates):
            raise ValueError("source candidates are invalid")
        if len(candidates) != FROZEN_CONFIG.source_candidate_count:
            raise ValueError("source snapshot must contain exactly 30 candidates")
        coordinates = tuple(item.coordinate for item in candidates)
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("source candidates contain duplicate coordinates")
        if len({item.dataset for item in candidates}) != len(candidates):
            raise ValueError("source snapshot must contain at most one candidate per dataset")
        expected_rank_order = tuple(range(1, FROZEN_CONFIG.source_candidate_count + 1))
        if tuple(item.original_rank for item in candidates) != expected_rank_order:
            raise ValueError("source candidate rank order must be exactly 1-30")
        selected_identity = _digest(self.selected_coordinate_identity_sha256, "selected_coordinate_identity_sha256")
        recomputed_identity = _selected_coordinate_identity_antigravity(candidates)
        if selected_identity != recomputed_identity or selected_identity != FROZEN_CONFIG.exp442_selected_coordinate_identity_sha256:
            raise ValueError("source snapshot coordinate identity is not the frozen selected identity")
        attestation = _digest(self.candidate_attestation_sha256, "candidate_attestation_sha256")
        if attestation != candidate_attestation_sha256_antigravity(candidates):
            raise ValueError("source snapshot candidate attestation is invalid")
        if not isinstance(self.source_hashes, Mapping) or not self.source_hashes:
            raise ValueError("source hashes are invalid")
        if any(type(key) is not str or not key for key in self.source_hashes):
            raise ValueError("source hash names are invalid")
        copied_hashes = {key: _digest(value, f"source hash {key}") for key, value in self.source_hashes.items()}
        if type(self.label_access_count) is not int or self.label_access_count != 0:
            raise ValueError("source verification accessed labels")
        if not isinstance(self.database_path, (str, bytes)) and not hasattr(self.database_path, "__fspath__"):
            raise ValueError("database path is invalid")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "source_hashes", MappingProxyType(copied_hashes))

    @property
    def database_sha256(self) -> str:
        try:
            return self.source_hashes["database"]
        except KeyError as error:
            raise ValueError("database source hash is unavailable") from error

    @property
    def selected_coordinates(self) -> tuple[tuple[str, int], ...]:
        return tuple(item.coordinate for item in self.candidates)


@dataclass(frozen=True)
class OmissionMaskAntigravity:
    view_index: int
    excluded_train_indices: tuple[int, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        _exact_int(self.view_index, "view_index")
        excluded = _tuple_ints(self.excluded_train_indices, "excluded_train_indices", sorted_unique=True)
        _digest(self.identity_sha256, "identity_sha256")
        object.__setattr__(self, "excluded_train_indices", excluded)

    def to_payload(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "excluded_train_indices": list(self.excluded_train_indices),
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "OmissionMaskAntigravity":
        raw = _exact_payload(payload, {"view_index", "excluded_train_indices", "identity_sha256"})
        if not isinstance(raw["excluded_train_indices"], list):
            raise ValueError("excluded_train_indices must be a list")
        raw["excluded_train_indices"] = tuple(raw["excluded_train_indices"])
        return cls(**raw)


@dataclass(frozen=True)
class DepthViewScoreAntigravity:
    depth: int
    score: float
    threshold: float
    exceeds: bool
    excess_ratio: float
    empirical_rank: float

    def __post_init__(self) -> None:
        _exact_int(self.depth, "depth", minimum=1)
        score = _finite(self.score, "score", minimum=0.0)
        threshold = _finite(self.threshold, "threshold", minimum=0.0)
        ratio = _finite(self.excess_ratio, "excess_ratio", minimum=0.0)
        rank = _finite(self.empirical_rank, "empirical_rank", minimum=0.0)
        if not 0.0 < rank <= 1.0:
            raise ValueError("empirical_rank must be in (0, 1]")
        if type(self.exceeds) is not bool or self.exceeds != (score > threshold):
            raise ValueError("exceeds is inconsistent with score and threshold")
        expected_ratio = score / max(threshold, FROZEN_CONFIG.robust_scale_floor)
        if not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("excess_ratio is inconsistent")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "excess_ratio", ratio)
        object.__setattr__(self, "empirical_rank", rank)

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DepthViewScoreAntigravity":
        return cls(**_exact_payload(payload, set(cls.__dataclass_fields__)))


@dataclass(frozen=True)
class StabilityViewAntigravity:
    mask: OmissionMaskAntigravity
    retained_train_indices: tuple[int, ...]
    depth_scores: tuple[DepthViewScoreAntigravity, ...]
    agreeing_depths: tuple[int, ...]
    survived: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mask, OmissionMaskAntigravity):
            raise ValueError("view mask is invalid")
        retained = _tuple_ints(self.retained_train_indices, "retained_train_indices", sorted_unique=True)
        if set(retained) & set(self.mask.excluded_train_indices):
            raise ValueError("retained and excluded TRAIN indices overlap")
        scores = tuple(self.depth_scores)
        if any(not isinstance(item, DepthViewScoreAntigravity) for item in scores):
            raise ValueError("view depth scores are invalid")
        agreeing = _tuple_ints(self.agreeing_depths, "agreeing_depths", sorted_unique=True)
        if not scores:
            if agreeing or self.survived:
                raise ValueError("empty view requires no agreeing depths and survived=False")
            depths = ()
        else:
            depths = tuple(item.depth for item in scores)
            if depths != tuple(sorted(set(depths))):
                raise ValueError("view depth scores must be sorted and unique")
        if any(depth not in depths for depth in agreeing):
            raise ValueError("agreeing depth is not present in view scores")
        if type(self.survived) is not bool:
            raise ValueError("survived must be a bool")
        expected_survival = len(agreeing) >= FROZEN_CONFIG.minimum_consensus_depths and all(
            next(score for score in scores if score.depth == depth).exceeds for depth in agreeing
        )
        if self.survived != expected_survival:
            raise ValueError("survived is inconsistent with agreeing depth evidence")
        object.__setattr__(self, "retained_train_indices", retained)
        object.__setattr__(self, "depth_scores", scores)
        object.__setattr__(self, "agreeing_depths", agreeing)

    @property
    def view_index(self) -> int:
        return self.mask.view_index

    def to_payload(self) -> dict[str, object]:
        return {
            "mask": self.mask.to_payload(),
            "retained_train_indices": list(self.retained_train_indices),
            "depth_scores": [score.to_payload() for score in self.depth_scores],
            "agreeing_depths": list(self.agreeing_depths),
            "survived": self.survived,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "StabilityViewAntigravity":
        raw = _exact_payload(payload, {"mask", "retained_train_indices", "depth_scores", "agreeing_depths", "survived"})
        if not isinstance(raw["mask"], Mapping) or not isinstance(raw["retained_train_indices"], list) or not isinstance(raw["depth_scores"], list) or not isinstance(raw["agreeing_depths"], list):
            raise ValueError("stability view payload is invalid")
        raw["mask"] = OmissionMaskAntigravity.from_payload(raw["mask"])
        raw["retained_train_indices"] = tuple(raw["retained_train_indices"])
        raw["depth_scores"] = tuple(DepthViewScoreAntigravity.from_payload(item) for item in raw["depth_scores"])
        raw["agreeing_depths"] = tuple(raw["agreeing_depths"])
        return cls(**raw)


@dataclass(frozen=True)
class CandidateStabilityResultAntigravity:
    candidate: SourceCandidateAntigravity
    views: tuple[StabilityViewAntigravity, ...]
    survival_count: int
    required_survival_count: int
    lane: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SourceCandidateAntigravity):
            raise ValueError("candidate is invalid")
        views = tuple(self.views)
        if not views or any(not isinstance(view, StabilityViewAntigravity) for view in views):
            raise ValueError("stability views are invalid")
        if tuple(view.view_index for view in views) != tuple(sorted({view.view_index for view in views})):
            raise ValueError("view indices must be sorted and unique")
        survival = _exact_int(self.survival_count, "survival_count")
        required = _exact_int(self.required_survival_count, "required_survival_count", minimum=1)
        if required > len(views) or survival > len(views):
            raise ValueError("survival counts exceed view count")
        expected_survival = sum(view.survived for view in views)
        if survival != expected_survival:
            raise ValueError("survival_count is inconsistent with views")
        if self.lane not in {"stable_review", "unstable_review"}:
            raise ValueError("lane must be stable_review or unstable_review")
        expected_lane = "stable_review" if survival >= required else "unstable_review"
        if self.lane != expected_lane:
            raise ValueError("lane is inconsistent with survival threshold")
        object.__setattr__(self, "views", views)

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.candidate.coordinate

    @property
    def source_count(self) -> int:
        return 1

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_payload(),
            "views": [view.to_payload() for view in self.views],
            "survival_count": self.survival_count,
            "required_survival_count": self.required_survival_count,
            "lane": self.lane,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CandidateStabilityResultAntigravity":
        raw = _exact_payload(payload, {"candidate", "views", "survival_count", "required_survival_count", "lane"})
        if not isinstance(raw["candidate"], Mapping) or not isinstance(raw["views"], list):
            raise ValueError("candidate stability payload is invalid")
        raw["candidate"] = SourceCandidateAntigravity.from_payload(raw["candidate"])
        raw["views"] = tuple(StabilityViewAntigravity.from_payload(item) for item in raw["views"])
        return cls(**raw)


@dataclass(frozen=True)
class StabilitySelectionAntigravity:
    source: tuple[SourceCandidateAntigravity, ...]
    stable: tuple[CandidateStabilityResultAntigravity, ...]
    unstable: tuple[CandidateStabilityResultAntigravity, ...]

    def __post_init__(self) -> None:
        source = tuple(self.source)
        stable = tuple(self.stable)
        unstable = tuple(self.unstable)
        if any(not isinstance(item, SourceCandidateAntigravity) for item in source):
            raise ValueError("source candidates are invalid")
        if any(not isinstance(item, CandidateStabilityResultAntigravity) for item in stable + unstable):
            raise ValueError("stability lane result is invalid")
        source_coordinates = tuple(item.coordinate for item in source)
        lane_coordinates = tuple(item.coordinate for item in stable + unstable)
        if len(source_coordinates) != len(set(source_coordinates)) or len(lane_coordinates) != len(set(lane_coordinates)):
            raise ValueError("duplicate candidate coordinate")
        if set(source_coordinates) != set(lane_coordinates):
            raise ValueError("stable and unstable lanes must conserve source coordinates")
        if any(item.lane != "stable_review" for item in stable) or any(item.lane != "unstable_review" for item in unstable):
            raise ValueError("lane result placed in the wrong review lane")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "stable", stable)
        object.__setattr__(self, "unstable", unstable)

    @property
    def source_count(self) -> int:
        return len(self.source)

    @property
    def stable_count(self) -> int:
        return len(self.stable)

    @property
    def unstable_count(self) -> int:
        return len(self.unstable)

    def to_payload(self) -> dict[str, object]:
        payload = {
            "source": [item.to_payload() for item in self.source],
            "stable": [item.to_payload() for item in self.stable],
            "unstable": [item.to_payload() for item in self.unstable],
        }
        validate_prediction_payload_antigravity(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "StabilitySelectionAntigravity":
        raw = _exact_payload(payload, {"source", "stable", "unstable"})
        if not all(isinstance(raw[name], list) for name in ("source", "stable", "unstable")):
            raise ValueError("selection payload lanes must be lists")
        raw["source"] = tuple(SourceCandidateAntigravity.from_payload(item) for item in raw["source"])
        raw["stable"] = tuple(CandidateStabilityResultAntigravity.from_payload(item) for item in raw["stable"])
        raw["unstable"] = tuple(CandidateStabilityResultAntigravity.from_payload(item) for item in raw["unstable"])
        return cls(**raw)


@dataclass(frozen=True)
class BackendDecisionAntigravity:
    backend: str
    workers: int
    evidence: tuple[Mapping[str, object], ...] = ()
    fallback_reason: str | None = "cpu_fallback_until_parity_speed_gate"

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "mlx", "numba"}:
            raise ValueError("backend is invalid")
        workers = _exact_int(self.workers, "workers", minimum=1)
        evidence = tuple(_freeze_json(item) for item in self.evidence)
        if any(not isinstance(item, Mapping) for item in evidence):
            raise ValueError("backend evidence must be mappings")
        if self.backend == "cpu":
            if type(self.fallback_reason) is not str or not self.fallback_reason:
                raise ValueError("CPU backend requires a fallback reason")
        elif self.fallback_reason is not None:
            raise ValueError("accelerated backend cannot carry fallback reason")
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "evidence", evidence)

    @property
    def selected_backend(self) -> str:
        return self.backend

    def to_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "workers": self.workers,
            "evidence": [_json_safe(item) for item in self.evidence],
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "BackendDecisionAntigravity":
        raw = _exact_payload(payload, {"backend", "workers", "evidence", "fallback_reason"})
        if not isinstance(raw["evidence"], list):
            raise ValueError("backend evidence must be a list")
        raw["evidence"] = tuple(raw["evidence"])
        return cls(**raw)


@dataclass(frozen=True)
class CheckpointContextAntigravity:
    experiment_id: str
    run_identity: str
    scientific_config_sha256: str
    environment_lock_sha256: str
    total_units: int
    outer_workers: int
    inner_threads: int
    reviewed_commit: str

    def __post_init__(self) -> None:
        _validate_exp444_lifecycle_profile_antigravity()
        if self.experiment_id != FROZEN_CONFIG.experiment_id or self.run_identity != FROZEN_CONFIG.run_identity:
            raise ValueError("checkpoint identity is not the frozen Exp444 identity")
        if self.scientific_config_sha256 != scientific_config_sha256_antigravity() or self.environment_lock_sha256 != FROZEN_CONFIG.environment_lock_sha256:
            raise ValueError("checkpoint hash does not match frozen Exp444 identity")
        if self.total_units != FROZEN_CONFIG.source_candidate_count or self.outer_workers != FROZEN_CONFIG.cpu_workers or self.inner_threads != FROZEN_CONFIG.inner_threads:
            raise ValueError("checkpoint topology is invalid")
        if type(self.reviewed_commit) is not str or len(self.reviewed_commit) != 40 or any(character not in "0123456789abcdef" for character in self.reviewed_commit):
            raise ValueError("checkpoint reviewed commit is invalid")

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CheckpointContextAntigravity":
        return cls(**_exact_payload(payload, set(cls.__dataclass_fields__)))

    def validate_antigravity(self) -> bool:
        self.__post_init__()
        return True


@dataclass(frozen=True)
class CheckpointStateAntigravity:
    status: str
    completed_units: int
    total_units: int
    error_count: int
    current_unit: str | None
    current_view: int | None
    payload_sha256: str

    def __post_init__(self) -> None:
        if self.status not in {"running", "interrupted", "complete", "failed"}:
            raise ValueError("checkpoint state status is invalid")
        completed = _exact_int(self.completed_units, "completed_units")
        total = _exact_int(self.total_units, "total_units", minimum=1)
        _exact_int(self.error_count, "error_count")
        if completed > total:
            raise ValueError("completed_units exceed total_units")
        if self.current_unit is not None:
            _name(self.current_unit, "current_unit")
        if self.current_view is not None:
            _exact_int(self.current_view, "current_view")
        _digest(self.payload_sha256, "payload_sha256")

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CheckpointStateAntigravity":
        return cls(**_exact_payload(payload, set(cls.__dataclass_fields__)))


@dataclass(frozen=True)
class EvaluationSummaryAntigravity:
    evaluation_status: str
    source_count: int
    stable_count: int
    unstable_count: int
    stable_tp: int
    stable_fp: int
    stable_micro_f1: float
    stable_macro_f1: float
    evaluation_sha256: str
    unstable_tp: int = 0
    unstable_fp: int = 0

    def __post_init__(self) -> None:
        if self.evaluation_status != FROZEN_CONFIG.evaluation_status:
            raise ValueError("evaluation status is not descriptive-only")
        source = _exact_int(self.source_count, "source_count")
        stable = _exact_int(self.stable_count, "stable_count")
        unstable = _exact_int(self.unstable_count, "unstable_count")
        if stable + unstable != source:
            raise ValueError("evaluation lane counts do not conserve source")
        for name in ("stable_tp", "stable_fp", "unstable_tp", "unstable_fp"):
            _exact_int(getattr(self, name), name)
        _finite(self.stable_micro_f1, "stable_micro_f1", minimum=0.0)
        _finite(self.stable_macro_f1, "stable_macro_f1", minimum=0.0)
        _digest(self.evaluation_sha256, "evaluation_sha256")

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EvaluationSummaryAntigravity":
        return cls(**_exact_payload(payload, set(cls.__dataclass_fields__)))


def canonical_contract_sha256_antigravity(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SourceCandidateAntigravity", "CandidateInputAntigravity", "FrozenSourceSnapshotAntigravity",
    "candidate_attestation_sha256_antigravity",
    "OmissionMaskAntigravity", "DepthViewScoreAntigravity",
    "StabilityViewAntigravity", "CandidateStabilityResultAntigravity",
    "StabilitySelectionAntigravity", "BackendDecisionAntigravity",
    "CheckpointContextAntigravity", "CheckpointStateAntigravity",
    "EvaluationSummaryAntigravity", "canonical_contract_sha256_antigravity",
    "validate_prediction_payload_antigravity",
]
