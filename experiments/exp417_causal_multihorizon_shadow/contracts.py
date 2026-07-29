"""Validated immutable domain records for Exp417's label-free workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from .config import Exp417FrozenConfig, FROZEN_CONFIG


_PRIMARY_PERTURBATIONS = frozenset(
    {"level_shift", "gradual_drift", "amplitude_scaling", "period_change"}
)
_GROUP_STATUSES = frozenset({"qualified", "diagnostic", "abstained", "failed"})
_DATASET_STATUSES = frozenset({"qualified", "abstained", "failed"})
_PROCESSED_GROUP_STATUSES = frozenset(
    {
        "candidate_eligible",
        "diagnostic_only",
        "insufficient_train_abstain",
        "horizon_abstain",
        "nonfinite_abstain",
        "zero_scale_abstain",
    }
)
_T = TypeVar("_T")


def _nonempty_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _raw_number(value: object, name: str) -> float:
    """Keep raw numeric values intact until TRAIN-only deterministic repair."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _tuple(values: Sequence[_T], name: str) -> tuple[_T, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return tuple(values)


def _immutable_mapping(values: Mapping[str, _T], name: str) -> Mapping[str, _T]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class RawSeries:
    instance_index: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _integer(self.instance_index, "instance_index")
        values = _tuple(self.values, "values")
        if not values:
            raise ValueError("values must not be empty")
        object.__setattr__(self, "values", tuple(_raw_number(value, "series value") for value in values))


def _freeze_lane_mapping(values: Mapping[str, Sequence[int]], name: str) -> Mapping[str, tuple[int, ...]]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    frozen: dict[str, tuple[int, ...]] = {}
    for dataset, coordinates in values.items():
        _nonempty_name(dataset, f"{name} dataset")
        copied = _tuple(coordinates, f"{name} coordinates")
        checked = tuple(_integer(index, f"{name} coordinate") for index in copied)
        if checked != tuple(sorted(checked)):
            raise ValueError(f"{name} coordinates must be sorted")
        if len(checked) != len(set(checked)):
            raise ValueError(f"{name} coordinates contain a duplicate")
        frozen[dataset] = checked
    return MappingProxyType(dict(sorted(frozen.items())))


@dataclass(frozen=True)
class FrozenLaneSnapshot:
    hard: Mapping[str, tuple[int, ...]]
    standard: Mapping[str, tuple[int, ...]]
    priority: Mapping[str, tuple[int, ...]]
    automatic_normal: Mapping[str, tuple[int, ...]]
    exp291_prediction_sha256: str = FROZEN_CONFIG.exp291_prediction_sha256
    exp291_manifest_sha256: str = FROZEN_CONFIG.exp291_manifest_sha256
    exp292_prediction_sha256: str = FROZEN_CONFIG.exp292_prediction_sha256
    exp292_manifest_sha256: str = FROZEN_CONFIG.exp292_manifest_sha256
    config: Exp417FrozenConfig = FROZEN_CONFIG

    def __post_init__(self) -> None:
        if not isinstance(self.config, Exp417FrozenConfig):
            raise ValueError("config must be an Exp417 frozen config")
        self.config.validate()
        for name in ("hard", "standard", "priority", "automatic_normal"):
            object.__setattr__(self, name, _freeze_lane_mapping(getattr(self, name), name))
        seen: set[tuple[str, int]] = set()
        for lane in (self.hard, self.standard, self.priority, self.automatic_normal):
            coordinates = {(dataset, index) for dataset, indices in lane.items() for index in indices}
            if seen & coordinates:
                raise ValueError("frozen lanes overlap")
            seen.update(coordinates)
        expected_totals = {
            "hard": self.config.expected_hard,
            "standard": self.config.expected_standard,
            "priority": self.config.expected_priority,
            "automatic_normal": self.config.expected_automatic_normal,
        }
        for name, expected_total in expected_totals.items():
            actual_total = sum(len(indices) for indices in getattr(self, name).values())
            if actual_total != expected_total:
                raise ValueError(f"frozen lane total mismatch: {name}")
        expected_hashes = {
            "exp291_prediction_sha256": self.config.exp291_prediction_sha256,
            "exp291_manifest_sha256": self.config.exp291_manifest_sha256,
            "exp292_prediction_sha256": self.config.exp292_prediction_sha256,
            "exp292_manifest_sha256": self.config.exp292_manifest_sha256,
        }
        for name, expected_hash in expected_hashes.items():
            if getattr(self, name) != expected_hash:
                raise ValueError(f"frozen {name.split('_')[0].capitalize()} artifact hash mismatch")

    def contains(self, dataset: str, instance_index: int) -> bool:
        return any(instance_index in lane.get(dataset, ()) for lane in (self.hard, self.standard, self.priority, self.automatic_normal))


@dataclass(frozen=True)
class RawTrainDataset:
    dataset: str
    series: tuple[RawSeries, ...]

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, "dataset")
        series = _tuple(self.series, "series")
        if not series or any(not isinstance(item, RawSeries) for item in series):
            raise ValueError("series must contain RawSeries records")
        _validate_unique_series_indices(series)
        object.__setattr__(self, "series", series)


@dataclass(frozen=True)
class RawTestDataset:
    dataset: str
    series: tuple[RawSeries, ...]

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, "dataset")
        series = _tuple(self.series, "series")
        if not series or any(not isinstance(item, RawSeries) for item in series):
            raise ValueError("series must contain RawSeries records")
        _validate_unique_series_indices(series)
        if any(not isfinite(value) for item in series for value in item.values):
            raise ValueError("TEST raw values must be finite")
        object.__setattr__(self, "series", series)


def _validate_unique_series_indices(series: tuple[RawSeries, ...]) -> None:
    indices = tuple(item.instance_index for item in series)
    if indices != tuple(sorted(indices)):
        raise ValueError("series indices must be sorted")
    if len(indices) != len(set(indices)):
        raise ValueError("series indices contain a duplicate")


@dataclass(frozen=True)
class FrozenRunContext:
    database_sha256: str
    datasets: tuple[str, ...]
    lanes: FrozenLaneSnapshot
    test_sizes: Mapping[str, int]
    code_commit: str
    runtime_versions: Mapping[str, str]
    config: Exp417FrozenConfig = FROZEN_CONFIG

    def __post_init__(self) -> None:
        _nonempty_name(self.database_sha256, "database_sha256")
        datasets = _tuple(self.datasets, "datasets")
        if not datasets or any(not isinstance(value, str) or not value.strip() for value in datasets):
            raise ValueError("datasets must contain non-empty names")
        if datasets != tuple(sorted(datasets)) or len(datasets) != len(set(datasets)):
            raise ValueError("datasets must be sorted and unique")
        if not isinstance(self.lanes, FrozenLaneSnapshot):
            raise ValueError("lanes must be frozen")
        test_sizes = _immutable_mapping(self.test_sizes, "test_sizes")
        if set(test_sizes) != set(datasets):
            raise ValueError("test_sizes must cover the exact frozen datasets")
        for dataset, size in test_sizes.items():
            _nonempty_name(dataset, "test_sizes dataset")
            _integer(size, "test_sizes value", minimum=1)
        _nonempty_name(self.code_commit, "code_commit")
        versions = _immutable_mapping(self.runtime_versions, "runtime_versions")
        if not versions or any(not isinstance(key, str) or not isinstance(value, str) for key, value in versions.items()):
            raise ValueError("runtime_versions must be non-empty string pairs")
        if not isinstance(self.config, Exp417FrozenConfig):
            raise ValueError("config must be an Exp417 frozen config")
        self.config.validate()
        if self.database_sha256 != self.config.database_sha256:
            raise ValueError("database identity is not frozen")
        if len(datasets) != self.config.expected_datasets:
            raise ValueError("population count is not frozen")
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "test_sizes", test_sizes)
        object.__setattr__(self, "runtime_versions", versions)


@dataclass(frozen=True)
class LengthGroup:
    length: int
    series: tuple[RawSeries, ...]

    def __post_init__(self) -> None:
        length = _integer(self.length, "length", minimum=1)
        series = _tuple(self.series, "series")
        if not series or any(not isinstance(item, RawSeries) for item in series):
            raise ValueError("series must contain RawSeries records")
        if any(len(item.values) != length for item in series):
            raise ValueError("series lengths must match the length group")
        _validate_unique_series_indices(series)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "series", series)


@dataclass(frozen=True)
class ProcessedGroup:
    group: LengthGroup
    raw_center: float
    raw_scale: float
    difference_center: float
    difference_scale: float
    status: str
    channels: Mapping[int, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group, LengthGroup):
            raise ValueError("group must be a LengthGroup")
        _finite(self.raw_center, "raw_center")
        _finite(self.difference_center, "difference_center")
        raw_scale = _finite(self.raw_scale, "raw_scale", minimum=0.0)
        difference_scale = _finite(self.difference_scale, "difference_scale", minimum=0.0)
        if self.status not in _PROCESSED_GROUP_STATUSES:
            raise ValueError("invalid processed group status")
        requires_zero_scale_abstention = min(raw_scale, difference_scale) <= 1e-12
        if requires_zero_scale_abstention and self.status not in {"zero_scale_abstain", "nonfinite_abstain"}:
            raise ValueError("zero_scale_abstain is required for scale <= 1e-12")
        if self.status == "zero_scale_abstain" and not requires_zero_scale_abstention:
            raise ValueError("zero_scale_abstain requires scale <= 1e-12")
        channels = self.channels
        if channels is None:
            frozen_channels: Mapping[int, np.ndarray] = MappingProxyType({})
        elif not isinstance(channels, Mapping):
            raise ValueError("channels must be a mapping")
        else:
            expected_indices = tuple(item.instance_index for item in self.group.series)
            if tuple(sorted(channels)) != expected_indices:
                raise ValueError("channels must cover the exact length-group indices")
            copied: dict[int, np.ndarray] = {}
            for index, values in channels.items():
                if type(index) is not int or not isinstance(values, np.ndarray):
                    raise ValueError("channels must map integer indices to arrays")
                if values.dtype != np.dtype("float32") or values.shape != (2, self.group.length):
                    raise ValueError("channels must contain exact float32 two-channel arrays")
                if not np.isfinite(values).all():
                    raise ValueError("channels must be finite")
                immutable_bytes = np.ascontiguousarray(values, dtype=np.float32).tobytes()
                immutable = np.frombuffer(immutable_bytes, dtype=np.float32).reshape(values.shape)
                copied[index] = immutable
            frozen_channels = MappingProxyType(copied)
        if self.status in {"candidate_eligible", "diagnostic_only"} and not frozen_channels:
            raise ValueError("usable processed groups require channels")
        if self.status not in {"candidate_eligible", "diagnostic_only"} and frozen_channels:
            raise ValueError("abstained processed groups cannot retain channels")
        object.__setattr__(self, "channels", frozen_channels)


@dataclass(frozen=True)
class HorizonScore:
    horizon: int
    score: float
    p_value: float
    persistence: float

    def __post_init__(self) -> None:
        _integer(self.horizon, "horizon", minimum=1)
        _finite(self.score, "score")
        _finite(self.p_value, "p_value", minimum=0.0, maximum=1.0)
        _finite(self.persistence, "persistence", minimum=0.0, maximum=1.0)


@dataclass(frozen=True)
class SeedDecision:
    seed: int
    scores: Mapping[int, HorizonScore]
    candidate: bool
    config: Exp417FrozenConfig = FROZEN_CONFIG

    def __post_init__(self) -> None:
        _integer(self.seed, "seed", minimum=1)
        if type(self.candidate) is not bool:
            raise ValueError("candidate must be a bool")
        if not isinstance(self.config, Exp417FrozenConfig):
            raise ValueError("config must be an Exp417 frozen config")
        self.config.validate()
        scores = _immutable_mapping(self.scores, "scores")
        horizons = tuple(scores)
        if len(horizons) != 3:
            raise ValueError("scores must contain three horizons")
        if horizons != tuple(sorted(horizons)) or len(set(horizons)) != 3:
            raise ValueError("horizons must be sorted and unique")
        if any(type(key) is not int or not isinstance(value, HorizonScore) or key != value.horizon for key, value in scores.items()):
            raise ValueError("scores must map exact horizon integers to HorizonScore records")
        expected_candidate = sum(
            score.p_value <= self.config.candidate_alpha
            and score.persistence >= self.config.persistence_fraction
            for score in scores.values()
        ) >= 2
        if self.candidate != expected_candidate:
            raise ValueError("candidate decision is inconsistent with horizon evidence")
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class GroupQualification:
    length: int
    series_count: int
    status: str
    oof_candidate_rate: float
    wilson_upper: float
    pooled_synthetic_sensitivity: float
    per_type_synthetic_sensitivity: Mapping[str, float]
    seed_agreement: bool
    finite: bool = True
    backend_fallback: bool = False

    def __post_init__(self) -> None:
        _integer(self.length, "length", minimum=1)
        _integer(self.series_count, "series_count", minimum=0)
        if self.status not in _GROUP_STATUSES:
            raise ValueError("invalid group status")
        for name in ("oof_candidate_rate", "wilson_upper", "pooled_synthetic_sensitivity"):
            _finite(getattr(self, name), name, minimum=0.0, maximum=1.0)
        if type(self.seed_agreement) is not bool or type(self.finite) is not bool or type(self.backend_fallback) is not bool:
            raise ValueError("qualification flags must be bool")
        sensitivity = _immutable_mapping(self.per_type_synthetic_sensitivity, "per_type_synthetic_sensitivity")
        if set(sensitivity) != _PRIMARY_PERTURBATIONS:
            raise ValueError("per_type_synthetic_sensitivity must cover every primary perturbation")
        for value in sensitivity.values():
            _finite(value, "per-type synthetic sensitivity", minimum=0.0, maximum=1.0)
        if self.status == "qualified":
            if self.series_count < FROZEN_CONFIG.min_train_candidate:
                raise ValueError("qualified group has insufficient TRAIN series")
            if self.oof_candidate_rate > FROZEN_CONFIG.oof_candidate_rate or self.wilson_upper > FROZEN_CONFIG.oof_wilson_upper:
                raise ValueError("qualified group fails OOF safety")
            if self.pooled_synthetic_sensitivity < FROZEN_CONFIG.pooled_synthetic_sensitivity or any(value < FROZEN_CONFIG.per_type_synthetic_sensitivity for value in sensitivity.values()):
                raise ValueError("qualified group fails synthetic safety")
            if not self.seed_agreement or not self.finite or self.backend_fallback:
                raise ValueError("qualified group fails deterministic backend safety")
        object.__setattr__(self, "per_type_synthetic_sensitivity", sensitivity)


@dataclass(frozen=True)
class BackendAudit:
    backend: str
    candidate_identity: bool
    max_normalized_error: float
    spearman: float
    speedup: float
    fallback: bool
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "mps", "numba"}:
            raise ValueError("backend is invalid")
        if type(self.candidate_identity) is not bool or type(self.fallback) is not bool:
            raise ValueError("backend flags must be bool")
        _finite(self.max_normalized_error, "max_normalized_error", minimum=0.0)
        _finite(self.spearman, "spearman", minimum=-1.0, maximum=1.0)
        _finite(self.speedup, "speedup", minimum=0.0)
        if self.fallback_reason is not None and (not isinstance(self.fallback_reason, str) or not self.fallback_reason):
            raise ValueError("fallback_reason must be a non-empty string or None")
        if self.backend == "mps" and (
            not self.candidate_identity
            or self.max_normalized_error > FROZEN_CONFIG.mps_max_error
            or self.spearman < FROZEN_CONFIG.mps_min_spearman
            or self.speedup < FROZEN_CONFIG.mps_min_speedup
            or self.fallback
        ):
            raise ValueError("MPS backend fails frozen parity gates")


@dataclass(frozen=True)
class FrozenTrainContract:
    dataset: str
    groups: tuple[GroupQualification, ...]
    backend: BackendAudit
    train_identity_sha256: str
    near_repeat_status: str = "not_available"

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, "dataset")
        groups = _tuple(self.groups, "groups")
        if not groups or any(not isinstance(group, GroupQualification) for group in groups):
            raise ValueError("groups must contain GroupQualification records")
        lengths = tuple(group.length for group in groups)
        if lengths != tuple(sorted(lengths)) or len(lengths) != len(set(lengths)):
            raise ValueError("group lengths must be sorted and unique")
        if not isinstance(self.backend, BackendAudit):
            raise ValueError("backend must be a BackendAudit")
        _nonempty_name(self.train_identity_sha256, "train_identity_sha256")
        if self.near_repeat_status != "not_available":
            raise ValueError("near-repeat guard must remain not_available")
        object.__setattr__(self, "groups", groups)


@dataclass(frozen=True)
class ShadowCandidate:
    dataset: str
    instance_index: int
    length: int
    seed_decisions: tuple[SeedDecision, ...]

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, "dataset")
        _integer(self.instance_index, "instance_index")
        _integer(self.length, "length", minimum=1)
        decisions = _tuple(self.seed_decisions, "seed_decisions")
        if len(decisions) != len(FROZEN_CONFIG.model_seeds):
            raise ValueError("candidate requires three seed decisions")
        if any(not isinstance(decision, SeedDecision) for decision in decisions):
            raise ValueError("seed_decisions must contain SeedDecision records")
        seeds = tuple(decision.seed for decision in decisions)
        if set(seeds) != set(FROZEN_CONFIG.model_seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("candidate requires the three exact model seeds")
        if not all(decision.candidate for decision in decisions):
            raise ValueError("candidate lacks all-seed consensus")
        horizons = tuple(next(iter(decisions)).scores)
        if any(tuple(decision.scores) != horizons for decision in decisions[1:]):
            raise ValueError("candidate seed horizons are inconsistent")
        expected_horizons = (
            1,
            ceil(FROZEN_CONFIG.horizon_fractions[0] * self.length),
            ceil(FROZEN_CONFIG.horizon_fractions[1] * self.length),
        )
        if len(set(expected_horizons)) != 3 or horizons != expected_horizons:
            raise ValueError("candidate horizons do not match its exact length")
        object.__setattr__(self, "seed_decisions", decisions)


@dataclass(frozen=True)
class TestInstanceDecision:
    """All frozen seed decisions for one exact TEST identity, including negatives."""

    dataset: str
    instance_index: int
    length: int
    seed_decisions: tuple[SeedDecision, ...]

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, "dataset")
        _integer(self.instance_index, "instance_index")
        _integer(self.length, "length", minimum=1)
        decisions = _tuple(self.seed_decisions, "seed_decisions")
        if len(decisions) != len(FROZEN_CONFIG.model_seeds) or any(
            not isinstance(decision, SeedDecision) for decision in decisions
        ):
            raise ValueError("instance decision requires the three exact seed decisions")
        seeds = tuple(decision.seed for decision in decisions)
        if set(seeds) != set(FROZEN_CONFIG.model_seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("instance decision requires the three exact frozen model seeds")
        horizons = tuple(decisions[0].scores)
        expected_horizons = (
            1,
            ceil(FROZEN_CONFIG.horizon_fractions[0] * self.length),
            ceil(FROZEN_CONFIG.horizon_fractions[1] * self.length),
        )
        if (
            len(set(expected_horizons)) != 3
            or horizons != expected_horizons
            or any(tuple(decision.scores) != horizons for decision in decisions[1:])
        ):
            raise ValueError("instance decision horizons do not match its exact length")
        object.__setattr__(self, "seed_decisions", decisions)

    @property
    def candidate(self) -> bool:
        return all(decision.candidate for decision in self.seed_decisions)


@dataclass(frozen=True)
class DatasetDecisionEvidence:
    """Exhaustive, provenance-bound TEST decisions for one frozen dataset."""

    dataset: str
    context: FrozenRunContext
    frozen_train_contract: FrozenTrainContract
    database_sha256: str
    train_identity_sha256: str
    decisions: tuple[TestInstanceDecision, ...]

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, "dataset")
        if not isinstance(self.context, FrozenRunContext):
            raise ValueError("context must be a FrozenRunContext")
        if not isinstance(self.frozen_train_contract, FrozenTrainContract):
            raise ValueError("frozen_train_contract must be a FrozenTrainContract")
        if self.dataset not in self.context.test_sizes:
            raise ValueError("dataset must be in the frozen context")
        if self.frozen_train_contract.dataset != self.dataset:
            raise ValueError("frozen train contract dataset does not match evidence")
        if self.database_sha256 != self.context.database_sha256:
            raise ValueError("database hash does not match frozen context")
        if self.train_identity_sha256 != self.frozen_train_contract.train_identity_sha256:
            raise ValueError("TRAIN contract hash does not match frozen contract")
        decisions = _tuple(self.decisions, "decisions")
        if any(not isinstance(decision, TestInstanceDecision) for decision in decisions):
            raise ValueError("decisions must contain TestInstanceDecision records")
        expected_size = self.context.test_sizes[self.dataset]
        indices = tuple(decision.instance_index for decision in decisions)
        if indices != tuple(range(expected_size)):
            raise ValueError("decisions must cover every exact TEST index in sorted order")
        if any(decision.dataset != self.dataset for decision in decisions):
            raise ValueError("decision dataset does not match evidence")
        train_lengths = {group.length for group in self.frozen_train_contract.groups}
        if any(decision.length not in train_lengths for decision in decisions):
            raise ValueError("decision length is absent from the frozen TRAIN contract")
        object.__setattr__(self, "decisions", decisions)

    @property
    def test_size(self) -> int:
        return self.context.test_sizes[self.dataset]

    @property
    def derived_candidates(self) -> tuple[ShadowCandidate, ...]:
        return tuple(
            ShadowCandidate(
                dataset=decision.dataset,
                instance_index=decision.instance_index,
                length=decision.length,
                seed_decisions=decision.seed_decisions,
            )
            for decision in self.decisions
            if decision.candidate
            and not self.context.lanes.contains(decision.dataset, decision.instance_index)
        )


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    test_size: int
    group_count: int
    groups: tuple[GroupQualification, ...]
    candidates: tuple[ShadowCandidate, ...]
    decision_evidence: DatasetDecisionEvidence
    lanes: FrozenLaneSnapshot
    context: FrozenRunContext
    status: str

    def __post_init__(self) -> None:
        _nonempty_name(self.dataset, "dataset")
        test_size = _integer(self.test_size, "test_size", minimum=1)
        group_count = _integer(self.group_count, "group_count", minimum=0)
        groups = _tuple(self.groups, "groups")
        if len(groups) != group_count or any(not isinstance(group, GroupQualification) for group in groups):
            raise ValueError("group accounting is inconsistent")
        lengths = tuple(group.length for group in groups)
        if lengths != tuple(sorted(lengths)) or len(lengths) != len(set(lengths)):
            raise ValueError("group accounting requires sorted unique lengths")
        candidates = _validated_dataset_candidates(self.candidates, self.dataset, test_size, "candidates")
        if not isinstance(self.decision_evidence, DatasetDecisionEvidence):
            raise ValueError("decision_evidence must be DatasetDecisionEvidence")
        if self.decision_evidence.dataset != self.dataset:
            raise ValueError("decision evidence dataset does not match result")
        observed = self.decision_evidence.derived_candidates
        if not isinstance(self.lanes, FrozenLaneSnapshot):
            raise ValueError("lanes must be frozen")
        if not isinstance(self.context, FrozenRunContext):
            raise ValueError("context must be a FrozenRunContext")
        if self.decision_evidence.context != self.context:
            raise ValueError("decision evidence context does not match result")
        if self.lanes != self.context.lanes:
            raise ValueError("lanes must match the frozen context")
        if self.dataset not in self.context.test_sizes or test_size != self.context.test_sizes[self.dataset]:
            raise ValueError("test_size must match the frozen context")
        if test_size != self.decision_evidence.test_size:
            raise ValueError("test_size must match complete decision evidence")
        if groups != self.decision_evidence.frozen_train_contract.groups:
            raise ValueError("groups must match the frozen TRAIN contract")
        if any(
            self.lanes.contains(candidate.dataset, candidate.instance_index)
            for candidate in (*candidates, *observed)
        ):
            raise ValueError("candidate overlaps a frozen lane")
        if self.status not in _DATASET_STATUSES:
            raise ValueError("invalid dataset status")
        rate = len(observed) / test_size
        if rate > FROZEN_CONFIG.dataset_workload_rate:
            if self.status != "abstained" or candidates:
                raise ValueError("dataset workload gate requires abstention without candidates")
        elif self.status in {"abstained", "failed"} and candidates:
            raise ValueError("abstained or failed dataset cannot retain candidates")
        elif self.status == "qualified" and candidates != observed:
            raise ValueError("qualified dataset must retain exact observed candidates")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "decision_evidence", self.decision_evidence)

    @property
    def observed_candidate_count(self) -> int:
        return len(self.decision_evidence.derived_candidates)

    @property
    def observed_candidates(self) -> tuple[ShadowCandidate, ...]:
        return self.decision_evidence.derived_candidates

    @property
    def candidate_rate(self) -> float:
        return self.observed_candidate_count / self.test_size


def _validated_dataset_candidates(
    values: Sequence[ShadowCandidate], dataset: str, test_size: int, name: str
) -> tuple[ShadowCandidate, ...]:
    candidates = _tuple(values, name)
    if any(not isinstance(candidate, ShadowCandidate) for candidate in candidates):
        raise ValueError(f"{name} must contain ShadowCandidate records")
    keys = tuple((candidate.dataset, candidate.instance_index) for candidate in candidates)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} coordinates contain a duplicate")
    if any(candidate.dataset != dataset or candidate.instance_index >= test_size for candidate in candidates):
        raise ValueError(f"{name} coordinate is outside its dataset TEST range")
    return candidates


@dataclass(frozen=True)
class RunSummary:
    results: tuple[DatasetResult, ...]
    total_test: int
    publication_allowed: bool
    status: str
    context: FrozenRunContext

    def __post_init__(self) -> None:
        if not isinstance(self.context, FrozenRunContext):
            raise ValueError("context must be a FrozenRunContext")
        results = _tuple(self.results, "results")
        if any(not isinstance(result, DatasetResult) for result in results):
            raise ValueError("results must contain DatasetResult records")
        datasets = tuple(result.dataset for result in results)
        if datasets != tuple(sorted(datasets)) or len(datasets) != len(set(datasets)):
            raise ValueError("results must have sorted unique datasets")
        if datasets != self.context.datasets:
            raise ValueError("results do not provide frozen context coverage")
        if any(result.lanes != self.context.lanes or result.context != self.context for result in results):
            raise ValueError("results do not use the frozen run context")
        total_test = _integer(self.total_test, "total_test", minimum=0)
        frozen_total = sum(self.context.test_sizes.values())
        if total_test != frozen_total or total_test != sum(result.test_size for result in results):
            raise ValueError("total_test must match the frozen TEST population")
        if type(self.publication_allowed) is not bool:
            raise ValueError("publication_allowed must be bool")
        if self.status not in {"complete", "global_workload_gate_failed", "failed"}:
            raise ValueError("invalid run status")
        prospective_total = sum(len(result.observed_candidates) for result in results)
        global_rate = prospective_total / total_test if total_test else 0.0
        if global_rate > FROZEN_CONFIG.global_workload_rate:
            if self.publication_allowed or self.status != "global_workload_gate_failed" or any(result.candidates for result in results):
                raise ValueError("global workload gate forbids publication")
        elif any(result.status == "failed" for result in results):
            if self.publication_allowed or self.status != "failed":
                raise ValueError("failed dataset summary cannot allow publication")
        elif not self.publication_allowed or self.status != "complete":
            raise ValueError("complete summary must allow publication")
        object.__setattr__(self, "results", results)
