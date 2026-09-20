from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


CHANNELS_ANTIGRAVITY = (
    "high_frequency",
    "low_frequency",
    "amplitude_complexity",
)


def _name_antigravity(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} is invalid")
    return value


def _integer_antigravity(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} is invalid")
    return value


def _finite_antigravity(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is invalid")
    return result


def _digest_antigravity(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is invalid")
    return value


def _vector_antigravity(
    value: object,
    label: str,
    *,
    finite: bool,
    nonempty: bool = True,
) -> NDArray[np.float32]:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if array.ndim != 1 or (nonempty and array.size == 0):
        raise ValueError(f"{label} is invalid")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


def _matrix_antigravity(value: object, label: str) -> NDArray[np.float32]:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} is invalid")
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class RawValueSeries:
    dataset: str
    instance_index: int
    split: str
    values: NDArray[np.float32]
    values_blob_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _name_antigravity(self.dataset, "dataset"))
        object.__setattr__(self, "instance_index", _integer_antigravity(self.instance_index, "instance_index"))
        if self.split not in {"TRAIN", "TEST"}:
            raise ValueError("split is invalid")
        object.__setattr__(self, "values", _vector_antigravity(self.values, "values", finite=False))
        object.__setattr__(self, "values_blob_sha256", _digest_antigravity(self.values_blob_sha256, "values_blob_sha256"))

    @property
    def raw_length(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class ExcludedTrainRowAudit:
    dataset: str
    instance_index: int
    raw_length: int
    nonfinite_positions: tuple[int, ...]
    values_blob_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _name_antigravity(self.dataset, "dataset"))
        object.__setattr__(self, "instance_index", _integer_antigravity(self.instance_index, "instance_index"))
        object.__setattr__(self, "raw_length", _integer_antigravity(self.raw_length, "raw_length", minimum=1))
        positions = tuple(_integer_antigravity(item, "nonfinite position") for item in self.nonfinite_positions)
        if not positions or tuple(sorted(set(positions))) != positions or positions[-1] >= self.raw_length:
            raise ValueError("nonfinite positions are invalid")
        object.__setattr__(self, "nonfinite_positions", positions)
        object.__setattr__(self, "values_blob_sha256", _digest_antigravity(self.values_blob_sha256, "values_blob_sha256"))


@dataclass(frozen=True)
class PreparedTrainGroup:
    dataset: str
    raw_length: int
    rows: tuple[RawValueSeries, ...]
    matrix: NDArray[np.float32] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dataset = _name_antigravity(self.dataset, "dataset")
        raw_length = _integer_antigravity(self.raw_length, "raw_length", minimum=1)
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("prepared TRAIN group rows are invalid")
        if any(
            type(row) is not RawValueSeries
            or row.dataset != dataset
            or row.split != "TRAIN"
            or row.raw_length != raw_length
            or not np.isfinite(row.values).all()
            for row in rows
        ):
            raise ValueError("prepared TRAIN rows must share the same dataset and raw length and be finite")
        indices = tuple(row.instance_index for row in rows)
        if len(set(indices)) != len(indices):
            raise ValueError("prepared TRAIN row identities are duplicated")
        matrix = np.ascontiguousarray(np.stack([row.values for row in rows]), dtype=np.float32)
        matrix.flags.writeable = False
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "raw_length", raw_length)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class DecompositionEvidence:
    depth: int
    low: NDArray[np.float32]
    high: NDArray[np.float32]
    amplitude_complexity: NDArray[np.float32]
    flat_or_zero_weight: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", _integer_antigravity(self.depth, "depth", minimum=1))
        low = _vector_antigravity(self.low, "low", finite=True)
        high = _vector_antigravity(self.high, "high", finite=True)
        complexity = _vector_antigravity(self.amplitude_complexity, "amplitude_complexity", finite=True)
        if low.shape != high.shape or complexity.size != 9:
            raise ValueError("decomposition evidence shape is invalid")
        flags = tuple(self.flat_or_zero_weight)
        if len(flags) != 9 or any(type(flag) is not bool for flag in flags):
            raise ValueError("decomposition audit flags are invalid")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "amplitude_complexity", complexity)
        object.__setattr__(self, "flat_or_zero_weight", flags)


@dataclass(frozen=True)
class ChannelCalibration:
    channel: str
    references: NDArray[np.float32] | None
    center: NDArray[np.float32] | None
    scale: NDArray[np.float32] | None
    oof_scores: NDArray[np.float32] | None
    threshold: float | None
    available: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS_ANTIGRAVITY:
            raise ValueError("channel is invalid")
        if type(self.available) is not bool:
            raise ValueError("calibration availability is invalid")
        if not self.available:
            if type(self.reason) is not str or not self.reason:
                raise ValueError("unavailable calibration requires a reason")
            if any(value is not None for value in (self.references, self.center, self.scale, self.oof_scores, self.threshold)):
                raise ValueError("unavailable calibration cannot contain numeric state")
            return
        if self.reason is not None or any(value is None for value in (self.references, self.center, self.scale, self.oof_scores, self.threshold)):
            raise ValueError("available calibration is incomplete")
        references = _matrix_antigravity(self.references, "references")
        center = _vector_antigravity(self.center, "center", finite=True)
        scale = _vector_antigravity(self.scale, "scale", finite=True)
        oof = _vector_antigravity(self.oof_scores, "oof_scores", finite=True)
        if center.size not in {1, references.shape[1]} or scale.shape != center.shape or np.any(scale <= 0.0):
            raise ValueError("calibration transform shape is invalid")
        if oof.size != references.shape[0]:
            raise ValueError("calibration OOF score count is invalid")
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "oof_scores", oof)
        object.__setattr__(self, "threshold", _finite_antigravity(self.threshold, "threshold"))


@dataclass(frozen=True)
class InstanceChannelEvidence:
    channel: str
    score: float | None
    threshold: float | None
    abnormal: bool
    available: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS_ANTIGRAVITY:
            raise ValueError("channel is invalid")
        if type(self.available) is not bool or type(self.abnormal) is not bool:
            raise ValueError("channel evidence flags are invalid")
        if self.available:
            if self.reason is not None or self.score is None or self.threshold is None:
                raise ValueError("available channel evidence is incomplete")
            object.__setattr__(self, "score", _finite_antigravity(self.score, "score"))
            object.__setattr__(self, "threshold", _finite_antigravity(self.threshold, "threshold"))
        elif self.abnormal or type(self.reason) is not str or not self.reason or self.score is not None or self.threshold is not None:
            raise ValueError("unavailable channel evidence is invalid")


@dataclass(frozen=True)
class InstanceScoreEvidence:
    dataset: str
    instance_index: int
    raw_length: int
    channels: tuple[InstanceChannelEvidence, ...]
    outside_all_frozen_lanes: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _name_antigravity(self.dataset, "dataset"))
        object.__setattr__(self, "instance_index", _integer_antigravity(self.instance_index, "instance_index"))
        object.__setattr__(self, "raw_length", _integer_antigravity(self.raw_length, "raw_length", minimum=1))
        channels = tuple(self.channels)
        if tuple(item.channel for item in channels) != CHANNELS_ANTIGRAVITY:
            raise ValueError("instance evidence must preserve the three separate channels")
        if type(self.outside_all_frozen_lanes) is not bool:
            raise ValueError("lane evidence is invalid")
        object.__setattr__(self, "channels", channels)


@dataclass(frozen=True)
class AbstentionEvidence:
    dataset: str
    instance_index: int | None
    raw_length: int | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", _name_antigravity(self.dataset, "dataset"))
        if self.instance_index is not None:
            object.__setattr__(self, "instance_index", _integer_antigravity(self.instance_index, "instance_index"))
        if self.raw_length is not None:
            object.__setattr__(self, "raw_length", _integer_antigravity(self.raw_length, "raw_length", minimum=1))
        object.__setattr__(self, "reason", _name_antigravity(self.reason, "reason"))


@dataclass(frozen=True)
class Exp438DatasetResult:
    dataset: str
    evidence: tuple[InstanceScoreEvidence, ...]
    abstentions: tuple[AbstentionEvidence, ...]
    train_exclusions: tuple[ExcludedTrainRowAudit, ...]

    def __post_init__(self) -> None:
        dataset = _name_antigravity(self.dataset, "dataset")
        evidence = tuple(self.evidence)
        abstentions = tuple(self.abstentions)
        exclusions = tuple(self.train_exclusions)
        if any(item.dataset != dataset for item in (*evidence, *abstentions, *exclusions)):
            raise ValueError("dataset result contains a foreign dataset")
        identities = tuple(item.instance_index for item in evidence)
        if len(set(identities)) != len(identities) or tuple(sorted(identities)) != identities:
            raise ValueError("dataset evidence identities are invalid")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "abstentions", abstentions)
        object.__setattr__(self, "train_exclusions", exclusions)

    @property
    def candidate_coordinates(self) -> Mapping[str, tuple[tuple[str, int], ...]]:
        result: dict[str, tuple[tuple[str, int], ...]] = {}
        for channel in CHANNELS_ANTIGRAVITY:
            result[channel] = tuple(
                (item.dataset, item.instance_index)
                for item in self.evidence
                if item.outside_all_frozen_lanes
                and next(value for value in item.channels if value.channel == channel).abnormal
            )
        return MappingProxyType(result)


@dataclass(frozen=True)
class BackendBatchEvidence:
    backend: str
    batch_size: int
    finite: bool
    candidate_parity: bool
    maximum_absolute_error: float
    cpu_median_seconds: float
    device_median_seconds: float
    speedup: float
    passed: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.backend not in {"numba", "mlx", "mps"}:
            raise ValueError("backend evidence name is invalid")
        object.__setattr__(self, "batch_size", _integer_antigravity(self.batch_size, "batch_size", minimum=1))
        for name in ("finite", "candidate_parity", "passed"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("backend evidence flag is invalid")
        for name in ("maximum_absolute_error", "cpu_median_seconds", "device_median_seconds", "speedup"):
            value = _finite_antigravity(getattr(self, name), name)
            if value < 0.0 or (name != "maximum_absolute_error" and value <= 0.0):
                raise ValueError("backend timing evidence is invalid")
            object.__setattr__(self, name, value)
        if self.passed and self.reason is not None:
            raise ValueError("passing backend evidence cannot contain a reason")
        if not self.passed and (type(self.reason) is not str or not self.reason):
            raise ValueError("failed backend evidence requires a reason")


@dataclass(frozen=True)
class BackendDecision:
    selected_backend: str
    selected_batch_size: int | None
    evidence: tuple[BackendBatchEvidence, ...]
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if self.selected_backend not in {"cpu", "numba", "mlx", "mps"}:
            raise ValueError("selected backend is invalid")
        evidence = tuple(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if self.selected_backend == "cpu":
            if self.selected_batch_size is not None or type(self.fallback_reason) is not str or not self.fallback_reason:
                raise ValueError("CPU fallback decision is invalid")
            return
        batch_size = _integer_antigravity(self.selected_batch_size, "selected_batch_size", minimum=1)
        if self.fallback_reason is not None or not any(item.backend == self.selected_backend and item.batch_size == batch_size and item.passed for item in evidence):
            raise ValueError("accelerated backend decision lacks passing evidence")
        object.__setattr__(self, "selected_batch_size", batch_size)


@dataclass(frozen=True)
class Exp438RunSummary:
    total_datasets: int
    completed_datasets: int
    successful_datasets: int
    abstained_datasets: int
    failed_datasets: int
    per_channel_candidate_counts: Mapping[str, int]
    backend: BackendDecision
    prediction_label_access_count: int

    def __post_init__(self) -> None:
        for name in ("total_datasets", "completed_datasets", "successful_datasets", "abstained_datasets", "failed_datasets"):
            object.__setattr__(self, name, _integer_antigravity(getattr(self, name), name))
        if self.completed_datasets > self.total_datasets or self.successful_datasets + self.abstained_datasets + self.failed_datasets != self.completed_datasets:
            raise ValueError("run summary counters are inconsistent")
        counts = dict(self.per_channel_candidate_counts)
        if tuple(sorted(counts)) != tuple(sorted(CHANNELS_ANTIGRAVITY)) or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("per-channel candidate counts are invalid")
        object.__setattr__(self, "per_channel_candidate_counts", MappingProxyType(dict(sorted(counts.items()))))
        if type(self.backend) is not BackendDecision:
            raise ValueError("run summary backend is invalid")
        if self.prediction_label_access_count != 0:
            raise ValueError("prediction label access must be zero")

    @property
    def total_candidate_count(self) -> int:
        return sum(self.per_channel_candidate_counts.values())


_FORBIDDEN_PREDICTION_FIELDS = {
    "label",
    "labels",
    "labels_blob",
    "anomaly_position",
    "anomaly_positions",
    "test_metric",
    "test_metrics",
    "test_f1",
    "mean_combined_f1",
    "precision",
    "recall",
    "confusion",
    "tp",
    "fp",
    "fn",
    "tn",
}


def reject_label_like_fields_antigravity(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("prediction field name is invalid")
            if key.casefold() in _FORBIDDEN_PREDICTION_FIELDS:
                raise ValueError(f"forbidden prediction field: {key}")
            reject_label_like_fields_antigravity(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            reject_label_like_fields_antigravity(item)
