"""Immutable missing-safe, label-free Exp436 scientific/runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isclose
from numbers import Integral, Real
import re
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .config_antigravity import FROZEN_CONFIG


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHANNELS = frozenset({"prototype", "gradient", "acf"})
_STATUSES = frozenset({"scored", "abstained", "failed"})
_BACKENDS = frozenset({"mlx", "mps"})
_COMPLETED_BACKEND_REASONS = frozenset({
    "passed", "nonfinite_output", "candidate_identity_mismatch",
    "absolute_error_exceeded", "normalized_error_exceeded", "speedup_below_threshold",
})


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _backend_accelerator_device(backend: str, value: object) -> bool:
    """Reject CPU-like provenance before evidence can claim accelerator availability."""

    device = str(value).strip().lower()
    if not device or "cpu" in device:
        return False
    if backend == "mlx":
        return "gpu" in device or "metal" in device
    return "mps" in device or "metal" in device


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _optional_finite(value: object, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _tuple(value: object, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def _canonical_digest(payload: Mapping[str, object]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _immutable_matrix(value: object, label: str, *, rows: int, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape != (rows, columns):
        raise ValueError(f"{label} must have shape ({rows}, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    # A bytes owner makes re-enabling writeability impossible, unlike merely
    # setting writeable=False on a mutable ndarray.
    immutable = bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return np.frombuffer(immutable, dtype=np.float32).reshape(rows, columns)


def _immutable_optional_matrix(value: object | None, label: str, *, rows: int, columns: int) -> np.ndarray | None:
    return None if value is None else _immutable_matrix(value, label, rows=rows, columns=columns)


def _matrix_metadata(value: np.ndarray) -> dict[str, object]:
    """Stable, JSON-safe representation for private bytes-backed matrices."""

    return {"dtype": "float32", "order": "C", "shape": list(value.shape),
            "sha256": sha256(value.tobytes(order="C")).hexdigest()}


def _raw_scalar(value: float) -> float | str:
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "+Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return value


@dataclass(frozen=True)
class RawValueSeries:
    """A raw values-only row; NaN and infinities are deliberately retained."""

    instance_index: int
    values: tuple[float, ...]
    raw_blob_sha256: str

    def __post_init__(self) -> None:
        _integer(self.instance_index, "instance_index")
        values = _tuple(self.values, "values")
        if not values:
            raise ValueError("values must be non-empty")
        converted: list[float] = []
        for value in values:
            if isinstance(value, (bool, str, bytes)):
                raise ValueError("raw value must be numeric")
            try:
                converted.append(float(value))
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("raw value must be numeric") from error
        object.__setattr__(self, "values", tuple(converted))
        object.__setattr__(self, "raw_blob_sha256", _digest(self.raw_blob_sha256, "raw_blob_sha256"))

    @property
    def length(self) -> int:
        return len(self.values)

    def to_payload(self) -> dict[str, object]:
        return {"instance_index": self.instance_index, "values": [_raw_scalar(value) for value in self.values],
                "raw_blob_sha256": self.raw_blob_sha256}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class MissingSafeRawDataset:
    """Values-only TRAIN or TEST identities, with no label-bearing fields."""

    dataset: str
    split: str
    series: tuple[RawValueSeries, ...]

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        if self.split not in {"TRAIN", "TEST"}:
            raise ValueError("split must be TRAIN or TEST")
        series = _tuple(self.series, "series")
        if not series or any(not isinstance(row, RawValueSeries) for row in series):
            raise ValueError("series must contain RawValueSeries records")
        indices = tuple(row.instance_index for row in series)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("series identities must be sorted and unique")
        object.__setattr__(self, "series", series)

    @property
    def instance_indices(self) -> tuple[int, ...]:
        return tuple(row.instance_index for row in self.series)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(row.length for row in self.series)

    @property
    def identity_lengths(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.instance_indices, self.lengths, strict=True))

    def to_payload(self) -> dict[str, object]:
        return {"dataset": self.dataset, "split": self.split, "series": [row.to_payload() for row in self.series]}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class ExcludedTrainRowAudit:
    """Canonical evidence for one raw TRAIN row removed for nonfinite values."""

    dataset: str
    instance_index: int
    length: int
    nonfinite_count: int
    raw_blob_sha256: str
    audit_digest: str | None = None

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.instance_index, "instance_index")
        _integer(self.length, "length", minimum=1)
        _integer(self.nonfinite_count, "nonfinite_count", minimum=1)
        if self.nonfinite_count > self.length:
            raise ValueError("nonfinite_count cannot exceed length")
        object.__setattr__(self, "raw_blob_sha256", _digest(self.raw_blob_sha256, "raw_blob_sha256"))
        expected = _canonical_digest(self.to_payload(include_digest=False))
        digest = expected if self.audit_digest is None else _digest(self.audit_digest, "audit_digest")
        if digest != expected:
            raise ValueError("audit_digest does not match canonical payload")
        object.__setattr__(self, "audit_digest", digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {"schema": "exp436.excluded-train-row.v1", "dataset": self.dataset,
            "instance_index": self.instance_index, "length": self.length, "nonfinite_count": self.nonfinite_count,
            "raw_blob_sha256": self.raw_blob_sha256}
        if include_digest:
            payload["audit_digest"] = self.audit_digest
        return payload

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


def _indices(value: object, *, label: str = "instance_indices", nonempty: bool = True) -> tuple[int, ...]:
    indices = tuple(_integer(item, "instance index") for item in _tuple(value, label))
    if (nonempty and not indices) or indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
        suffix = " and non-empty" if nonempty else ""
        raise ValueError(f"{label} must be sorted, unique{suffix}")
    return indices


@dataclass(frozen=True)
class PreparedTrainGroup:
    dataset: str
    length: int
    instance_indices: tuple[int, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.length, "length", minimum=1)
        indices = _indices(self.instance_indices)
        if len(indices) < FROZEN_CONFIG.min_finite_train_rows:
            raise ValueError("PreparedTrainGroup requires at least min_finite_train_rows references")
        object.__setattr__(self, "instance_indices", indices)
        object.__setattr__(self, "values", _immutable_matrix(self.values, "values", rows=len(indices), columns=self.length))

    @property
    def reference_count(self) -> int:
        return len(self.instance_indices)

    def to_payload(self) -> dict[str, object]:
        return {"schema": "exp436.prepared-train-group.v1", "dataset": self.dataset, "length": self.length,
                "instance_indices": list(self.instance_indices), "reference_count": self.reference_count,
                "values": _matrix_metadata(self.values)}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class UnavailableTrainGroup:
    dataset: str
    length: int
    instance_indices: tuple[int, ...]
    reason: str = "insufficient_finite_train_rows"
    identity_digest: str | None = None

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.length, "length", minimum=1)
        object.__setattr__(self, "instance_indices", _indices(self.instance_indices, nonempty=False))
        if self.reference_count >= FROZEN_CONFIG.min_finite_train_rows:
            raise ValueError("UnavailableTrainGroup requires fewer than min_finite_train_rows references")
        if self.reason != "insufficient_finite_train_rows":
            raise ValueError("reason must be insufficient_finite_train_rows")
        expected = _canonical_digest(self.to_payload(include_digest=False))
        digest = expected if self.identity_digest is None else _digest(self.identity_digest, "identity_digest")
        if digest != expected:
            raise ValueError("identity_digest does not match canonical payload")
        object.__setattr__(self, "identity_digest", digest)

    @property
    def available(self) -> bool:
        return False

    @property
    def reference_count(self) -> int:
        return len(self.instance_indices)

    @property
    def evidence_digest(self) -> str:
        assert self.identity_digest is not None
        return self.identity_digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {"schema": "exp436.unavailable-train-group.v1", "dataset": self.dataset,
            "length": self.length, "instance_indices": list(self.instance_indices), "reason": self.reason}
        if include_digest:
            payload["identity_digest"] = self.identity_digest
        return payload

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


TrainGroup = PreparedTrainGroup | UnavailableTrainGroup


@dataclass(frozen=True)
class TrainPreparation:
    """All original TRAIN lengths exactly once, plus distinct ordered exclusions."""

    dataset: str
    train_identity_lengths: tuple[tuple[int, int], ...]
    groups: tuple[TrainGroup, ...]
    excluded_rows: tuple[ExcludedTrainRowAudit, ...] = ()

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        manifest = _tuple(self.train_identity_lengths, "train_identity_lengths")
        if not manifest:
            raise ValueError("train_identity_lengths must be non-empty")
        identity_lengths: list[tuple[int, int]] = []
        for item in manifest:
            pair = _tuple(item, "train identity-length")
            if len(pair) != 2:
                raise ValueError("train identity-length must contain identity and length")
            identity_lengths.append((_integer(pair[0], "train instance_index"), _integer(pair[1], "train length", minimum=1)))
        source_indices = tuple(index for index, _ in identity_lengths)
        if source_indices != tuple(sorted(source_indices)) or len(source_indices) != len(set(source_indices)):
            raise ValueError("train_identity_lengths must be sorted and unique by identity")
        source_by_index = dict(identity_lengths)
        object.__setattr__(self, "train_identity_lengths", tuple(identity_lengths))
        groups = _tuple(self.groups, "groups")
        if not groups or any(not isinstance(group, (PreparedTrainGroup, UnavailableTrainGroup)) for group in groups):
            raise ValueError("groups must contain TRAIN group records")
        lengths = tuple(group.length for group in groups)
        if lengths != tuple(sorted(lengths)) or len(lengths) != len(set(lengths)):
            raise ValueError("groups must cover every original length once in sorted order")
        if any(group.dataset != self.dataset for group in groups):
            raise ValueError("group dataset does not match preparation")
        if set(lengths) != set(source_by_index.values()):
            raise ValueError("group lengths must equal exact source TRAIN lengths")
        audits = _tuple(self.excluded_rows, "excluded_rows")
        if any(not isinstance(audit, ExcludedTrainRowAudit) for audit in audits):
            raise ValueError("excluded_rows must contain audit records")
        coordinates = tuple((audit.length, audit.instance_index) for audit in audits)
        if coordinates != tuple(sorted(coordinates)) or len(coordinates) != len(set(coordinates)):
            raise ValueError("excluded_rows must be sorted and unique")
        if any(audit.dataset != self.dataset for audit in audits):
            raise ValueError("audit dataset does not match preparation")
        surviving = {(index, group.length) for group in groups for index in group.instance_indices}
        excluded = {(audit.instance_index, audit.length) for audit in audits}
        source = set(identity_lengths)
        if any(source_by_index.get(index) != length for index, length in surviving | excluded):
            raise ValueError("surviving group or exclusion audit coordinate does not match source TRAIN manifest")
        if surviving & excluded or surviving | excluded != source:
            raise ValueError("surviving and excluded identities must exactly partition source TRAIN manifest")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "excluded_rows", audits)

    @property
    def prepared_groups(self) -> tuple[PreparedTrainGroup, ...]:
        return tuple(group for group in self.groups if isinstance(group, PreparedTrainGroup))

    @property
    def unavailable_groups(self) -> tuple[UnavailableTrainGroup, ...]:
        return tuple(group for group in self.groups if isinstance(group, UnavailableTrainGroup))

    def to_payload(self) -> dict[str, object]:
        return {"schema": "exp436.train-preparation.v1", "dataset": self.dataset,
                "train_identity_lengths": [list(item) for item in self.train_identity_lengths],
                "groups": [group.to_payload() for group in self.groups],
                "excluded_rows": [audit.to_payload() for audit in self.excluded_rows]}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class ChannelCalibration:
    channel: str
    available: bool = True
    threshold: float | None = None
    loo_scores: tuple[float, ...] = ()
    center: float | None = None
    scale: float | None = None
    reference_count: int = 0
    reference_digest: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.channel not in _CHANNELS:
            raise ValueError("channel is invalid")
        if type(self.available) is not bool:
            raise ValueError("available must be boolean")
        scores = tuple(_finite(item, "loo_score") for item in _tuple(self.loo_scores, "loo_scores"))
        threshold, center, scale = (_optional_finite(self.threshold, "threshold"), _optional_finite(self.center, "center"), _optional_finite(self.scale, "scale"))
        _integer(self.reference_count, "reference_count")
        object.__setattr__(self, "loo_scores", scores)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        if self.available:
            if len(scores) < FROZEN_CONFIG.min_finite_train_rows or threshold is None:
                raise ValueError("available calibration requires threshold and four LOO scores")
            if self.reference_count != len(scores) or self.reference_digest is None:
                raise ValueError("available calibration requires count and digest")
            object.__setattr__(self, "reference_digest", _digest(self.reference_digest, "reference_digest"))
            if center is None or scale is None or scale <= FROZEN_CONFIG.numeric_floor:
                raise ValueError("available calibration requires finite center and positive scale")
            expected = float(np.quantile(np.asarray(scores, dtype=np.float32), FROZEN_CONFIG.train_quantile, method=FROZEN_CONFIG.quantile_method))
            if threshold != expected:
                raise ValueError("available calibration threshold must equal higher quantile")
            if self.reason is not None:
                raise ValueError("available calibration cannot have reason")
        else:
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("unavailable calibration requires structured reason")
            if scores or threshold is not None or center is not None or scale is not None or self.reference_count != 0 or self.reference_digest is not None:
                raise ValueError("unavailable calibration cannot publish scalars")

    @classmethod
    def unavailable(cls, channel: str, reason: str) -> "ChannelCalibration":
        return cls(channel=channel, available=False, reason=reason)

    @classmethod
    def fixture(cls, channel: str, *, reference_count: int = 4) -> "ChannelCalibration":
        return cls(channel, threshold=1.0, loo_scores=(0.0, 0.1, 0.2, 1.0), center=0.0, scale=1.0,
                   reference_count=reference_count, reference_digest="0" * 64)

    def is_abnormal(self, score: object) -> bool:
        return bool(self.available and _finite(score, "score") > self.threshold)  # type: ignore[operator]

    def to_payload(self) -> dict[str, object]:
        return {"channel": self.channel, "available": self.available, "threshold": self.threshold,
                "loo_scores": list(self.loo_scores), "center": self.center, "scale": self.scale,
                "reference_count": self.reference_count, "reference_digest": self.reference_digest,
                "reason": self.reason}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class CalibratedTrainGroup:
    dataset: str
    length: int
    instance_indices: tuple[int, ...]
    values: np.ndarray | None
    gradients: np.ndarray | None
    acf_values: np.ndarray | None
    prototype: ChannelCalibration
    gradient: ChannelCalibration
    acf: ChannelCalibration

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.length, "length", minimum=1)
        indices = _indices(self.instance_indices)
        rows = len(indices)
        if rows < FROZEN_CONFIG.min_finite_train_rows:
            raise ValueError("CalibratedTrainGroup has insufficient references")
        object.__setattr__(self, "instance_indices", indices)
        for expected, calibration in (("prototype", self.prototype), ("gradient", self.gradient), ("acf", self.acf)):
            if not isinstance(calibration, ChannelCalibration) or calibration.channel != expected:
                raise ValueError("calibration channels do not match group")
            if calibration.available and calibration.reference_count != rows:
                raise ValueError("calibration reference_count must match group rows")
        object.__setattr__(self, "values", _immutable_optional_matrix(self.values, "values", rows=rows, columns=self.length))
        if (self.values is None) != (not self.prototype.available):
            raise ValueError("prototype availability must match values matrix")
        if not self.prototype.available and self.prototype.reason != "zero_scale":
            raise ValueError("prototype unavailability requires zero_scale")
        object.__setattr__(self, "gradients", _immutable_optional_matrix(self.gradients, "gradients", rows=rows, columns=self.length))
        if (self.gradients is None) != (not self.gradient.available):
            raise ValueError("gradient availability must match gradient matrix")
        if not self.gradient.available and self.gradient.reason != "zero_scale":
            raise ValueError("gradient unavailability requires zero_scale")
        acf_columns = min(FROZEN_CONFIG.max_acf_lag, self.length // 4)
        if self.length < FROZEN_CONFIG.min_acf_length and (self.acf_values is not None or self.acf.available):
            raise ValueError("ACF must be unavailable below min_acf_length")
        object.__setattr__(self, "acf_values", _immutable_optional_matrix(self.acf_values, "acf_values", rows=rows, columns=acf_columns))
        if (self.acf_values is None) != (not self.acf.available):
            raise ValueError("acf availability must match acf matrix")
        expected_acf_reason = "short_series" if self.length < FROZEN_CONFIG.min_acf_length else "zero_acf_energy"
        if not self.acf.available and self.acf.reason != expected_acf_reason:
            raise ValueError(f"acf unavailability requires {expected_acf_reason}")

    def to_payload(self) -> dict[str, object]:
        return {"schema": "exp436.calibrated-train-group.v1", "dataset": self.dataset, "length": self.length,
                "instance_indices": list(self.instance_indices), "values": None if self.values is None else _matrix_metadata(self.values),
                "gradients": None if self.gradients is None else _matrix_metadata(self.gradients),
                "acf_values": None if self.acf_values is None else _matrix_metadata(self.acf_values),
                "prototype": self.prototype.to_payload(), "gradient": self.gradient.to_payload(), "acf": self.acf.to_payload()}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


CalibratedTrainGroupResult = CalibratedTrainGroup | UnavailableTrainGroup


@dataclass(frozen=True)
class InstanceScoreEvidence:
    dataset: str
    instance_index: int
    length: int
    prototype_score: float | None = None
    prototype_threshold: float | None = None
    prototype_abnormal: bool = False
    prototype_unavailability_reason: str | None = None
    gradient_score: float | None = None
    gradient_threshold: float | None = None
    gradient_abnormal: bool = False
    gradient_unavailability_reason: str | None = None
    acf_score: float | None = None
    acf_threshold: float | None = None
    acf_abnormal: bool = False
    acf_unavailability_reason: str | None = None
    consensus_candidate: bool = False
    status: str = "scored"
    exclusion_reason: str | None = None
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.instance_index, "instance_index")
        if self.status not in {"scored", "abstained"}:
            raise ValueError("status is invalid")
        _integer(self.length, "length", minimum=1 if self.status != "scored" else FROZEN_CONFIG.min_scored_length)
        allowed_reasons = {
            "prototype": {"zero_scale"},
            "gradient": {"zero_scale"},
            "acf": {"short_series", "zero_acf_energy"},
        }
        scalar_presence: dict[str, bool] = {}
        for channel, score, threshold, abnormal, reason in (
            ("prototype", self.prototype_score, self.prototype_threshold, self.prototype_abnormal, self.prototype_unavailability_reason),
            ("gradient", self.gradient_score, self.gradient_threshold, self.gradient_abnormal, self.gradient_unavailability_reason),
            ("acf", self.acf_score, self.acf_threshold, self.acf_abnormal, self.acf_unavailability_reason),
        ):
            if type(abnormal) is not bool:
                raise ValueError(f"{channel}_abnormal must be boolean")
            score_value, threshold_value = _optional_finite(score, f"{channel}_score"), _optional_finite(threshold, f"{channel}_threshold")
            object.__setattr__(self, f"{channel}_score", score_value)
            object.__setattr__(self, f"{channel}_threshold", threshold_value)
            if (score_value is None) != (threshold_value is None):
                raise ValueError(f"{channel} score and threshold must be both present or absent")
            if abnormal != (score_value is not None and score_value > threshold_value):  # type: ignore[operator]
                raise ValueError(f"{channel} abnormal decision is inconsistent")
            if reason is not None and reason not in allowed_reasons[channel]:
                raise ValueError(f"{channel} unavailability reason is invalid")
            scalar_presence[channel] = score_value is not None
        if self.acf_unavailability_reason is not None:
            expected_acf_reason = (
                "short_series"
                if self.length < FROZEN_CONFIG.min_acf_length
                else "zero_acf_energy"
            )
            if self.acf_unavailability_reason != expected_acf_reason:
                raise ValueError(f"acf unavailability requires {expected_acf_reason}")
        expected = self.prototype_abnormal and self.gradient_abnormal and self.acf_abnormal
        if type(self.consensus_candidate) is not bool or self.consensus_candidate != expected:
            raise ValueError("consensus candidate is inconsistent with strict three-channel AND")
        if self.status != "scored" and self.consensus_candidate:
            raise ValueError("abstained TEST evidence cannot be a candidate")
        routing_reasons = {
            "nonfinite_test", "insufficient_series_length", "missing_length", "insufficient_finite_train_rows"
        }
        scored_exclusions = {"frozen_lane", "lane_check_error"}
        valid_exclusions = routing_reasons | scored_exclusions | {"three_channel_consensus_unavailable"}
        if self.exclusion_reason is not None:
            object.__setattr__(self, "exclusion_reason", _name(self.exclusion_reason, "exclusion_reason"))
            if self.exclusion_reason not in valid_exclusions:
                raise ValueError("exclusion reason is invalid")
        channel_reasons = (
            self.prototype_unavailability_reason,
            self.gradient_unavailability_reason,
            self.acf_unavailability_reason,
        )
        if self.exclusion_reason in routing_reasons:
            if self.status != "abstained" or any(scalar_presence.values()) or any(channel_reasons) or any((self.prototype_abnormal, self.gradient_abnormal, self.acf_abnormal, self.consensus_candidate)):
                raise ValueError(f"{self.exclusion_reason} requires scalar-free abstained evidence")
            if (self.exclusion_reason == "insufficient_series_length"
                    and self.length >= FROZEN_CONFIG.min_scored_length):
                raise ValueError("routing insufficient_series_length requires a short row")
            if (self.exclusion_reason in {"missing_length", "insufficient_finite_train_rows"}
                    and self.length < FROZEN_CONFIG.min_scored_length):
                raise ValueError(f"routing {self.exclusion_reason} requires a scored-length row")
        elif self.exclusion_reason == "three_channel_consensus_unavailable":
            if (self.status != "abstained" or not any(channel_reasons) or self.consensus_candidate
                    or any(scalar_presence[channel] != (reason is None) for channel, reason in (
                        ("prototype", self.prototype_unavailability_reason),
                        ("gradient", self.gradient_unavailability_reason),
                        ("acf", self.acf_unavailability_reason),
                    ))):
                raise ValueError("three_channel_consensus_unavailable requires unavailable abstained evidence")
        elif self.exclusion_reason in scored_exclusions:
            if self.status != "scored" or not all(scalar_presence.values()) or any(channel_reasons) or not self.consensus_candidate:
                raise ValueError(f"{self.exclusion_reason} requires scored strict-consensus evidence")
        elif self.exclusion_reason is None:
            if self.status != "scored" or not all(scalar_presence.values()) or any(channel_reasons):
                raise ValueError("unexcluded evidence requires all scored channel scalars")
        expected_digest = _canonical_digest(self.to_payload(include_digest=False))
        digest = expected_digest if self.evidence_digest is None else _digest(self.evidence_digest, "evidence_digest")
        if digest != expected_digest:
            raise ValueError("evidence_digest does not match canonical payload")
        object.__setattr__(self, "evidence_digest", digest)

    @classmethod
    def fixture(cls, *, dataset: str = "Demo", instance_index: int = 3, length: int = 64,
                prototype_abnormal: bool = False, gradient_abnormal: bool = False, acf_abnormal: bool = False,
                status: str = "scored", exclusion_reason: str | None = None) -> "InstanceScoreEvidence":
        if exclusion_reason == "nonfinite_test":
            return cls(dataset, instance_index, length, status=status, exclusion_reason=exclusion_reason)
        def channel(flag: bool) -> tuple[float, float, bool]:
            return (2.0 if flag else 0.0, 1.0, flag)
        p, g, a = channel(prototype_abnormal), channel(gradient_abnormal), channel(acf_abnormal)
        return cls(
            dataset, instance_index, length, p[0], p[1], p[2], None,
            g[0], g[1], g[2], None, a[0], a[1], a[2], None,
            prototype_abnormal and gradient_abnormal and acf_abnormal, status, exclusion_reason,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {name: getattr(self, name) for name in (
            "dataset", "instance_index", "length", "prototype_score", "prototype_threshold", "prototype_abnormal",
            "prototype_unavailability_reason", "gradient_score", "gradient_threshold", "gradient_abnormal",
            "gradient_unavailability_reason", "acf_score", "acf_threshold", "acf_abnormal", "acf_unavailability_reason",
            "consensus_candidate", "status", "exclusion_reason")}
        if include_digest:
            payload["evidence_digest"] = self.evidence_digest
        return payload

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class Exp436Candidate:
    dataset: str
    instance_index: int
    length: int
    evidence_digest: str

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.instance_index, "instance_index")
        _integer(self.length, "length", minimum=FROZEN_CONFIG.min_scored_length)
        object.__setattr__(self, "evidence_digest", _digest(self.evidence_digest, "evidence_digest"))

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.dataset, self.instance_index

    def to_payload(self) -> dict[str, object]:
        return {"dataset": self.dataset, "instance_index": self.instance_index, "length": self.length, "evidence_digest": self.evidence_digest}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class Exp436DatasetResult:
    dataset: str
    status: str
    test_instance_indices: tuple[int, ...]
    candidates: tuple[Exp436Candidate, ...] = ()
    evidence: tuple[InstanceScoreEvidence, ...] = ()
    train_exclusion_audits: tuple[ExcludedTrainRowAudit, ...] = ()
    abstention_reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        if self.status not in _STATUSES:
            raise ValueError("status is invalid")
        test_indices = _indices(self.test_instance_indices, label="test_instance_indices")
        object.__setattr__(self, "test_instance_indices", test_indices)
        evidence = _tuple(self.evidence, "evidence")
        if any(not isinstance(item, InstanceScoreEvidence) or item.dataset != self.dataset for item in evidence):
            raise ValueError("evidence dataset/type does not match result")
        evidence_indices = tuple(item.instance_index for item in evidence)
        if evidence_indices != test_indices:
            raise ValueError("evidence must cover exactly every TEST identity with no extra identity")
        object.__setattr__(self, "evidence", evidence)
        candidates = _tuple(self.candidates, "candidates")
        if any(not isinstance(item, Exp436Candidate) or item.dataset != self.dataset for item in candidates):
            raise ValueError("candidate dataset/type does not match result")
        candidate_indices = tuple(item.instance_index for item in candidates)
        if candidate_indices != tuple(sorted(candidate_indices)) or len(candidate_indices) != len(set(candidate_indices)):
            raise ValueError("candidates must be sorted and unique")
        evidence_by_index = {item.instance_index: item for item in evidence}
        for candidate in candidates:
            matched = evidence_by_index.get(candidate.instance_index)
            if matched is None or not matched.consensus_candidate or matched.status != "scored":
                raise ValueError("candidate requires matching scored strict-consensus evidence")
            if matched.exclusion_reason is not None:
                raise ValueError("candidate evidence has an exclusion reason")
            if candidate.length != matched.length or candidate.evidence_digest != matched.evidence_digest:
                raise ValueError("candidate does not match evidence")
        if self.status == "abstained":
            if any(item.status != "abstained" for item in evidence):
                raise ValueError("abstained result requires every row evidence abstained")
            if self.abstention_reason != "all_instances_abstained":
                raise ValueError("abstained result requires all_instances_abstained")
            if self.error is not None:
                raise ValueError("abstained result cannot contain error")
        elif self.status == "scored" and not any(item.status == "scored" for item in evidence):
            raise ValueError("scored result requires scored row evidence")
        expected_candidates = tuple(
            Exp436Candidate(item.dataset, item.instance_index, item.length, item.evidence_digest)
            for item in evidence
            if item.status == "scored" and item.consensus_candidate and item.exclusion_reason is None
        )
        if candidates != expected_candidates:
            raise ValueError("candidates must exactly match eligible strict-consensus evidence")
        if self.status != "scored" and candidates:
            raise ValueError("abstained or failed result cannot contain candidates")
        audits = _tuple(self.train_exclusion_audits, "train_exclusion_audits")
        if any(not isinstance(item, ExcludedTrainRowAudit) or item.dataset != self.dataset for item in audits):
            raise ValueError("TRAIN exclusion audit dataset/type does not match result")
        audit_order = tuple((item.length, item.instance_index) for item in audits)
        if audit_order != tuple(sorted(audit_order)) or len(audit_order) != len(set(audit_order)):
            raise ValueError("TRAIN exclusion audits must be ordered and unique")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "train_exclusion_audits", audits)
        if self.status == "failed":
            object.__setattr__(self, "error", _name(self.error, "error"))
            if self.abstention_reason is not None:
                raise ValueError("failed result cannot contain abstention reason")
        elif self.status == "scored" and (self.abstention_reason is not None or self.error is not None):
            raise ValueError("scored result cannot contain status metadata")

    @property
    def test_size(self) -> int:
        return len(self.test_instance_indices)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def to_payload(self) -> dict[str, object]:
        return {"dataset": self.dataset, "status": self.status, "test_instance_indices": list(self.test_instance_indices),
                "candidates": [item.to_payload() for item in self.candidates], "evidence": [item.to_payload() for item in self.evidence],
                "train_exclusion_audits": [item.to_payload() for item in self.train_exclusion_audits],
                "abstention_reason": self.abstention_reason, "error": self.error}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class BackendBatchEvidence:
    """One truthful batch execution record from the frozen paired protocol."""

    backend: str
    batch_size: int
    state: str
    warmup_pairs_completed: int
    timing_pairs: tuple[tuple[float, float], ...]
    fixture_input_sha256: str
    fixture_output_sha256: str
    finite_outputs: bool | None
    candidate_identity_match: bool | None
    absolute_error_max: float | None
    normalized_error_max: float | None
    cpu_median_seconds: float | None
    device_median_seconds: float | None
    speedup: float | None
    parity_passed: bool | None
    speed_passed: bool | None
    passed: bool
    reason: str
    synchronization_inside_timing: bool = True

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError("backend is invalid")
        object.__setattr__(self, "batch_size", _integer(self.batch_size, "batch_size", minimum=1))
        if self.state not in {"completed", "runtime_failed"}:
            raise ValueError("state must be completed or runtime_failed")
        warmups = _integer(self.warmup_pairs_completed, "warmup_pairs_completed")
        if warmups > FROZEN_CONFIG.backend_warmup_runs:
            raise ValueError("warmup_pairs_completed exceeds frozen count")
        object.__setattr__(self, "warmup_pairs_completed", warmups)
        if self.synchronization_inside_timing is not True:
            raise ValueError("synchronization_inside_timing must be true")
        pairs = _tuple(self.timing_pairs, "timing_pairs")
        maximum_pairs = FROZEN_CONFIG.backend_timed_runs
        if len(pairs) > maximum_pairs or (self.state == "completed" and len(pairs) != maximum_pairs):
            raise ValueError("completed timing_pairs must contain exactly 5 pairs")
        validated: list[tuple[float, float]] = []
        for pair in pairs:
            values = _tuple(pair, "timing pair")
            if len(values) != 2:
                raise ValueError("timing pair must contain CPU and device seconds")
            validated.append((_finite(values[0], "cpu_seconds", positive=True), _finite(values[1], "device_seconds", positive=True)))
        object.__setattr__(self, "timing_pairs", tuple(validated))
        object.__setattr__(self, "fixture_input_sha256", _digest(self.fixture_input_sha256, "fixture_input_sha256"))
        object.__setattr__(self, "fixture_output_sha256", _digest(self.fixture_output_sha256, "fixture_output_sha256"))
        if type(self.passed) is not bool:
            raise ValueError("passed must be boolean")
        if self.state == "runtime_failed":
            if any(getattr(self, name) is not None for name in (
                "finite_outputs", "candidate_identity_match", "absolute_error_max", "normalized_error_max",
                "cpu_median_seconds", "device_median_seconds", "speedup", "parity_passed", "speed_passed",
            )) or self.passed:
                raise ValueError("runtime_failed batches retain no derived metrics and cannot pass")
            reason = _name(self.reason, "reason")
            if re.fullmatch(rf"{self.backend}_runtime_failed:[A-Za-z][A-Za-z0-9_]*", reason) is None:
                raise ValueError("runtime_failed reason must be canonical backend exception evidence")
            return
        if warmups != FROZEN_CONFIG.backend_warmup_runs:
            raise ValueError("completed batches require exactly 2 warmup pairs")
        for name in ("finite_outputs", "candidate_identity_match", "parity_passed", "speed_passed"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        absolute = _finite(self.absolute_error_max, "absolute_error_max")
        normalized = _finite(self.normalized_error_max, "normalized_error_max")
        if absolute < 0 or normalized < 0:
            raise ValueError("errors must be non-negative")
        cpu_median, device_median = float(median(pair[0] for pair in validated)), float(median(pair[1] for pair in validated))
        if not isclose(_finite(self.cpu_median_seconds, "cpu_median_seconds", positive=True), cpu_median, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("cpu median does not match timing pairs")
        if not isclose(_finite(self.device_median_seconds, "device_median_seconds", positive=True), device_median, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("device median does not match timing pairs")
        speedup = cpu_median / device_median
        if not isclose(_finite(self.speedup, "speedup", positive=True), speedup, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("speedup does not match timing medians")
        parity = bool(self.finite_outputs and self.candidate_identity_match and absolute <= FROZEN_CONFIG.score_tolerance and normalized <= FROZEN_CONFIG.score_tolerance)
        speed = speedup >= FROZEN_CONFIG.backend_min_speedup
        if self.parity_passed != parity or self.speed_passed != speed or self.passed != (parity and speed):
            raise ValueError("derived parity/speed/pass result is inconsistent")
        expected_reason = "passed" if parity and speed else (
            "nonfinite_output" if not self.finite_outputs else
            "candidate_identity_mismatch" if not self.candidate_identity_match else
            "absolute_error_exceeded" if absolute > FROZEN_CONFIG.score_tolerance else
            "normalized_error_exceeded" if normalized > FROZEN_CONFIG.score_tolerance else
            "speedup_below_threshold"
        )
        if self.reason != expected_reason or self.reason not in _COMPLETED_BACKEND_REASONS:
            raise ValueError("completed batch reason is inconsistent with gates")
        object.__setattr__(self, "absolute_error_max", absolute)
        object.__setattr__(self, "normalized_error_max", normalized)
        object.__setattr__(self, "cpu_median_seconds", cpu_median)
        object.__setattr__(self, "device_median_seconds", device_median)
        object.__setattr__(self, "speedup", speedup)
        _name(self.reason, "reason")

    def to_payload(self) -> dict[str, object]:
        return {"backend": self.backend, "batch_size": self.batch_size, "state": self.state,
                "warmup_pairs_completed": self.warmup_pairs_completed, "timing_pairs": [list(pair) for pair in self.timing_pairs],
                "synchronization_inside_timing": self.synchronization_inside_timing, "finite_outputs": self.finite_outputs,
                "candidate_identity_match": self.candidate_identity_match, "absolute_error_max": self.absolute_error_max,
                "normalized_error_max": self.normalized_error_max, "fixture_input_sha256": self.fixture_input_sha256,
                "fixture_output_sha256": self.fixture_output_sha256, "cpu_median_seconds": self.cpu_median_seconds,
                "device_median_seconds": self.device_median_seconds, "speedup": self.speedup,
                "parity_passed": self.parity_passed, "speed_passed": self.speed_passed, "passed": self.passed,
                "reason": self.reason}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


@dataclass(frozen=True)
class BackendBenchmarkEvidence:
    """All frozen batches and backend-specific version provenance."""

    backend: str
    available: bool
    batches: tuple[BackendBatchEvidence, ...]
    python_version: str | None
    numpy_version: str | None
    device: str | None
    device_info: str | None
    metal_available: bool | None
    protocol: str
    selected_batch_size: int | None
    selected_device_median_seconds: float | None
    fallback_reason: str | None
    seed: int = FROZEN_CONFIG.backend_smoke_seed
    mlx_version: str | None = None
    mlx_metal_version: str | None = None
    pytorch_version: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError("backend is invalid")
        seed = _integer(self.seed, "seed")
        if seed != FROZEN_CONFIG.backend_smoke_seed:
            raise ValueError("seed must equal frozen backend seed")
        object.__setattr__(self, "seed", seed)
        if type(self.available) is not bool:
            raise ValueError("available must be boolean")
        batches = _tuple(self.batches, "batches")
        if any(not isinstance(item, BackendBatchEvidence) for item in batches):
            raise ValueError("batches contain invalid evidence")
        sizes = tuple(item.batch_size for item in batches)
        if self.available and (sizes != FROZEN_CONFIG.backend_batch_sizes or any(item.backend != self.backend for item in batches)):
            raise ValueError("available benchmark batches must cover exactly all frozen batch sizes")
        if not self.available and batches:
            raise ValueError("unavailable benchmark must retain empty batches")
        object.__setattr__(self, "batches", batches)
        if self.protocol != FROZEN_CONFIG.backend_protocol:
            raise ValueError("protocol must equal frozen backend protocol")
        if self.backend == "mlx" and self.pytorch_version is not None:
            raise ValueError("MLX evidence cannot claim PyTorch provenance")
        if self.backend == "mps" and (self.mlx_version is not None or self.mlx_metal_version is not None):
            raise ValueError("MPS evidence cannot claim MLX provenance")
        if self.available:
            input_digests = {item.fixture_input_sha256 for item in batches}
            output_digests = {item.fixture_output_sha256 for item in batches}
            if len(input_digests) != 1 or len(output_digests) != 1:
                raise ValueError("available benchmark batches must retain one shared fixture digest pair")
            for name in ("python_version", "numpy_version", "device", "device_info"):
                _name(getattr(self, name), name)
            if not _backend_accelerator_device(self.backend, self.device):
                raise ValueError("available benchmark requires backend-specific accelerator device")
            if self.metal_available is not True:
                raise ValueError("available benchmark requires metal_available=True")
            if self.backend == "mlx":
                _name(self.mlx_version, "mlx_version")
                _name(self.mlx_metal_version, "mlx_metal_version")
            else:
                _name(self.pytorch_version, "pytorch_version")
        passing = [item for item in batches if item.passed]
        chosen = min(passing, key=lambda item: (item.device_median_seconds, item.batch_size)) if passing else None
        if not self.available and chosen is not None:
            raise ValueError("unavailable benchmark cannot select a passing backend")
        if chosen is None:
            if self.selected_batch_size is not None or self.selected_device_median_seconds is not None:
                raise ValueError("failed backend cannot select a batch")
            fallback = _name(self.fallback_reason, "fallback_reason")
            if self.available:
                if fallback != "no_passing_batch":
                    raise ValueError("available failed benchmark fallback_reason must be no_passing_batch")
            elif re.fullmatch(rf"{self.backend}_unavailable:[a-z][a-z0-9_]*", fallback) is None:
                raise ValueError("unavailable benchmark fallback_reason must be canonical")
        else:
            if isinstance(self.selected_batch_size, bool) or not isinstance(self.selected_batch_size, Integral) or int(self.selected_batch_size) != chosen.batch_size:
                raise ValueError("selected batch does not match fastest passing batch")
            if self.selected_device_median_seconds is None or not isclose(_finite(self.selected_device_median_seconds, "selected_device_median_seconds", positive=True), chosen.device_median_seconds, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("selected device median does not match chosen batch")
            if self.fallback_reason is not None:
                raise ValueError("passing backend cannot have fallback reason")
            object.__setattr__(self, "selected_batch_size", chosen.batch_size)
            object.__setattr__(self, "selected_device_median_seconds", chosen.device_median_seconds)

    @property
    def passed(self) -> bool:
        return self.selected_batch_size is not None

    def to_payload(self) -> dict[str, object]:
        return {"backend": self.backend, "available": self.available, "batches": [batch.to_payload() for batch in self.batches],
                "python_version": self.python_version, "numpy_version": self.numpy_version, "device": self.device,
                "device_info": self.device_info, "metal_available": self.metal_available, "protocol": self.protocol,
                "seed": self.seed, "mlx_version": self.mlx_version, "mlx_metal_version": self.mlx_metal_version,
                "pytorch_version": self.pytorch_version, "selected_batch_size": self.selected_batch_size,
                "selected_device_median_seconds": self.selected_device_median_seconds,
                "fallback_reason": self.fallback_reason, "passed": self.passed}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()

    @classmethod
    def unavailable(cls, backend: str, reason: str, **provenance: object) -> "BackendBenchmarkEvidence":
        """Record an honest optional-framework failure with no invented work."""

        raw_reason = _name(reason, "reason")
        suffix = raw_reason.split(":", 1)[-1]
        if re.fullmatch(r"[a-z][a-z0-9_]*", suffix) is None:
            raise ValueError("unavailable reason must be a canonical token")
        kwargs: dict[str, object] = {
            "backend": backend, "available": False, "batches": (), "python_version": provenance.pop("python_version", None),
            "numpy_version": provenance.pop("numpy_version", None), "device": provenance.pop("device", None),
            "device_info": provenance.pop("device_info", None), "metal_available": provenance.pop("metal_available", None),
            "protocol": FROZEN_CONFIG.backend_protocol, "selected_batch_size": None,
            "selected_device_median_seconds": None, "fallback_reason": f"{backend}_unavailable:{suffix}",
            "mlx_version": provenance.pop("mlx_version", None), "mlx_metal_version": provenance.pop("mlx_metal_version", None),
            "pytorch_version": provenance.pop("pytorch_version", None),
        }
        if provenance:
            raise ValueError("unknown unavailable backend provenance")
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class BackendDecision:
    """Fail-closed selection retaining both accelerator benchmark records."""

    mlx: BackendBenchmarkEvidence
    mps: BackendBenchmarkEvidence
    backend: str
    adopted: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.mlx, BackendBenchmarkEvidence) or self.mlx.backend != "mlx":
            raise ValueError("mlx benchmark evidence is required")
        if not isinstance(self.mps, BackendBenchmarkEvidence) or self.mps.backend != "mps":
            raise ValueError("mps benchmark evidence is required")
        if self.mlx.available and self.mps.available:
            mlx_batch, mps_batch = self.mlx.batches[0], self.mps.batches[0]
            if (mlx_batch.fixture_input_sha256 != mps_batch.fixture_input_sha256
                    or mlx_batch.fixture_output_sha256 != mps_batch.fixture_output_sha256):
                raise ValueError("available backend fixture digest pairs must match before selection")
        choices = [item for item in (self.mlx, self.mps) if item.passed]
        expected = "cpu" if not choices else min(choices, key=lambda item: (item.selected_device_median_seconds, item.backend)).backend
        if self.backend != expected or type(self.adopted) is not bool or self.adopted != (expected != "cpu"):
            raise ValueError("backend decision is inconsistent with retained evidence")
        expected_reason = "selected_fastest_passing_backend" if expected != "cpu" else "no_accelerator_passed"
        if self.reason != expected_reason:
            raise ValueError("backend decision reason is inconsistent with selected backend")

    @property
    def outer_worker_count(self) -> int:
        return 1 if self.adopted else FROZEN_CONFIG.cpu_workers

    @property
    def serial_outer_execution(self) -> bool:
        return self.adopted

    @property
    def selected_batch_size(self) -> int | None:
        return None if self.backend == "cpu" else self._selected_benchmark.selected_batch_size

    @property
    def selected_device_median_seconds(self) -> float | None:
        return None if self.backend == "cpu" else self._selected_benchmark.selected_device_median_seconds

    @property
    def _selected_benchmark(self) -> BackendBenchmarkEvidence:
        return self.mlx if self.backend == "mlx" else self.mps

    def to_payload(self) -> dict[str, object]:
        return {"mlx": self.mlx.to_payload(), "mps": self.mps.to_payload(), "backend": self.backend,
                "adopted": self.adopted, "reason": self.reason, "selected_batch_size": self.selected_batch_size,
                "selected_device_median_seconds": self.selected_device_median_seconds,
                "outer_worker_count": self.outer_worker_count, "serial_outer_execution": self.serial_outer_execution}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "BackendDecision":
        """Strictly reconstruct the persisted decision, rejecting derived junk."""

        expected = {"mlx", "mps", "backend", "adopted", "reason", "selected_batch_size",
                    "selected_device_median_seconds", "outer_worker_count", "serial_outer_execution"}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("backend decision payload keys are invalid")

        def benchmark(raw: object) -> BackendBenchmarkEvidence:
            if not isinstance(raw, Mapping):
                raise ValueError("backend benchmark payload is invalid")
            item = dict(raw)
            benchmark_keys = {"backend", "available", "batches", "python_version", "numpy_version", "device",
                              "device_info", "metal_available", "protocol", "seed", "mlx_version", "mlx_metal_version",
                              "pytorch_version", "selected_batch_size", "selected_device_median_seconds", "fallback_reason", "passed"}
            if set(item) != benchmark_keys or not isinstance(item.get("batches"), list):
                raise ValueError("backend benchmark payload keys are invalid")
            batches = item.pop("batches")
            item.pop("passed")
            try:
                return BackendBenchmarkEvidence(
                    batches=tuple(BackendBatchEvidence(**dict(batch)) for batch in batches if isinstance(batch, Mapping)),
                    **item,
                )
            except (TypeError, ValueError) as error:
                raise ValueError("backend benchmark payload is invalid") from error

        decision = cls(benchmark(payload["mlx"]), benchmark(payload["mps"]),
                       payload["backend"], payload["adopted"], payload["reason"])
        if decision.to_payload() != dict(payload):
            raise ValueError("backend decision derived fields are invalid")
        return decision

    @classmethod
    def cpu_fallback(cls) -> "BackendDecision":
        return cls(BackendBenchmarkEvidence.unavailable("mlx", "no_accelerator_passed"), BackendBenchmarkEvidence.unavailable("mps", "no_accelerator_passed"), "cpu", False, "no_accelerator_passed")


@dataclass(frozen=True)
class Exp436RunSummary:
    run_identity: str
    status: str
    total_datasets: int
    completed_datasets: int
    successful_datasets: int
    abstained_datasets: int
    failed_datasets: int
    candidate_count: int
    backend: BackendDecision
    nonfinite_train_rows: int = 0
    nonfinite_test_rows: int = 0
    prediction_sha256: str | None = None
    manifest_sha256: str | None = None
    artifact_sha256: str | None = None
    review_only: bool = True
    operational_alert_eligible: bool = False

    def __post_init__(self) -> None:
        _name(self.run_identity, "run_identity")
        if self.status not in {"initializing", "running", "interrupted", "failed", "checkpoint_complete", "complete_review_only"}:
            raise ValueError("status is invalid")
        for name in ("total_datasets", "completed_datasets", "successful_datasets", "abstained_datasets", "failed_datasets", "candidate_count", "nonfinite_train_rows", "nonfinite_test_rows"):
            _integer(getattr(self, name), name)
        if self.completed_datasets != self.successful_datasets + self.abstained_datasets + self.failed_datasets or self.completed_datasets > self.total_datasets:
            raise ValueError("dataset counts do not conserve")
        if self.candidate_count > FROZEN_CONFIG.global_candidate_budget:
            raise ValueError("candidate count exceeds frozen budget")
        if not isinstance(self.backend, BackendDecision):
            raise ValueError("backend must be a BackendDecision")
        if self.review_only is not True or self.operational_alert_eligible is not False:
            raise ValueError("Exp436 publication must remain review-only")
        hashes = (self.prediction_sha256, self.manifest_sha256, self.artifact_sha256)
        if self.status in {"checkpoint_complete", "complete_review_only"}:
            if self.total_datasets != FROZEN_CONFIG.expected_datasets or self.completed_datasets != self.total_datasets or self.failed_datasets != 0:
                raise ValueError("terminal summary must conserve all frozen datasets without failures")
        if self.status == "complete_review_only":
            if any(value is None for value in hashes):
                raise ValueError("complete publication requires all hashes")
        elif any(value is not None for value in hashes):
            raise ValueError("only complete_review_only publication may carry hashes")
        for name in ("prediction_sha256", "manifest_sha256", "artifact_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _digest(value, name))

    def to_payload(self) -> dict[str, object]:
        return {"run_identity": self.run_identity, "status": self.status, "total_datasets": self.total_datasets,
                "completed_datasets": self.completed_datasets, "successful_datasets": self.successful_datasets,
                "abstained_datasets": self.abstained_datasets, "failed_datasets": self.failed_datasets,
                "candidate_count": self.candidate_count, "backend": self.backend.to_payload(),
                "nonfinite_train_rows": self.nonfinite_train_rows, "nonfinite_test_rows": self.nonfinite_test_rows,
                "prediction_sha256": self.prediction_sha256, "manifest_sha256": self.manifest_sha256,
                "artifact_sha256": self.artifact_sha256, "review_only": self.review_only,
                "operational_alert_eligible": self.operational_alert_eligible}

    def to_dict(self) -> dict[str, object]:
        return self.to_payload()


__all__ = [
    "RawValueSeries", "MissingSafeRawDataset", "ExcludedTrainRowAudit", "PreparedTrainGroup", "UnavailableTrainGroup",
    "TrainGroup", "TrainPreparation", "ChannelCalibration", "CalibratedTrainGroup", "CalibratedTrainGroupResult",
    "InstanceScoreEvidence", "Exp436Candidate", "Exp436DatasetResult", "BackendBatchEvidence",
    "BackendBenchmarkEvidence", "BackendDecision", "Exp436RunSummary",
]
