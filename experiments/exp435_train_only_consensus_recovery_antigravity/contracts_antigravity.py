"""Immutable shared contracts for Exp435's TRAIN-only prediction path.

Only the two in-memory feature records contain NumPy arrays.  All published
records below them contain tuples and scalar values so they can be canonicalised
without depending on NumPy's mutable object model.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isclose
import re
from numbers import Real
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .config_antigravity import FROZEN_CONFIG

if TYPE_CHECKING:  # Boundary-only imports; prediction never constructs them here.
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        FrozenLaneSnapshot,
        RawSeries,
        RawTestDataset,
        RawTrainDataset,
    )


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_STATUSES = frozenset({"scored", "abstained", "failed"})
_CHANNELS = frozenset({"prototype", "gradient", "acf"})
_UNAVAILABLE_GROUP_REASONS = frozenset(
    {
        "insufficient_train_rows",
        "insufficient_series_length",
        "nonfinite_train",
        "zero_scale",
    }
)


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _optional_finite(value: object, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _tuple(values: Sequence[Any], label: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return tuple(values)


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return sha256(encoded).hexdigest()


def _readonly_matrix(value: object, label: str, *, rows: int, columns: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape != (rows, columns):
        raise ValueError(f"{label} must have shape ({rows}, {columns})")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    copied = np.array(array, dtype=np.float32, order="C", copy=True)
    copied.flags.writeable = False
    return copied


def _readonly_optional_matrix(
    value: object | None, label: str, *, rows: int, columns: int
) -> np.ndarray | None:
    return None if value is None else _readonly_matrix(value, label, rows=rows, columns=columns)


@dataclass(frozen=True)
class PreparedTrainGroup:
    """An exact-length TRAIN group before channel calibration."""

    dataset: str
    length: int
    instance_indices: tuple[int, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.length, "length", minimum=FROZEN_CONFIG.min_series_length)
        indices = tuple(_integer(index, "instance index") for index in _tuple(self.instance_indices, "instance_indices"))
        if not indices or indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("instance_indices must be sorted, unique, and non-empty")
        object.__setattr__(self, "instance_indices", indices)
        object.__setattr__(
            self,
            "values",
            _readonly_matrix(self.values, "values", rows=len(indices), columns=self.length),
        )

    @property
    def reference_count(self) -> int:
        return len(self.instance_indices)


@dataclass(frozen=True)
class UnavailableTrainGroup:
    """Immutable evidence for an exact-length TRAIN group that abstains.

    This record intentionally contains no numerical array.  It preserves the
    affected TRAIN instance coordinates and the deterministic reason so a
    downstream scorer can abstain that exact length while continuing to score
    valid sibling lengths.
    """

    dataset: str
    length: int
    instance_indices: tuple[int, ...]
    reason: str
    identity_digest: str | None = None

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.length, "length", minimum=1)
        indices = tuple(
            _integer(index, "instance index")
            for index in _tuple(self.instance_indices, "instance_indices")
        )
        if not indices or indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("instance_indices must be sorted, unique, and non-empty")
        object.__setattr__(self, "instance_indices", indices)
        if not isinstance(self.reason, str) or self.reason not in _UNAVAILABLE_GROUP_REASONS:
            raise ValueError("reason is not preregistered")
        payload = self.to_payload(include_digest=False)
        expected_digest = _canonical_digest(payload)
        if self.identity_digest is None:
            digest = expected_digest
        else:
            digest = _digest(self.identity_digest, "identity_digest")
            if digest != expected_digest:
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
        """Alias for consumers that call all group identity evidence a digest."""

        assert self.identity_digest is not None
        return self.identity_digest

    @property
    def reference_digest(self) -> str:
        """Digest alias matching channel calibration terminology."""

        assert self.identity_digest is not None
        return self.identity_digest

    @property
    def identity_sha256(self) -> str:
        """Readable SHA256 alias for publication/checkpoint adapters."""

        assert self.identity_digest is not None
        return self.identity_digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "exp435.unavailable-train-group.v1",
            "dataset": self.dataset,
            "length": self.length,
            "instance_indices": list(self.instance_indices),
            "reason": self.reason,
        }
        if include_digest:
            payload["identity_digest"] = self.identity_digest
        return payload


TrainGroup = PreparedTrainGroup | UnavailableTrainGroup


@dataclass(frozen=True)
class ChannelCalibration:
    """TRAIN-only threshold and transform metadata for one score channel."""

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
        scores = tuple(_finite(score, "loo score") for score in _tuple(self.loo_scores, "loo_scores"))
        object.__setattr__(self, "loo_scores", scores)
        threshold = _optional_finite(self.threshold, "threshold")
        center = _optional_finite(self.center, "center")
        scale = _optional_finite(self.scale, "scale")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        _integer(self.reference_count, "reference_count")
        if self.reference_digest is not None:
            object.__setattr__(self, "reference_digest", _digest(self.reference_digest, "reference_digest"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _name(self.reason, "reason"))
        if self.available:
            if threshold is None or len(scores) < FROZEN_CONFIG.min_train_rows:
                raise ValueError("available calibration requires threshold and four LOO scores")
            if self.reference_count != len(scores):
                raise ValueError("available calibration reference_count must equal LOO score count")
            expected_threshold = float(
                np.quantile(
                    np.asarray(scores, dtype=np.float32),
                    FROZEN_CONFIG.train_quantile,
                    method=FROZEN_CONFIG.quantile_method,
                )
            )
            if threshold != expected_threshold:
                raise ValueError("available calibration threshold must equal higher quantile")
            if scale is None or scale <= FROZEN_CONFIG.numeric_floor:
                raise ValueError("available calibration requires a positive scale")
            if self.reason is not None:
                raise ValueError("available calibration cannot have an unavailable reason")
        else:
            if self.reason is None:
                raise ValueError("unavailable calibration requires a reason")
            if self.reference_count != 0:
                raise ValueError("unavailable calibration reference_count must be zero")
            if scores or threshold is not None or center is not None or scale is not None:
                raise ValueError("unavailable calibration cannot contain usable scores or transform")

    @classmethod
    def unavailable(cls, channel: str, reason: str) -> "ChannelCalibration":
        return cls(channel=channel, available=False, reason=reason)

    @classmethod
    def fixture(
        cls, channel: str, *, reference_count: int = 4, threshold: float = 1.0
    ) -> "ChannelCalibration":
        return cls(
            channel=channel,
            available=True,
            threshold=threshold,
            loo_scores=(0.0, 0.1, 0.2, threshold),
            center=0.0,
            scale=1.0,
            reference_count=reference_count,
            reference_digest="0" * 64,
        )

    def is_abnormal(self, score: object) -> bool:
        if not self.available or self.threshold is None:
            return False
        return _finite(score, "score") > self.threshold

    def to_payload(self) -> dict[str, object]:
        return {
            "channel": self.channel,
            "available": self.available,
            "threshold": self.threshold,
            "loo_scores": list(self.loo_scores),
            "center": self.center,
            "scale": self.scale,
            "reference_count": self.reference_count,
            "reference_digest": self.reference_digest,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CalibratedTrainGroup:
    """A TRAIN group with normalized channel arrays and frozen thresholds."""

    dataset: str
    length: int
    instance_indices: tuple[int, ...]
    values: np.ndarray
    gradients: np.ndarray | None
    acf_values: np.ndarray | None
    prototype: ChannelCalibration
    gradient: ChannelCalibration
    acf: ChannelCalibration

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.length, "length", minimum=FROZEN_CONFIG.min_series_length)
        indices = tuple(_integer(index, "instance index") for index in _tuple(self.instance_indices, "instance_indices"))
        if not indices or indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("instance_indices must be sorted, unique, and non-empty")
        object.__setattr__(self, "instance_indices", indices)
        rows = len(indices)
        object.__setattr__(self, "values", _readonly_matrix(self.values, "values", rows=rows, columns=self.length))
        for name, calibration in (
            ("prototype", self.prototype),
            ("gradient", self.gradient),
            ("acf", self.acf),
        ):
            if not isinstance(calibration, ChannelCalibration):
                raise ValueError(f"{name} must be a ChannelCalibration")
        if self.prototype.channel != "prototype" or self.gradient.channel != "gradient" or self.acf.channel != "acf":
            raise ValueError("calibration channels do not match group")
        if not self.prototype.available:
            raise ValueError("prototype calibration must be available")
        if self.prototype.reference_count != rows:
            raise ValueError("prototype reference_count must match group rows")
        object.__setattr__(
            self,
            "gradients",
            _readonly_optional_matrix(self.gradients, "gradients", rows=rows, columns=self.length),
        )
        if (self.gradients is None) != (not self.gradient.available):
            raise ValueError("gradient calibration availability must match gradient array")
        if self.gradients is not None and self.gradient.reference_count != rows:
            raise ValueError("gradient reference_count must match group rows")
        if self.length < FROZEN_CONFIG.min_acf_length and (self.acf_values is not None or self.acf.available):
            raise ValueError("ACF channel unavailable below min_acf_length=16")
        if (self.acf_values is None) != (not self.acf.available):
            raise ValueError("acf calibration availability must match acf array")
        if self.acf_values is not None and self.acf.reference_count != rows:
            raise ValueError("acf reference_count must match group rows")
        acf_columns = min(FROZEN_CONFIG.max_acf_lag, self.length // 4)
        if self.acf_values is not None:
            acf_array = np.asarray(self.acf_values, dtype=np.float32)
            if acf_array.ndim != 2 or acf_array.shape[0] != rows or acf_array.shape[1] != acf_columns:
                raise ValueError(f"acf must have shape ({rows}, {acf_columns})")
            if not np.isfinite(acf_array).all():
                raise ValueError("acf must contain finite values")
            acf_copy = np.array(acf_array, dtype=np.float32, order="C", copy=True)
            acf_copy.flags.writeable = False
            object.__setattr__(self, "acf_values", acf_copy)

    @property
    def acf_calibration(self) -> ChannelCalibration:
        """Readable alias used by scoring code while retaining ``acf`` arrays."""

        return self.acf

    @property
    def acf_vectors(self) -> np.ndarray | None:
        """Alias for the normalized ACF matrix."""

        return self.acf_values


CalibratedTrainGroupResult = CalibratedTrainGroup | UnavailableTrainGroup


@dataclass(frozen=True)
class InstanceScoreEvidence:
    """All finite scalar evidence and the deterministic consensus decision."""

    dataset: str
    instance_index: int
    length: int
    prototype_score: float | None = None
    prototype_threshold: float | None = None
    prototype_abnormal: bool = False
    gradient_score: float | None = None
    gradient_threshold: float | None = None
    gradient_abnormal: bool = False
    acf_score: float | None = None
    acf_threshold: float | None = None
    acf_abnormal: bool = False
    consensus_candidate: bool = False
    status: str = "scored"
    exclusion_reason: str | None = None
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.instance_index, "instance_index")
        if self.status not in _STATUSES:
            raise ValueError("status is invalid")
        # A length-one row can only be retained as explicit abstention/failure
        # evidence.  A scored row still requires the frozen minimum length, so
        # no short row can become a candidate through this exception.
        minimum_length = (
            1 if self.status in {"abstained", "failed"} else FROZEN_CONFIG.min_series_length
        )
        _integer(self.length, "length", minimum=minimum_length)
        for channel, score, threshold, abnormal in (
            ("prototype", self.prototype_score, self.prototype_threshold, self.prototype_abnormal),
            ("gradient", self.gradient_score, self.gradient_threshold, self.gradient_abnormal),
            ("acf", self.acf_score, self.acf_threshold, self.acf_abnormal),
        ):
            if type(abnormal) is not bool:
                raise ValueError(f"{channel}_abnormal must be boolean")
            score_value = _optional_finite(score, f"{channel}_score")
            threshold_value = _optional_finite(threshold, f"{channel}_threshold")
            object.__setattr__(self, f"{channel}_score", score_value)
            object.__setattr__(self, f"{channel}_threshold", threshold_value)
            if (score_value is None) != (threshold_value is None):
                raise ValueError(f"{channel} score and threshold must be both present or absent")
            expected_abnormal = score_value is not None and score_value > threshold_value  # type: ignore[operator]
            if abnormal != expected_abnormal:
                raise ValueError(f"{channel} abnormal decision is inconsistent")
        if type(self.consensus_candidate) is not bool:
            raise ValueError("consensus_candidate must be boolean")
        expected_consensus = self.prototype_abnormal and (self.gradient_abnormal or self.acf_abnormal)
        if self.consensus_candidate != expected_consensus:
            raise ValueError("consensus candidate is inconsistent with channel decisions")
        if self.status != "scored" and self.consensus_candidate:
            raise ValueError("abstained or failed evidence cannot be a consensus candidate")
        if self.exclusion_reason is not None:
            object.__setattr__(self, "exclusion_reason", _name(self.exclusion_reason, "exclusion_reason"))
        payload = self.to_payload(include_digest=False)
        expected_digest = _canonical_digest(payload)
        if self.evidence_digest is None:
            digest = expected_digest
        else:
            digest = _digest(self.evidence_digest, "evidence_digest")
            if digest != expected_digest:
                raise ValueError("evidence_digest does not match canonical payload")
        object.__setattr__(self, "evidence_digest", digest)

    @classmethod
    def fixture(
        cls,
        *,
        dataset: str = "Demo",
        instance_index: int = 3,
        length: int = 64,
        prototype_abnormal: bool = False,
        gradient_abnormal: bool = False,
        acf_abnormal: bool = False,
        status: str = "scored",
        exclusion_reason: str | None = None,
    ) -> "InstanceScoreEvidence":
        def values(abnormal: bool) -> tuple[float, float, bool]:
            return (2.0 if abnormal else 0.0, 1.0, abnormal)

        prototype_score, prototype_threshold, _ = values(prototype_abnormal)
        gradient_score, gradient_threshold, _ = values(gradient_abnormal)
        acf_score, acf_threshold, _ = values(acf_abnormal)
        return cls(
            dataset=dataset,
            instance_index=instance_index,
            length=length,
            prototype_score=prototype_score,
            prototype_threshold=prototype_threshold,
            prototype_abnormal=prototype_abnormal,
            gradient_score=gradient_score,
            gradient_threshold=gradient_threshold,
            gradient_abnormal=gradient_abnormal,
            acf_score=acf_score,
            acf_threshold=acf_threshold,
            acf_abnormal=acf_abnormal,
            consensus_candidate=prototype_abnormal and (gradient_abnormal or acf_abnormal),
            status=status,
            exclusion_reason=exclusion_reason,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "dataset": self.dataset,
            "instance_index": self.instance_index,
            "length": self.length,
            "prototype_score": self.prototype_score,
            "prototype_threshold": self.prototype_threshold,
            "prototype_abnormal": self.prototype_abnormal,
            "gradient_score": self.gradient_score,
            "gradient_threshold": self.gradient_threshold,
            "gradient_abnormal": self.gradient_abnormal,
            "acf_score": self.acf_score,
            "acf_threshold": self.acf_threshold,
            "acf_abnormal": self.acf_abnormal,
            "consensus_candidate": self.consensus_candidate,
            "status": self.status,
            "exclusion_reason": self.exclusion_reason,
        }
        if include_digest:
            payload["evidence_digest"] = self.evidence_digest
        return payload


@dataclass(frozen=True)
class Exp435Candidate:
    """A published new coordinate and the digest of its scalar evidence."""

    dataset: str
    instance_index: int
    length: int
    evidence_digest: str

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.instance_index, "instance_index")
        _integer(self.length, "length", minimum=FROZEN_CONFIG.min_series_length)
        object.__setattr__(self, "evidence_digest", _digest(self.evidence_digest, "evidence_digest"))

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.dataset, self.instance_index

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "instance_index": self.instance_index,
            "length": self.length,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class Exp435DatasetResult:
    """One dataset's ordered evidence and candidates."""

    dataset: str
    status: str
    test_size: int
    candidates: tuple[Exp435Candidate, ...] = ()
    evidence: tuple[InstanceScoreEvidence, ...] = ()
    abstention_reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        if self.status not in _STATUSES:
            raise ValueError("status is invalid")
        _integer(self.test_size, "test_size")
        candidates = _tuple(self.candidates, "candidates")
        if any(not isinstance(item, Exp435Candidate) for item in candidates):
            raise ValueError("candidates must contain Exp435Candidate records")
        candidate_indices = tuple(item.instance_index for item in candidates)
        if any(item.dataset != self.dataset for item in candidates):
            raise ValueError("candidate dataset does not match result dataset")
        if candidate_indices != tuple(sorted(candidate_indices)):
            raise ValueError("candidate indices must be sorted")
        if len(candidate_indices) != len(set(candidate_indices)):
            raise ValueError("candidate indices contain a duplicate")
        if any(index >= self.test_size for index in candidate_indices):
            raise ValueError("candidate index is outside TEST bounds")
        object.__setattr__(self, "candidates", tuple(candidates))
        evidence = _tuple(self.evidence, "evidence")
        if any(not isinstance(item, InstanceScoreEvidence) for item in evidence):
            raise ValueError("evidence must contain InstanceScoreEvidence records")
        evidence_indices = tuple(item.instance_index for item in evidence)
        if any(item.dataset != self.dataset for item in evidence):
            raise ValueError("evidence dataset does not match result dataset")
        if evidence_indices != tuple(sorted(evidence_indices)) or len(evidence_indices) != len(set(evidence_indices)):
            raise ValueError("evidence indices must be sorted and unique")
        if any(index >= self.test_size for index in evidence_indices):
            raise ValueError("evidence index is outside TEST bounds")
        object.__setattr__(self, "evidence", tuple(evidence))
        evidence_by_coordinate = {
            (item.dataset, item.instance_index): item for item in evidence
        }
        for candidate in candidates:
            matched = evidence_by_coordinate.get(candidate.coordinate)
            if matched is None:
                raise ValueError("candidate coordinate has no matching evidence")
            if matched.status != "scored" or not matched.consensus_candidate:
                raise ValueError("candidate evidence must be a scored consensus")
            if matched.exclusion_reason is not None:
                raise ValueError("candidate evidence has an exclusion reason")
            if candidate.length != matched.length:
                raise ValueError("candidate length does not match evidence")
            if candidate.evidence_digest != matched.evidence_digest:
                raise ValueError("candidate evidence digest mismatch")
        if self.status in {"abstained", "failed"} and candidates:
            raise ValueError("abstained or failed result cannot contain candidates")
        if self.status == "abstained":
            object.__setattr__(self, "abstention_reason", _name(self.abstention_reason, "abstention_reason"))
            if self.error is not None:
                raise ValueError("abstained result cannot contain an error")
        elif self.status == "failed":
            object.__setattr__(self, "error", _name(self.error, "error"))
            if self.abstention_reason is not None:
                raise ValueError("failed result cannot contain an abstention reason")
        else:
            if self.abstention_reason is not None or self.error is not None:
                raise ValueError("scored result cannot contain failure metadata")

    @classmethod
    def fixture(
        cls,
        *,
        dataset: str = "Demo",
        status: str = "scored",
        test_size: int = 64,
        candidates: tuple[Exp435Candidate, ...] = (),
        evidence: tuple[InstanceScoreEvidence, ...] = (),
    ) -> "Exp435DatasetResult":
        kwargs: dict[str, object] = {
            "dataset": dataset,
            "status": status,
            "test_size": test_size,
            "candidates": candidates,
            "evidence": evidence,
        }
        if status == "abstained":
            kwargs["abstention_reason"] = "fixture_abstention"
        elif status == "failed":
            kwargs["error"] = "fixture_failure"
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "status": self.status,
            "test_size": self.test_size,
            "candidates": [item.to_payload() for item in self.candidates],
            "evidence": [item.to_payload() for item in self.evidence],
            "abstention_reason": self.abstention_reason,
            "error": self.error,
        }


@dataclass(frozen=True)
class BackendBatchEvidence:
    """Immutable parity, finite-output, and timing evidence for one batch.

    A batch that fails one gate must not erase a sibling batch that passes.  A
    ``BackendDecision`` retains all these records and selects at most one for
    adoption, making the batch-level gate auditable after readiness.
    """

    batch_size: int
    finite_outputs: bool
    candidate_identity_match: bool
    parity_passed: bool
    normalized_error_max: float | None
    absolute_error_max: float | None
    speed_passed: bool
    cpu_median_seconds: float | None
    mlx_median_seconds: float | None
    speedup: float | None
    reason: str

    def __post_init__(self) -> None:
        _integer(self.batch_size, "batch_size", minimum=1)
        for name in (
            "finite_outputs",
            "candidate_identity_match",
            "parity_passed",
            "speed_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("normalized_error_max", "absolute_error_max", "cpu_median_seconds", "mlx_median_seconds", "speedup"):
            value = _optional_finite(getattr(self, name), name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        medians_present = self.cpu_median_seconds is not None or self.mlx_median_seconds is not None
        if medians_present:
            if self.cpu_median_seconds is None or self.mlx_median_seconds is None:
                raise ValueError("CPU and MLX batch medians must be supplied together")
            if self.cpu_median_seconds <= 0 or self.mlx_median_seconds <= 0:
                raise ValueError("CPU and MLX batch medians must be positive")
            if self.speedup is None:
                raise ValueError("batch speedup is required with medians")
            if not isclose(
                self.speedup,
                self.cpu_median_seconds / self.mlx_median_seconds,
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError("batch speedup does not match medians")
        elif self.speedup is not None:
            raise ValueError("batch speedup requires CPU and MLX medians")
        if self.parity_passed:
            if not self.finite_outputs or not self.candidate_identity_match:
                raise ValueError("passed batch parity requires finite matching output")
            if self.normalized_error_max is None or self.normalized_error_max > FROZEN_CONFIG.score_tolerance:
                raise ValueError("passed batch normalized error exceeds 1e-5")
            if self.absolute_error_max is None or self.absolute_error_max > FROZEN_CONFIG.score_tolerance:
                raise ValueError("passed batch absolute error exceeds 1e-5")
        if self.speed_passed and (self.speedup is None or self.speedup < FROZEN_CONFIG.mlx_min_speedup):
            raise ValueError("passed batch speed gate is below 1.1")
        object.__setattr__(self, "reason", _name(self.reason, "reason"))

    @property
    def adopted(self) -> bool:
        return self.parity_passed and self.speed_passed

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "finite_outputs": self.finite_outputs,
            "candidate_identity_match": self.candidate_identity_match,
            "parity_passed": self.parity_passed,
            "normalized_error_max": self.normalized_error_max,
            "absolute_error_max": self.absolute_error_max,
            "speed_passed": self.speed_passed,
            "cpu_median_seconds": self.cpu_median_seconds,
            "mlx_median_seconds": self.mlx_median_seconds,
            "speedup": self.speedup,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BackendDecision:
    """The deterministic CPU/MLX adoption result and parity evidence."""

    backend: str
    adopted: bool
    smoke_seed: int = FROZEN_CONFIG.mlx_smoke_seed
    finite_outputs: bool = True
    candidate_identity_match: bool = True
    parity_passed: bool = True
    normalized_error_max: float | None = 0.0
    absolute_error_max: float | None = 0.0
    speedup: float | None = None
    batch_size: int | None = None
    selected_batch_size: int | None = None
    cpu_median_seconds: float | None = None
    mlx_median_seconds: float | None = None
    mlx_version: str | None = None
    mlx_metal_version: str | None = None
    device: str | None = None
    device_info: str | None = None
    metal_available: bool | None = None
    timings: tuple[tuple[int, float, float], ...] = ()
    batch_evidence: tuple[BackendBatchEvidence, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "mlx"}:
            raise ValueError("backend is invalid")
        if type(self.adopted) is not bool:
            raise ValueError("adopted must be boolean")
        _integer(self.smoke_seed, "smoke_seed")
        for name in ("finite_outputs", "candidate_identity_match", "parity_passed"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("normalized_error_max", "absolute_error_max", "speedup", "cpu_median_seconds", "mlx_median_seconds"):
            value = _optional_finite(getattr(self, name), name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.batch_size is not None:
            _integer(self.batch_size, "batch_size", minimum=1)
        if self.selected_batch_size is not None:
            _integer(self.selected_batch_size, "selected_batch_size", minimum=1)
        if self.batch_size is not None and self.selected_batch_size is not None and self.batch_size != self.selected_batch_size:
            raise ValueError("batch_size and selected_batch_size disagree")
        selected_batch_size = self.selected_batch_size if self.selected_batch_size is not None else self.batch_size
        object.__setattr__(self, "batch_size", selected_batch_size)
        object.__setattr__(self, "selected_batch_size", selected_batch_size)
        if self.metal_available is not None and type(self.metal_available) is not bool:
            raise ValueError("metal_available must be boolean or None")
        for name in ("mlx_version", "mlx_metal_version", "device", "device_info"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _name(value, name))
        timings = []
        timing_batches: set[int] = set()
        for item in _tuple(self.timings, "timings"):
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError("timings must be (batch_size, cpu_seconds, mlx_seconds) tuples")
            batch, cpu_seconds, mlx_seconds = item
            _integer(batch, "timing batch size", minimum=1)
            if batch in timing_batches:
                raise ValueError("timings contain duplicate batch size")
            timing_batches.add(batch)
            timings.append((batch, _finite(cpu_seconds, "cpu timing"), _finite(mlx_seconds, "mlx timing")))
        object.__setattr__(self, "timings", tuple(timings))
        batch_evidence = []
        seen_batches: set[int] = set()
        for item in _tuple(self.batch_evidence, "batch_evidence"):
            if not isinstance(item, BackendBatchEvidence):
                raise ValueError("batch_evidence must contain BackendBatchEvidence")
            if item.batch_size in seen_batches:
                raise ValueError("batch_evidence contains duplicate batch size")
            seen_batches.add(item.batch_size)
            batch_evidence.append(item)
        object.__setattr__(self, "batch_evidence", tuple(batch_evidence))
        object.__setattr__(self, "reason", _name(self.reason, "reason"))
        medians_present = self.cpu_median_seconds is not None or self.mlx_median_seconds is not None
        if medians_present:
            if self.cpu_median_seconds is None or self.mlx_median_seconds is None:
                raise ValueError("CPU and MLX medians must be supplied together")
            if self.cpu_median_seconds <= 0 or self.mlx_median_seconds <= 0:
                raise ValueError("CPU and MLX medians must be positive")
            if self.speedup is None:
                raise ValueError("speedup is required with benchmark medians")
            recorded_speedup = self.cpu_median_seconds / self.mlx_median_seconds
            if not isclose(self.speedup, recorded_speedup, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("declared speedup does not match recorded medians")
            if self.selected_batch_size is None:
                raise ValueError("selected batch size is required with benchmark medians")
            selected_timings = [item for item in self.timings if item[0] == self.selected_batch_size]
            if not selected_timings:
                raise ValueError("selected batch timing is missing")
            if len(selected_timings) != 1:
                raise ValueError("duplicate selected batch timing entries")
            _, selected_cpu, selected_mlx = selected_timings[0]
            if not isclose(selected_cpu, self.cpu_median_seconds, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("selected timing CPU median does not match top-level median")
            if not isclose(selected_mlx, self.mlx_median_seconds, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("selected timing MLX median does not match top-level median")
        elif self.speedup is not None or self.selected_batch_size is not None or self.timings:
            raise ValueError("benchmark evidence requires CPU/MLX medians")
        if self.adopted and self.backend != "mlx":
            raise ValueError("adopted backend must be mlx")
        if self.adopted:
            if self.smoke_seed != FROZEN_CONFIG.mlx_smoke_seed:
                raise ValueError("adopted MLX smoke seed must equal frozen seed 43501")
            if not self.finite_outputs:
                raise ValueError("adopted MLX output must be finite")
            if not self.candidate_identity_match:
                raise ValueError("adopted MLX candidate identity mismatch")
            if not self.parity_passed:
                raise ValueError("adopted MLX parity gate failed")
            if self.normalized_error_max is None or self.normalized_error_max > FROZEN_CONFIG.score_tolerance:
                raise ValueError("adopted MLX normalized error exceeds 1e-5")
            if self.absolute_error_max is None or self.absolute_error_max > FROZEN_CONFIG.score_tolerance:
                raise ValueError("adopted MLX absolute error exceeds 1e-5")
            recorded_speedup = self.cpu_median_seconds / self.mlx_median_seconds  # type: ignore[operator]
            if recorded_speedup < FROZEN_CONFIG.mlx_min_speedup:
                raise ValueError("adopted MLX speedup from recorded medians must be at least 1.1")
            if self.mlx_version is None or self.mlx_metal_version is None or self.device is None:
                raise ValueError("adopted MLX requires mlx_version, mlx_metal_version, and device")
            if self.device_info is None:
                raise ValueError("adopted MLX requires device_info")
            if self.metal_available is not True:
                raise ValueError("adopted MLX requires available Metal")
            if self.selected_batch_size is None or not self.timings:
                raise ValueError("adopted MLX requires benchmark timings")
            if self.cpu_median_seconds is None or self.cpu_median_seconds <= 0:
                raise ValueError("adopted MLX requires positive CPU timing")
            if self.mlx_median_seconds is None or self.mlx_median_seconds <= 0:
                raise ValueError("adopted MLX requires positive MLX timing")
            if any(cpu_seconds <= 0 or mlx_seconds <= 0 for _, cpu_seconds, mlx_seconds in self.timings):
                raise ValueError("adopted MLX timings must be positive")
            if not any(item[0] == self.batch_size for item in self.timings):
                raise ValueError("adopted MLX batch size is absent from timings")
            if not self.batch_evidence:
                raise ValueError("adopted MLX requires batch evidence")
        if self.batch_evidence and self.selected_batch_size is not None:
            selected_evidence = [
                item for item in self.batch_evidence if item.batch_size == self.selected_batch_size
            ]
            if len(selected_evidence) != 1:
                raise ValueError("selected batch evidence is missing or duplicated")
            selected_record = selected_evidence[0]
            if self.adopted:
                if not selected_record.adopted:
                    raise ValueError("adopted decision selected batch evidence did not pass")
                if self.finite_outputs != selected_record.finite_outputs:
                    raise ValueError("adopted finite_outputs does not match selected batch evidence")
                if self.candidate_identity_match != selected_record.candidate_identity_match:
                    raise ValueError("adopted candidate_identity_match does not match selected batch evidence")
                if self.parity_passed != selected_record.parity_passed:
                    raise ValueError("adopted parity_passed does not match selected batch evidence")
                for name in (
                    "normalized_error_max",
                    "absolute_error_max",
                    "cpu_median_seconds",
                    "mlx_median_seconds",
                    "speedup",
                ):
                    if getattr(self, name) != getattr(selected_record, name):
                        raise ValueError(f"adopted {name} does not match selected batch evidence")
            evidence_by_batch = {item.batch_size: item for item in self.batch_evidence}
            for batch, cpu_seconds, mlx_seconds in self.timings:
                record = evidence_by_batch.get(batch)
                if record is None:
                    raise ValueError("timings contain batch without batch evidence")
                if record.cpu_median_seconds is None or record.mlx_median_seconds is None:
                    raise ValueError("timing has no batch medians")
                if record.cpu_median_seconds != cpu_seconds or record.mlx_median_seconds != mlx_seconds:
                    raise ValueError("timing does not match batch evidence")
            for record in self.batch_evidence:
                if record.cpu_median_seconds is None and record.mlx_median_seconds is None:
                    continue
                matching = [item for item in self.timings if item[0] == record.batch_size]
                if len(matching) != 1:
                    raise ValueError("batch evidence medians are missing from timings")

    @classmethod
    def cpu_fallback(
        cls,
        reason: str,
        *,
        mlx_version: str | None = None,
        mlx_metal_version: str | None = None,
        device: str | None = None,
        device_info: str | None = None,
        metal_available: bool | None = None,
        batch_evidence: tuple[BackendBatchEvidence, ...] = (),
    ) -> "BackendDecision":
        return cls(
            backend="cpu",
            adopted=False,
            smoke_seed=FROZEN_CONFIG.mlx_smoke_seed,
            finite_outputs=True,
            candidate_identity_match=True,
            parity_passed=False,
            normalized_error_max=None,
            absolute_error_max=None,
            mlx_version=mlx_version,
            mlx_metal_version=mlx_metal_version,
            device=device,
            device_info=device_info,
            metal_available=metal_available,
            batch_evidence=batch_evidence,
            reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "adopted": self.adopted,
            "smoke_seed": self.smoke_seed,
            "finite_outputs": self.finite_outputs,
            "candidate_identity_match": self.candidate_identity_match,
            "parity_passed": self.parity_passed,
            "normalized_error_max": self.normalized_error_max,
            "absolute_error_max": self.absolute_error_max,
            "speedup": self.speedup,
            "batch_size": self.batch_size,
            "selected_batch_size": self.selected_batch_size,
            "cpu_median_seconds": self.cpu_median_seconds,
            "mlx_median_seconds": self.mlx_median_seconds,
            "mlx_version": self.mlx_version,
            "mlx_metal_version": self.mlx_metal_version,
            "device": self.device,
            "device_info": self.device_info,
            "metal_available": self.metal_available,
            "timings": [list(item) for item in self.timings],
            "batch_evidence": [item.to_dict() for item in self.batch_evidence],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Exp435RunSummary:
    """Scalar run accounting suitable for progress and publication manifests."""

    run_identity: str
    status: str
    total_datasets: int
    completed_datasets: int
    successful_datasets: int
    abstained_datasets: int
    failed_datasets: int
    candidate_count: int
    backend: BackendDecision
    prediction_sha256: str | None = None
    manifest_sha256: str | None = None
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        _name(self.run_identity, "run_identity")
        if self.status not in {
            "initializing", "running", "interrupted", "complete", "complete_review_only", "failed"
        }:
            raise ValueError("status is invalid")
        values = (
            ("total_datasets", self.total_datasets),
            ("completed_datasets", self.completed_datasets),
            ("successful_datasets", self.successful_datasets),
            ("abstained_datasets", self.abstained_datasets),
            ("failed_datasets", self.failed_datasets),
            ("candidate_count", self.candidate_count),
        )
        for label, value in values:
            _integer(value, label)
        if not isinstance(self.backend, BackendDecision):
            raise ValueError("backend must be a BackendDecision")
        terminal_complete = self.status in {"complete", "complete_review_only"}
        if terminal_complete:
            if self.total_datasets != FROZEN_CONFIG.expected_datasets:
                raise ValueError("complete summary total must equal frozen population 1117")
            if self.completed_datasets != FROZEN_CONFIG.expected_datasets:
                raise ValueError("complete summary must process all 1117 datasets")
            if self.failed_datasets != 0 or self.successful_datasets + self.abstained_datasets != self.completed_datasets:
                raise ValueError("complete summary requires zero errors and conserved success/abstention counts")
            if self.candidate_count > FROZEN_CONFIG.global_candidate_budget:
                raise ValueError("complete summary candidate budget exceeded")
            if any(getattr(self, label) is None for label in ("prediction_sha256", "manifest_sha256", "artifact_sha256")):
                raise ValueError("complete summary requires prediction, manifest, and artifact hashes")
        else:
            if self.completed_datasets > self.total_datasets:
                raise ValueError("completed_datasets exceeds total_datasets")
            if self.successful_datasets + self.abstained_datasets + self.failed_datasets != self.completed_datasets:
                raise ValueError("dataset count conservation is invalid")
        for label in ("prediction_sha256", "manifest_sha256", "artifact_sha256"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, _digest(value, label))

    @property
    def remaining_datasets(self) -> int:
        return self.total_datasets - self.completed_datasets

    def to_payload(self) -> dict[str, object]:
        return {
            "run_identity": self.run_identity,
            "status": self.status,
            "total_datasets": self.total_datasets,
            "completed_datasets": self.completed_datasets,
            "successful_datasets": self.successful_datasets,
            "abstained_datasets": self.abstained_datasets,
            "failed_datasets": self.failed_datasets,
            "candidate_count": self.candidate_count,
            "backend": self.backend.to_dict(),
            "prediction_sha256": self.prediction_sha256,
            "manifest_sha256": self.manifest_sha256,
            "artifact_sha256": self.artifact_sha256,
        }


__all__ = [
    "PreparedTrainGroup",
    "ChannelCalibration",
    "CalibratedTrainGroup",
    "InstanceScoreEvidence",
    "Exp435Candidate",
    "Exp435DatasetResult",
    "BackendBatchEvidence",
    "BackendDecision",
    "Exp435RunSummary",
]
