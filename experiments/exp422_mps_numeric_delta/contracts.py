"""Immutable evidence records for the Exp422 CPU/MPS parity diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real

from .config import FROZEN_CONFIG


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _sha256(value: object, name: str) -> str:
    if not (isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class BatchKey:
    batch_size: int
    model_seed: int
    perturbation_seed: int

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size not in FROZEN_CONFIG.candidate_batch_sizes:
            raise ValueError("batch_size is not preregistered for Exp422")
        if type(self.model_seed) is not int or self.model_seed not in FROZEN_CONFIG.model_seeds:
            raise ValueError("model_seed is not frozen by Exp417")
        if type(self.perturbation_seed) is not int or self.perturbation_seed not in FROZEN_CONFIG.perturbation_seeds:
            raise ValueError("perturbation_seed is not frozen by Exp417")


@dataclass(frozen=True)
class TensorDelta:
    name: str
    finite: bool
    max_absolute_error: float
    mean_absolute_error: float
    max_normalized_error: float
    max_ulp: int
    cpu_sha256: str
    mps_sha256: str
    failing_index: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if type(self.finite) is not bool:
            raise ValueError("finite must be a bool")
        for name in ("max_absolute_error", "mean_absolute_error", "max_normalized_error"):
            _finite(getattr(self, name), name, minimum=0.0)
        if type(self.max_ulp) is not int or self.max_ulp < 0:
            raise ValueError("max_ulp must be a nonnegative integer")
        _sha256(self.cpu_sha256, "cpu_sha256")
        _sha256(self.mps_sha256, "mps_sha256")
        if self.failing_index is not None and (type(self.failing_index) is not int or self.failing_index < 0):
            raise ValueError("failing_index must be a nonnegative integer or None")

    @property
    def passes(self) -> bool:
        return self.finite and self.max_normalized_error <= FROZEN_CONFIG.max_normalized_error


@dataclass(frozen=True)
class TimingEvidence:
    backend: str
    transfer_seconds: float
    compute_seconds: float
    synchronize_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "mps"}:
            raise ValueError("backend must be cpu or mps")
        for name in ("transfer_seconds", "compute_seconds", "synchronize_seconds", "total_seconds"):
            _finite(getattr(self, name), name, minimum=0.0)
        expected_total = self.transfer_seconds + self.compute_seconds + self.synchronize_seconds
        if not isclose(self.total_seconds, expected_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("total_seconds must include transfer, compute, and synchronization")


@dataclass(frozen=True)
class CoreParityReport:
    batch_key: BatchKey
    tensor_deltas: tuple[TensorDelta, ...]
    cpu_timing: TimingEvidence
    mps_timing: TimingEvidence
    candidate_indices_match: bool

    def __post_init__(self) -> None:
        if not isinstance(self.batch_key, BatchKey):
            raise ValueError("batch_key must be a BatchKey")
        if not isinstance(self.tensor_deltas, tuple) or not self.tensor_deltas or any(not isinstance(delta, TensorDelta) for delta in self.tensor_deltas):
            raise ValueError("tensor_deltas must be a nonempty tuple of TensorDelta records")
        if not isinstance(self.cpu_timing, TimingEvidence) or self.cpu_timing.backend != "cpu":
            raise ValueError("cpu_timing must be CPU TimingEvidence")
        if not isinstance(self.mps_timing, TimingEvidence) or self.mps_timing.backend != "mps":
            raise ValueError("mps_timing must be MPS TimingEvidence")
        if type(self.candidate_indices_match) is not bool:
            raise ValueError("candidate_indices_match must be a bool")

    @property
    def speedup(self) -> float:
        if self.mps_timing.total_seconds <= 0.0:
            return 0.0
        return self.cpu_timing.total_seconds / self.mps_timing.total_seconds

    @property
    def passes(self) -> bool:
        return (
            self.candidate_indices_match
            and all(delta.passes for delta in self.tensor_deltas)
            and self.speedup >= FROZEN_CONFIG.min_speedup
        )


@dataclass(frozen=True)
class StageResult:
    state: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state not in {"passed", "failed", "skipped"}:
            raise ValueError("state must be passed, failed, or skipped")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("reason must be a nonempty string or None")
        if self.state == "passed" and self.reason is not None:
            raise ValueError("passed stage cannot have a failure reason")
        if self.state in {"failed", "skipped"} and self.reason is None:
            raise ValueError("failed or skipped stage requires a reason")

    @classmethod
    def passed(cls) -> StageResult:
        return cls("passed")

    @classmethod
    def failed(cls, reason: str) -> StageResult:
        return cls("failed", reason)

    @classmethod
    def skipped(cls, reason: str) -> StageResult:
        return cls("skipped", reason)
