"""Validated immutable provenance records for Exp421 acceleration gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from types import MappingProxyType

import numpy as np

from .config import Exp421Config, FROZEN_CONFIG


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _sha256(value: object, name: str) -> str:
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _readonly_indices(value: object, name: str, *, require_readonly: bool = False) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("int64") or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int64 array")
    if require_readonly and value.flags.writeable:
        raise ValueError(f"{name} must be read-only before serialization")
    if value.size and (np.any(value < 0) or np.any(value[1:] <= value[:-1])):
        raise ValueError(f"{name} must be sorted, unique, nonnegative indices")
    copied = np.array(value, dtype=np.int64, copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _no_test_token(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "test" in value.lower():
        raise ValueError(f"{name} must be a non-TEST boundary string")
    return value


@dataclass(frozen=True)
class BatchKey:
    batch_size: int
    model_seed: int
    perturbation_seed: int

    def __post_init__(self) -> None:
        if self.batch_size not in FROZEN_CONFIG.candidate_batch_sizes:
            raise ValueError("batch_size is not preregistered")
        if type(self.model_seed) is not int or self.model_seed not in FROZEN_CONFIG.frozen_model_seeds:
            raise ValueError("model_seed is not frozen")
        if type(self.perturbation_seed) is not int or self.perturbation_seed not in FROZEN_CONFIG.frozen_perturbation_seeds:
            raise ValueError("perturbation_seed is not frozen")


@dataclass(frozen=True)
class BackendMeasurement:
    batch_size: int
    backend: str
    transfer_seconds: float
    compute_seconds: float
    synchronize_seconds: float
    total_seconds: float
    finite: bool
    candidate_coordinates_sha256: str

    def __post_init__(self) -> None:
        if self.batch_size not in FROZEN_CONFIG.candidate_batch_sizes:
            raise ValueError("measurement batch_size is not preregistered")
        if self.backend not in {"cpu", "mlx"}:
            raise ValueError("measurement backend must be cpu or mlx")
        for name in ("transfer_seconds", "compute_seconds", "synchronize_seconds", "total_seconds"):
            _finite(getattr(self, name), name, minimum=0.0)
        if type(self.finite) is not bool or not self.finite:
            raise ValueError("measurement finite must be true")
        _sha256(self.candidate_coordinates_sha256, "candidate_coordinates_sha256")
        stage_total = self.transfer_seconds + self.compute_seconds + self.synchronize_seconds
        if not isclose(self.total_seconds, stage_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("total_seconds must include transfer, compute, and synchronization")


@dataclass(frozen=True)
class ParityReport:
    batch_key: BatchKey
    candidate_indices_by_alpha: Mapping[float, np.ndarray]
    max_normalized_error: float
    spearman: float

    def __post_init__(self) -> None:
        if not isinstance(self.batch_key, BatchKey):
            raise ValueError("batch_key must be a BatchKey")
        if not isinstance(self.candidate_indices_by_alpha, Mapping):
            raise ValueError("candidate_indices_by_alpha must be a mapping")
        expected_alphas = (FROZEN_CONFIG.exp417_config.candidate_alpha,)
        if tuple(self.candidate_indices_by_alpha) != expected_alphas:
            raise ValueError("candidate indices must cover exact preregistered alphas")
        frozen: dict[float, np.ndarray] = {}
        for alpha in expected_alphas:
            _finite(alpha, "candidate alpha", minimum=0.0)
            frozen[alpha] = _readonly_indices(self.candidate_indices_by_alpha[alpha], f"candidate_indices_by_alpha[{alpha}]")
        _finite(self.max_normalized_error, "max_normalized_error", minimum=0.0)
        _finite(self.spearman, "spearman")
        object.__setattr__(self, "candidate_indices_by_alpha", MappingProxyType(frozen))

    @property
    def passes(self) -> bool:
        return self.max_normalized_error <= FROZEN_CONFIG.max_normalized_error and self.spearman >= FROZEN_CONFIG.min_spearman


@dataclass(frozen=True)
class MlxAssessment:
    batch_key: BatchKey
    cpu_measurement: BackendMeasurement
    mlx_measurement: BackendMeasurement
    parity_report: ParityReport
    speedup: float
    uses_mlx: bool
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.batch_key, BatchKey) or not isinstance(self.parity_report, ParityReport):
            raise ValueError("assessment must contain frozen batch and parity records")
        if self.parity_report.batch_key != self.batch_key:
            raise ValueError("assessment parity batch key mismatch")
        if not isinstance(self.cpu_measurement, BackendMeasurement) or self.cpu_measurement.backend != "cpu":
            raise ValueError("cpu_measurement must be a CPU measurement")
        if not isinstance(self.mlx_measurement, BackendMeasurement) or self.mlx_measurement.backend != "mlx":
            raise ValueError("mlx_measurement must be an MLX measurement")
        if self.cpu_measurement.batch_size != self.batch_key.batch_size or self.mlx_measurement.batch_size != self.batch_key.batch_size:
            raise ValueError("measurement batch size mismatch")
        if self.cpu_measurement.candidate_coordinates_sha256 != self.mlx_measurement.candidate_coordinates_sha256:
            raise ValueError("CPU and MLX candidate coordinates differ")
        speedup = _finite(self.speedup, "speedup", minimum=0.0)
        expected_speedup = self.cpu_measurement.total_seconds / self.mlx_measurement.total_seconds
        if not isclose(speedup, expected_speedup, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("speedup must match complete measured timings")
        if type(self.uses_mlx) is not bool:
            raise ValueError("uses_mlx must be a bool")
        passed = self.parity_report.passes and speedup >= FROZEN_CONFIG.min_speedup
        if self.uses_mlx != passed:
            raise ValueError("MLX selection must match all preregistered gates")
        if passed and self.fallback_reason is not None:
            raise ValueError("passing MLX assessment cannot have a fallback reason")
        if not passed and (not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()):
            raise ValueError("failed MLX assessment requires a fallback reason")


@dataclass(frozen=True)
class BackendPolicy:
    backend: str
    assessment: MlxAssessment

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "mlx"} or not isinstance(self.assessment, MlxAssessment):
            raise ValueError("backend policy is invalid")
        if (self.backend == "mlx") != self.assessment.uses_mlx:
            raise ValueError("backend policy must fail closed to the assessment")


@dataclass(frozen=True)
class Exp421Manifest:
    state: str
    policy: BackendPolicy
    cached_candidate_indices: np.ndarray
    provenance: Mapping[str, str]
    config: Exp421Config = FROZEN_CONFIG
    experiment_id: str = "Exp421"

    def __post_init__(self) -> None:
        if self.state not in {"draft", "passed", "failed"}:
            raise ValueError("manifest state is invalid")
        if not isinstance(self.policy, BackendPolicy):
            raise ValueError("manifest policy is invalid")
        if type(self.config) is not Exp421Config:
            raise ValueError("manifest config is invalid")
        self.config.validate()
        if self.experiment_id != self.config.experiment_id:
            raise ValueError("manifest experiment_id is invalid")
        cached = _readonly_indices(self.cached_candidate_indices, "cached_candidate_indices", require_readonly=True)
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("manifest provenance must be a non-empty mapping")
        provenance = {_no_test_token(key, "provenance key"): _no_test_token(value, "provenance value") for key, value in self.provenance.items()}
        object.__setattr__(self, "cached_candidate_indices", cached)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))

