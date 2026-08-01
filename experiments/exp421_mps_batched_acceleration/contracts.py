"""Validated immutable provenance records for Exp421 acceleration gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
import re
from types import MappingProxyType

import numpy as np

from .config import Exp421Config, FROZEN_CONFIG, frozen_config_sha256


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
    if not isinstance(value, str) or not value.strip() or re.search(r"\btest\b", value, re.IGNORECASE):
        raise ValueError(f"{name} must be a non-TEST boundary string")
    return value


_MANIFEST_PROVENANCE_FIELDS = frozenset(
    {
        "git_commit",
        "database_sha256",
        "config_sha256",
        "device",
        "runtime",
        "source",
        "fresh_no_clobber_attempt",
        "train_only",
        "mlx_availability",
        "mlx_version",
        "mlx_metal_version",
        "mlx_eligibility_reason",
    }
)
_MLX_UNAVAILABLE_REASONS = frozenset({"package_absent", "algorithm_rewrite_required"})


@dataclass(frozen=True)
class BatchKey:
    batch_size: int
    model_seed: int
    perturbation_seed: int

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size not in FROZEN_CONFIG.candidate_batch_sizes:
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
        if type(self.batch_size) is not int or self.batch_size not in FROZEN_CONFIG.candidate_batch_sizes:
            raise ValueError("measurement batch_size is not preregistered")
        if self.backend not in {"cpu", "mps"}:
            raise ValueError("measurement backend must be cpu or mps")
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
    cpu_candidate_indices_by_alpha: Mapping[float, np.ndarray]
    mps_candidate_indices_by_alpha: Mapping[float, np.ndarray]
    max_normalized_error: float
    spearman: float

    def __post_init__(self) -> None:
        if not isinstance(self.batch_key, BatchKey):
            raise ValueError("batch_key must be a BatchKey")
        if not isinstance(self.cpu_candidate_indices_by_alpha, Mapping):
            raise ValueError("cpu_candidate_indices_by_alpha must be a mapping")
        if not isinstance(self.mps_candidate_indices_by_alpha, Mapping):
            raise ValueError("mps_candidate_indices_by_alpha must be a mapping")
        expected_alphas = (FROZEN_CONFIG.exp417_config.candidate_alpha,)
        if tuple(self.cpu_candidate_indices_by_alpha) != expected_alphas or tuple(self.mps_candidate_indices_by_alpha) != expected_alphas:
            raise ValueError("CPU and MPS candidate indices must cover exact preregistered alphas")
        frozen_cpu: dict[float, np.ndarray] = {}
        frozen_mps: dict[float, np.ndarray] = {}
        for alpha in expected_alphas:
            _finite(alpha, "candidate alpha", minimum=0.0)
            frozen_cpu[alpha] = _readonly_indices(self.cpu_candidate_indices_by_alpha[alpha], f"cpu_candidate_indices_by_alpha[{alpha}]")
            frozen_mps[alpha] = _readonly_indices(self.mps_candidate_indices_by_alpha[alpha], f"mps_candidate_indices_by_alpha[{alpha}]")
        _finite(self.max_normalized_error, "max_normalized_error", minimum=0.0)
        _finite(self.spearman, "spearman")
        object.__setattr__(self, "cpu_candidate_indices_by_alpha", MappingProxyType(frozen_cpu))
        object.__setattr__(self, "mps_candidate_indices_by_alpha", MappingProxyType(frozen_mps))

    @property
    def candidate_indices_match(self) -> bool:
        return all(
            np.array_equal(self.cpu_candidate_indices_by_alpha[alpha], self.mps_candidate_indices_by_alpha[alpha])
            for alpha in self.cpu_candidate_indices_by_alpha
        )

    @property
    def passes(self) -> bool:
        return self.candidate_indices_match and self.max_normalized_error <= FROZEN_CONFIG.max_normalized_error and self.spearman >= FROZEN_CONFIG.min_spearman


@dataclass(frozen=True)
class MpsAssessment:
    batch_key: BatchKey
    cpu_measurement: BackendMeasurement
    mps_measurement: BackendMeasurement | None
    parity_report: ParityReport | None
    speedup: float | None
    uses_mps: bool
    fallback_reason: str | None
    mps_available: bool
    device: str

    def __post_init__(self) -> None:
        if not isinstance(self.batch_key, BatchKey):
            raise ValueError("assessment must contain a frozen batch key")
        if not isinstance(self.cpu_measurement, BackendMeasurement) or self.cpu_measurement.backend != "cpu":
            raise ValueError("cpu_measurement must be a CPU measurement")
        if self.cpu_measurement.batch_size != self.batch_key.batch_size:
            raise ValueError("measurement batch size mismatch")
        if type(self.uses_mps) is not bool:
            raise ValueError("uses_mps must be a bool")
        if type(self.mps_available) is not bool:
            raise ValueError("mps_available must be a bool")
        _no_test_token(self.device, "device")
        if not self.mps_available:
            if any(value is not None for value in (self.mps_measurement, self.parity_report, self.speedup)):
                raise ValueError("unavailable MPS cannot have MPS measurement or parity evidence")
            if self.uses_mps or not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip():
                raise ValueError("unavailable MPS requires an explicit CPU fallback reason")
            return
        if not isinstance(self.mps_measurement, BackendMeasurement) or self.mps_measurement.backend != "mps":
            raise ValueError("available MPS requires an MPS measurement")
        if not isinstance(self.parity_report, ParityReport) or self.parity_report.batch_key != self.batch_key:
            raise ValueError("available MPS requires parity for the exact batch key")
        if self.mps_measurement.batch_size != self.batch_key.batch_size:
            raise ValueError("measurement batch size mismatch")
        if self.cpu_measurement.candidate_coordinates_sha256 != self.mps_measurement.candidate_coordinates_sha256:
            raise ValueError("CPU and MPS candidate coordinates differ")
        speedup = _finite(self.speedup, "speedup", minimum=0.0)
        if self.mps_measurement.total_seconds <= 0.0:
            raise ValueError("available MPS total_seconds must be positive")
        expected_speedup = self.cpu_measurement.total_seconds / self.mps_measurement.total_seconds
        if not isclose(speedup, expected_speedup, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("speedup must match complete measured timings")
        eligible = (
            self.batch_key.batch_size != 1
            and self.parity_report.candidate_indices_match
            and self.parity_report.passes
            and speedup >= FROZEN_CONFIG.min_speedup
        )
        if self.uses_mps != eligible:
            raise ValueError("MPS selection must match all preregistered gates")
        if eligible and self.fallback_reason is not None:
            raise ValueError("passing MPS assessment cannot have a fallback reason")
        if not eligible and (not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()):
            raise ValueError("failed MPS assessment requires a fallback reason")


@dataclass(frozen=True)
class BackendPolicy:
    backend: str
    assessment: MpsAssessment

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "mps"} or not isinstance(self.assessment, MpsAssessment):
            raise ValueError("backend policy is invalid")
        if (self.backend == "mps") != self.assessment.uses_mps:
            raise ValueError("backend policy must fail closed to the assessment")


@dataclass(frozen=True)
class Exp421Manifest:
    state: str
    policy: BackendPolicy
    cached_candidate_indices: np.ndarray
    provenance: Mapping[str, object]
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
        if not isinstance(self.provenance, Mapping) or set(self.provenance) != _MANIFEST_PROVENANCE_FIELDS:
            raise ValueError("manifest provenance must contain the exact required fields")
        provenance = dict(self.provenance)
        _sha256(provenance["config_sha256"], "config_sha256")
        if provenance["config_sha256"] != frozen_config_sha256(self.config):
            raise ValueError("config_sha256 must bind to the frozen Exp421 config")
        if not (isinstance(provenance["git_commit"], str) and len(provenance["git_commit"]) in {40, 64} and all(c in "0123456789abcdef" for c in provenance["git_commit"])):
            raise ValueError("git_commit must be a lowercase Git object digest")
        if provenance["database_sha256"] != self.config.exp417_config.database_sha256:
            raise ValueError("database_sha256 must bind to Exp417")
        for name in ("device", "runtime", "source"):
            _no_test_token(provenance[name], name)
        if type(provenance["fresh_no_clobber_attempt"]) is not bool or not provenance["fresh_no_clobber_attempt"]:
            raise ValueError("fresh_no_clobber_attempt must be true")
        if type(provenance["train_only"]) is not bool or not provenance["train_only"]:
            raise ValueError("train_only must be true")
        if provenance["device"] != self.policy.assessment.device:
            raise ValueError("manifest device must match the MPS assessment")
        availability = provenance["mlx_availability"]
        if availability not in {"available", "unavailable"}:
            raise ValueError("manifest MLX availability must be available or unavailable")
        reason = provenance["mlx_eligibility_reason"]
        if availability == "unavailable":
            if any(provenance[name] not in (None, "unavailable") for name in ("mlx_version", "mlx_metal_version")):
                raise ValueError("unavailable MLX must have absent or unavailable versions")
            _no_test_token(reason, "mlx_eligibility_reason")
            if reason.strip().lower() not in _MLX_UNAVAILABLE_REASONS:
                raise ValueError("unavailable MLX must have an explicit unavailable reason")
        else:
            for name in ("mlx_version", "mlx_metal_version"):
                _no_test_token(provenance[name], name)
                if provenance[name].strip().lower() == "unavailable":
                    raise ValueError("available MLX must have concrete versions")
            _no_test_token(reason, "mlx_eligibility_reason")
            if reason.strip().lower() in _MLX_UNAVAILABLE_REASONS:
                raise ValueError("available MLX cannot have an unavailable reason")
        object.__setattr__(self, "cached_candidate_indices", cached)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
