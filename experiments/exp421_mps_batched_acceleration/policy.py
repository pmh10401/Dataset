"""Fail-closed CPU/MPS backend selection for the Exp421 diagnostic run.

The frozen contracts keep one :class:`BackendPolicy` per batch key.  This
module adds the small aggregate used by the hybrid runner: it validates exact
coverage of the preregistered batch schedule and exposes the selected backend
and reason for every row.  Runtime measurements are CPU or MPS only.  MLX is
looked up through package metadata for provenance and is never imported or
used to execute a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from math import isclose, isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from .config import Exp421Config, FROZEN_CONFIG
from .contracts import (
    BackendMeasurement,
    BackendPolicy,
    BatchKey,
    MpsAssessment,
    ParityReport,
)


_TIMING_TOLERANCE = 1e-12
_MLX_MISSING = "unavailable"
_MLX_UNAVAILABLE_REASONS = frozenset({"package_absent", "algorithm_rewrite_required"})


@dataclass(frozen=True)
class BackendPolicyBundle:
    """Immutable hybrid policy covering every preregistered batch size.

    ``policies`` contains the original per-batch ``BackendPolicy`` records,
    while ``reasons`` is an explicit decision record even for the mandatory
    CPU batch.  ``fail_closed`` is intentionally required to be true: a
    policy result is only constructible after all rows have been validated and
    every failed gate has a CPU route.
    """

    policies: Mapping[int, BackendPolicy]
    reasons: Mapping[int, str]
    fail_closed: bool = True

    def __post_init__(self) -> None:
        expected = FROZEN_CONFIG.candidate_batch_sizes
        if not isinstance(self.policies, Mapping) or tuple(self.policies) != expected:
            raise ValueError("policy must cover the exact frozen batch sizes")
        if not isinstance(self.reasons, Mapping) or tuple(self.reasons) != expected:
            raise ValueError("policy reasons must cover the exact frozen batch sizes")
        for batch_size in expected:
            if type(batch_size) is not int:
                raise ValueError("policy batch sizes must be exact integers")
            policy = self.policies[batch_size]
            if not isinstance(policy, BackendPolicy):
                raise ValueError("policy rows must be BackendPolicy records")
            reason = self.reasons[batch_size]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("policy rows require an explicit reason")
        if type(self.fail_closed) is not bool or not self.fail_closed:
            raise ValueError("policy must be explicitly fail-closed")
        object.__setattr__(self, "policies", MappingProxyType(dict(self.policies)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))

    @property
    def selected_backend_by_batch_size(self) -> Mapping[int, str]:
        """Return the selected CPU/MPS backend keyed by exact batch size."""
        return MappingProxyType({size: policy.backend for size, policy in self.policies.items()})

    @property
    def backend_by_batch_size(self) -> Mapping[int, str]:
        """Compatibility alias for callers serializing the policy."""
        return self.selected_backend_by_batch_size

    @property
    def assessments(self) -> Mapping[int, MpsAssessment]:
        return MappingProxyType({size: policy.assessment for size, policy in self.policies.items()})

    def __getitem__(self, batch_size: int) -> BackendPolicy:
        return self.policies[batch_size]


@dataclass(frozen=True)
class MpsCapability:
    """Runtime and MLX provenance detected without importing optional MLX."""

    torch_version: str
    mps_built: bool
    mps_available: bool
    device_name: str
    mlx_version: str
    mlx_metal_version: str
    mlx_availability: str
    mlx_eligibility_reason: str

    def __post_init__(self) -> None:
        for name in (
            "torch_version",
            "device_name",
            "mlx_version",
            "mlx_metal_version",
            "mlx_eligibility_reason",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "test" in value.lower():
                raise ValueError(f"{name} must be a non-test boundary string")
        if type(self.mps_built) is not bool or type(self.mps_available) is not bool:
            raise ValueError("MPS capability flags must be bools")
        if self.mlx_availability not in {"available", "unavailable"}:
            raise ValueError("MLX availability must be available or unavailable")
        if self.mlx_eligibility_reason not in _MLX_UNAVAILABLE_REASONS:
            raise ValueError("MLX must have an exact ineligibility reason")
        versions_absent = self.mlx_version == _MLX_MISSING and self.mlx_metal_version == _MLX_MISSING
        if (self.mlx_availability == "unavailable") != versions_absent:
            raise ValueError("MLX availability must agree with recorded versions")


def _distribution_version(name: str) -> str:
    """Read an installed distribution version without importing its package."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return _MLX_MISSING
    except Exception:
        # A broken metadata provider is indistinguishable from absence for a
        # fail-closed capability record; importantly, never install or import.
        return _MLX_MISSING


def assess_mps_capability() -> MpsCapability:
    """Capture PyTorch MPS and optional MLX provenance for a manifest.

    The function does not instantiate an MPS tensor and does not import MLX.
    If both MLX distributions exist, the prototype still records
    ``algorithm_rewrite_required`` because the runtime path is the frozen
    PyTorch MPS adapter, not a rewritten MLX implementation.
    """
    try:
        import torch
    except Exception:
        torch_version = "unavailable"
        mps_built = False
        mps_available = False
        device_name = "cpu"
    else:
        torch_version = str(getattr(torch, "__version__", "unavailable"))
        try:
            mps_built = bool(torch.backends.mps.is_built())
        except Exception:
            mps_built = False
        try:
            mps_available = bool(torch.backends.mps.is_available())
        except Exception:
            mps_available = False
        # PyTorch does not expose a stable cross-version MPS hardware-name
        # API.  Keep an explicit device label and retain host details in the
        # surrounding manifest/runtime provenance.
        device_name = "mps" if mps_available else "cpu"

    mlx_version = _distribution_version("mlx")
    mlx_metal_version = _distribution_version("mlx-metal")
    mlx_available = mlx_version != _MLX_MISSING and mlx_metal_version != _MLX_MISSING
    return MpsCapability(
        torch_version=torch_version,
        mps_built=mps_built,
        mps_available=mps_available,
        device_name=device_name,
        mlx_version=mlx_version,
        mlx_metal_version=mlx_metal_version,
        mlx_availability="available" if mlx_available else "unavailable",
        mlx_eligibility_reason=("algorithm_rewrite_required" if mlx_available else "package_absent"),
    )


# Explicit aliases make the capability record discoverable to benchmark code
# without creating a second implementation.
MpsCapabilityAssessment = MpsCapability
assess_capability = assess_mps_capability


def validate_mps_capability(capability: MpsCapability) -> MpsCapability:
    """Revalidate a capability record before copying it into provenance."""
    if not isinstance(capability, MpsCapability):
        raise ValueError("capability must be an MpsCapability")
    # Reconstructing through the public frozen constructor catches records
    # produced by an unsafe deserializer or an object forged without running
    # ``__post_init__``.
    return MpsCapability(**capability.__dict__)


def capability_provenance(capability: MpsCapability) -> Mapping[str, object]:
    """Return immutable manifest-ready capability and MLX fields."""
    checked = validate_mps_capability(capability)
    return MappingProxyType({
        "torch_version": checked.torch_version,
        "mps_built": checked.mps_built,
        "mps_available": checked.mps_available,
        "device_name": checked.device_name,
        "mlx_availability": checked.mlx_availability,
        "mlx_version": checked.mlx_version,
        "mlx_metal_version": checked.mlx_metal_version,
        "mlx_eligibility_reason": checked.mlx_eligibility_reason,
    })


def _validate_config(config: Exp421Config) -> None:
    if type(config) is not Exp421Config:
        raise ValueError("config must be an Exp421Config")
    config.validate()


def _validate_timing(measurement: Any, name: str) -> None:
    if not isinstance(measurement, BackendMeasurement):
        raise ValueError(f"{name} must be a BackendMeasurement")
    for field in ("transfer_seconds", "compute_seconds", "synchronize_seconds", "total_seconds"):
        value = getattr(measurement, field, None)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name}.{field} must be finite and nonnegative")
    expected = (
        measurement.transfer_seconds
        + measurement.compute_seconds
        + measurement.synchronize_seconds
    )
    if not isclose(measurement.total_seconds, expected, rel_tol=0.0, abs_tol=_TIMING_TOLERANCE):
        raise ValueError(f"{name}.total_seconds must include transfer, compute, and synchronization")


def _validate_parity_shape(report: Any, config: Exp421Config) -> None:
    if not isinstance(report, ParityReport):
        raise ValueError("available MPS requires a ParityReport")
    expected_alpha = config.exp417_config.candidate_alpha
    for name in ("cpu_candidate_indices_by_alpha", "mps_candidate_indices_by_alpha"):
        values = getattr(report, name, None)
        if not isinstance(values, Mapping) or tuple(values) != (expected_alpha,):
            raise ValueError("parity must cover exactly the preregistered alpha")
        indices = values[expected_alpha]
        if not isinstance(indices, np.ndarray) or indices.dtype != np.dtype("int64") or indices.ndim != 1:
            raise ValueError("parity candidate indices must be one-dimensional int64 arrays")
        if indices.size and (np.any(indices < 0) or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("parity candidate indices must be sorted and unique")
    for name in ("max_normalized_error", "spearman"):
        value = getattr(report, name, None)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
            raise ValueError(f"parity {name} must be finite")


def _candidate_indices_match(report: ParityReport, config: Exp421Config) -> bool:
    alpha = config.exp417_config.candidate_alpha
    return bool(np.array_equal(
        report.cpu_candidate_indices_by_alpha[alpha],
        report.mps_candidate_indices_by_alpha[alpha],
    ))


def _assessment_reason(assessment: MpsAssessment, config: Exp421Config) -> str:
    """Validate an assessment and return its deterministic policy reason."""
    if not isinstance(assessment, MpsAssessment):
        raise ValueError("assessment must be an MpsAssessment")
    if not isinstance(assessment.batch_key, BatchKey):
        raise ValueError("assessment must contain a frozen BatchKey")
    if not isinstance(assessment.device, str) or not assessment.device.strip():
        raise ValueError("MPS assessment requires a device label")
    batch_size = assessment.batch_key.batch_size
    cpu = assessment.cpu_measurement
    _validate_timing(cpu, "cpu_measurement")
    if cpu.batch_size != batch_size or cpu.backend != "cpu":
        raise ValueError("CPU measurement does not match the batch key")

    if not cpu.finite:
        if assessment.uses_mps or not isinstance(assessment.fallback_reason, str) or not assessment.fallback_reason.strip():
            raise ValueError("nonfinite CPU output requires an explicit CPU fallback")
        return "nonfinite_output"

    if not assessment.mps_available:
        if assessment.mps_measurement is not None or assessment.parity_report is not None or assessment.speedup is not None:
            raise ValueError("unavailable MPS cannot include MPS evidence")
        if assessment.uses_mps or not isinstance(assessment.fallback_reason, str) or not assessment.fallback_reason.strip():
            raise ValueError("unavailable MPS requires a CPU fallback reason")
        return "mps_unavailable"

    if assessment.device.strip().lower() in {"unavailable", "cpu"}:
        raise ValueError("available MPS requires a non-CPU device label")

    mps = assessment.mps_measurement
    _validate_timing(mps, "mps_measurement")
    if mps.batch_size != batch_size or mps.backend != "mps":
        raise ValueError("MPS measurement does not match the batch key")
    if not mps.finite:
        reason = "nonfinite_output"
        expected_uses_mps = False
    else:
        report = assessment.parity_report
        _validate_parity_shape(report, config)
        if cpu.candidate_coordinates_sha256 != mps.candidate_coordinates_sha256:
            reason = "candidate_coordinates_mismatch"
            expected_uses_mps = False
        elif not _candidate_indices_match(report, config):
            reason = "candidate_coordinates_mismatch"
            expected_uses_mps = False
        elif report.max_normalized_error > config.max_normalized_error:
            reason = "normalized_error_above_1e-4"
            expected_uses_mps = False
        elif report.spearman < config.min_spearman:
            reason = "spearman_below_0.999"
            expected_uses_mps = False
        elif mps.total_seconds <= 0.0:
            raise ValueError("MPS total_seconds must be positive")
        else:
            expected_speedup = cpu.total_seconds / mps.total_seconds
            supplied_speedup = assessment.speedup
            if (
                isinstance(supplied_speedup, bool)
                or not isinstance(supplied_speedup, Real)
                or not isfinite(float(supplied_speedup))
                or not isclose(float(supplied_speedup), expected_speedup, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError("speedup must match complete transfer, compute, and synchronization timing")
            if expected_speedup < config.min_speedup:
                reason = "speedup_below_1.1"
                expected_uses_mps = False
            elif batch_size == 1:
                reason = "batch_size_one_cpu_mandatory"
                expected_uses_mps = False
            else:
                reason = "eligible"
                expected_uses_mps = True

    if assessment.uses_mps != expected_uses_mps:
        raise ValueError("MPS assessment selection disagrees with all frozen gates")
    if expected_uses_mps:
        if assessment.fallback_reason is not None:
            raise ValueError("eligible MPS assessment cannot have a fallback reason")
    elif not isinstance(assessment.fallback_reason, str) or not assessment.fallback_reason.strip():
        raise ValueError("failed MPS assessment requires an explicit fallback reason")
    return reason


def _normalize_measurements(measurements: Mapping[int, Any] | Sequence[Any], config: Exp421Config) -> dict[int, MpsAssessment]:
    expected = config.candidate_batch_sizes
    if isinstance(measurements, Mapping):
        items = list(measurements.items())
    elif isinstance(measurements, Sequence) and not isinstance(measurements, (str, bytes, bytearray)):
        items = []
        for item in measurements:
            if isinstance(item, tuple) and len(item) == 2 and type(item[0]) is int:
                items.append((item[0], item[1]))
            elif isinstance(item, MpsAssessment):
                items.append((item.batch_key.batch_size, item))
            elif isinstance(item, BackendPolicy):
                items.append((item.assessment.batch_key.batch_size, item))
            else:
                raise ValueError("measurements must contain MpsAssessment records")
    else:
        raise ValueError("measurements must be a mapping or sequence")

    normalized: dict[int, MpsAssessment] = {}
    for batch_size, value in items:
        if type(batch_size) is not int or batch_size not in expected:
            raise ValueError("measurements contain a non-preregistered batch size")
        if batch_size in normalized:
            raise ValueError("measurements contain duplicate batch sizes")
        if isinstance(value, BackendPolicy):
            assessment = value.assessment
            if value.backend not in {"cpu", "mps"}:
                raise ValueError("measurement policy contains an invalid backend")
        else:
            assessment = value
        if not isinstance(assessment, MpsAssessment):
            raise ValueError("measurements must contain MpsAssessment records")
        if assessment.batch_key.batch_size != batch_size:
            raise ValueError("measurement batch size does not match its key")
        normalized[batch_size] = assessment
    if set(normalized) != set(expected):
        raise ValueError("measurements must contain one assessment for every frozen batch size")
    return normalized


def choose_backend(
    measurements: Mapping[int, MpsAssessment] | Sequence[MpsAssessment],
    config: Exp421Config = FROZEN_CONFIG,
) -> BackendPolicyBundle:
    """Choose CPU/MPS for every preregistered batch size.

    The function is deliberately deterministic: configuration is validated as
    the exact frozen object, every assessment is checked independently, and a
    missing or malformed row raises instead of being interpreted as a pass.
    """
    _validate_config(config)
    assessments = _normalize_measurements(measurements, config)
    policies: dict[int, BackendPolicy] = {}
    reasons: dict[int, str] = {}
    for batch_size in config.candidate_batch_sizes:
        assessment = assessments[batch_size]
        reason = _assessment_reason(assessment, config)
        backend = "mps" if assessment.uses_mps else "cpu"
        policies[batch_size] = BackendPolicy(backend, assessment)
        reasons[batch_size] = reason
    return BackendPolicyBundle(policies=policies, reasons=reasons, fail_closed=True)


__all__ = [
    "BackendPolicyBundle",
    "MpsCapability",
    "MpsCapabilityAssessment",
    "assess_capability",
    "assess_mps_capability",
    "capability_provenance",
    "choose_backend",
    "validate_mps_capability",
]
