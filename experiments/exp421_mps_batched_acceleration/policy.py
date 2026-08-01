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
import platform
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
_MISSING = object()
_MLX_REASONS = frozenset({"package_absent", "algorithm_rewrite_required"})
_POLICY_REASONS = frozenset(
    {
        "batch_size_one_cpu_mandatory",
        "mps_unavailable",
        "nonfinite_output",
        "candidate_coordinates_mismatch",
        "qualification_mismatch",
        "normalized_error_above_1e-4",
        "spearman_below_0.999",
        "speedup_below_1.1",
        "eligible",
    }
)


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
        policy_keys = tuple(self.policies) if isinstance(self.policies, Mapping) else ()
        reason_keys = tuple(self.reasons) if isinstance(self.reasons, Mapping) else ()
        if (
            not isinstance(self.policies, Mapping)
            or policy_keys != expected
            or any(type(key) is not int for key in policy_keys)
        ):
            raise ValueError("policy must cover the exact frozen batch sizes")
        if (
            not isinstance(self.reasons, Mapping)
            or reason_keys != expected
            or any(type(key) is not int for key in reason_keys)
        ):
            raise ValueError("policy reasons must cover the exact frozen batch sizes")
        any_mps = False
        any_fallback = False
        for batch_size in expected:
            if type(batch_size) is not int:
                raise ValueError("policy batch sizes must be exact integers")
            policy = self.policies[batch_size]
            if not isinstance(policy, BackendPolicy):
                raise ValueError("policy rows must be BackendPolicy records")
            backend = getattr(policy, "backend", _MISSING)
            assessment = getattr(policy, "assessment", _MISSING)
            if backend not in {"cpu", "mps"} or not isinstance(assessment, MpsAssessment):
                raise ValueError("policy rows must contain valid CPU/MPS assessments")
            assessment_key = _validate_batch_key(
                getattr(assessment, "batch_key", None),
                "policy assessment.batch_key",
            )
            if assessment_key.batch_size != batch_size:
                raise ValueError("policy mapping key must match assessment batch size")
            uses_mps = getattr(assessment, "uses_mps", _MISSING)
            if type(uses_mps) is not bool or (backend == "mps") != uses_mps:
                raise ValueError("policy backend must agree with its MPS assessment")
            derived_reason = _assessment_reason(assessment, FROZEN_CONFIG)
            reason = self.reasons[batch_size]
            if reason != derived_reason or reason not in _POLICY_REASONS:
                raise ValueError("policy reason must exactly match the frozen assessment gates")
            any_mps = any_mps or backend == "mps"
            any_fallback = any_fallback or reason != "eligible"
        expected_fail_closed = (not any_mps) or any_fallback
        if type(self.fail_closed) is not bool or self.fail_closed != expected_fail_closed:
            raise ValueError("policy fail_closed must match MPS selection and fallback rows")
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
    mlx_version: str | None
    mlx_metal_version: str | None
    mlx_availability: str
    mlx_eligibility_reason: str

    def __post_init__(self) -> None:
        for name in ("torch_version", "device_name", "mlx_eligibility_reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "test" in value.lower():
                raise ValueError(f"{name} must be a non-test boundary string")
        for name in ("mlx_version", "mlx_metal_version"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip() or "test" in value.lower()
            ):
                raise ValueError(f"{name} must be absent or a non-test version string")
        if type(self.mps_built) is not bool or type(self.mps_available) is not bool:
            raise ValueError("MPS capability flags must be bools")
        if self.mps_available and not self.mps_built:
            raise ValueError("available MPS requires a built backend")
        if self.mlx_availability not in {"available", "unavailable"}:
            raise ValueError("MLX availability must be available or unavailable")
        if self.mlx_eligibility_reason not in _MLX_REASONS:
            raise ValueError("MLX must have an exact ineligibility reason")
        versions_absent = all(
            value is None or (isinstance(value, str) and value.strip().lower() == _MLX_MISSING)
            for value in (self.mlx_version, self.mlx_metal_version)
        )
        versions_concrete = all(
            isinstance(value, str) and value.strip() and value.strip().lower() != _MLX_MISSING
            for value in (self.mlx_version, self.mlx_metal_version)
        )
        if self.mlx_eligibility_reason == "package_absent":
            if self.mlx_availability != "unavailable" or not versions_absent:
                raise ValueError("package_absent requires unavailable MLX with absent versions")
        elif self.mlx_availability != "available" or not versions_concrete:
            raise ValueError("algorithm_rewrite_required requires available MLX versions")


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
    # PyTorch does not expose a stable cross-version Apple hardware-name API.
    # Keep only public platform and architecture labels; do not embed
    # hostnames or other private machine identifiers in the manifest.
    system = platform.system() or "unknown"
    machine = platform.machine() or "unknown"
    device_name = f"{system} {machine} {'MPS' if mps_available else 'CPU'}"

    raw_mlx_version = _distribution_version("mlx")
    raw_mlx_metal_version = _distribution_version("mlx-metal")
    mlx_available = raw_mlx_version != _MLX_MISSING and raw_mlx_metal_version != _MLX_MISSING
    # A partial optional installation is still unavailable to this prototype;
    # normalize both provenance slots so package_absent remains coherent.
    mlx_version = raw_mlx_version if mlx_available else _MLX_MISSING
    mlx_metal_version = raw_mlx_metal_version if mlx_available else _MLX_MISSING
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
    try:
        values = dict(capability.__dict__)
        return MpsCapability(**values)
    except Exception as exc:
        raise ValueError("capability record is malformed") from exc


def capability_provenance(
    capability: MpsCapability,
    base: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Return immutable manifest-ready capability and MLX fields.

    When ``base`` is supplied, the required Git/DB/config/runtime fields are
    preserved and capability fields are overlaid, yielding a complete mapping
    accepted by :class:`Exp421Manifest`.
    """
    checked = validate_mps_capability(capability)
    if base is not None and not isinstance(base, Mapping):
        raise ValueError("base provenance must be a mapping")
    values = dict(base or {})
    values.update({
        "torch_version": checked.torch_version,
        "mps_built": checked.mps_built,
        "mps_available": checked.mps_available,
        "device_name": checked.device_name,
        "mlx_availability": checked.mlx_availability,
        "mlx_version": checked.mlx_version,
        "mlx_metal_version": checked.mlx_metal_version,
        "mlx_eligibility_reason": checked.mlx_eligibility_reason,
    })
    return MappingProxyType(values)


def _validate_config(config: Exp421Config) -> None:
    if type(config) is not Exp421Config:
        raise ValueError("config must be an Exp421Config")
    config.validate()


def _validate_batch_key(key: Any, name: str = "batch_key") -> BatchKey:
    if not isinstance(key, BatchKey):
        raise ValueError(f"{name} must be a BatchKey")
    try:
        batch_size = key.batch_size
        model_seed = key.model_seed
        perturbation_seed = key.perturbation_seed
    except Exception as exc:
        raise ValueError(f"{name} is malformed") from exc
    # Reconstruct through the immutable contract so object.__new__ records
    # cannot smuggle an unregistered batch or seed into an MPS decision.
    try:
        return BatchKey(batch_size, model_seed, perturbation_seed)
    except Exception as exc:
        raise ValueError(f"{name} contains invalid frozen batch or seed fields") from exc


def _validate_timing(
    measurement: Any,
    name: str,
    *,
    expected_batch_size: int | None = None,
    expected_backend: str | None = None,
) -> BackendMeasurement:
    if not isinstance(measurement, BackendMeasurement):
        raise ValueError(f"{name} must be a BackendMeasurement")
    try:
        batch_size = measurement.batch_size
        backend = measurement.backend
        finite = measurement.finite
        digest = measurement.candidate_coordinates_sha256
    except Exception as exc:
        raise ValueError(f"{name} is malformed") from exc
    if type(batch_size) is not int or batch_size not in FROZEN_CONFIG.candidate_batch_sizes:
        raise ValueError(f"{name}.batch_size is not preregistered")
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValueError(f"{name}.batch_size does not match the batch key")
    if backend not in {"cpu", "mps"} or (expected_backend is not None and backend != expected_backend):
        raise ValueError(f"{name}.backend is invalid")
    if type(finite) is not bool:
        raise ValueError(f"{name}.finite must be a bool")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name}.candidate_coordinates_sha256 is invalid")
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
    return measurement


def _validate_parity_shape(report: Any, config: Exp421Config, batch_key: BatchKey) -> ParityReport:
    if not isinstance(report, ParityReport):
        raise ValueError("available MPS requires a ParityReport")
    report_key = _validate_batch_key(getattr(report, "batch_key", None), "parity_report.batch_key")
    if report_key != batch_key:
        raise ValueError("parity report must bind to the exact assessment batch key")
    expected_alpha = config.exp417_config.candidate_alpha
    for name in ("cpu_candidate_indices_by_alpha", "mps_candidate_indices_by_alpha"):
        values = getattr(report, name, None)
        if not isinstance(values, MappingProxyType) or tuple(values) != (expected_alpha,):
            raise ValueError("parity must cover exactly the preregistered alpha")
        indices = values[expected_alpha]
        if not isinstance(indices, np.ndarray) or indices.dtype != np.dtype("int64") or indices.ndim != 1:
            raise ValueError("parity candidate indices must be one-dimensional int64 arrays")
        if indices.flags.writeable:
            raise ValueError("parity candidate indices must be read-only")
        if indices.size and (np.any(indices < 0) or np.any(indices[1:] <= indices[:-1])):
            raise ValueError("parity candidate indices must be sorted and unique")
    for name in ("max_normalized_error", "spearman"):
        value = getattr(report, name, None)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
            raise ValueError(f"parity {name} must be finite")
        if name == "max_normalized_error" and float(value) < 0.0:
            raise ValueError("parity max_normalized_error must be nonnegative")
    if type(getattr(report, "qualification_match", None)) is not bool:
        raise ValueError("parity qualification_match must be a bool")
    return report


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
    batch_key = _validate_batch_key(getattr(assessment, "batch_key", None), "assessment.batch_key")
    uses_mps = getattr(assessment, "uses_mps", _MISSING)
    mps_available = getattr(assessment, "mps_available", _MISSING)
    fallback_reason = getattr(assessment, "fallback_reason", _MISSING)
    cpu = getattr(assessment, "cpu_measurement", _MISSING)
    mps = getattr(assessment, "mps_measurement", _MISSING)
    parity_report = getattr(assessment, "parity_report", _MISSING)
    speedup = getattr(assessment, "speedup", _MISSING)
    device = getattr(assessment, "device", _MISSING)
    if type(uses_mps) is not bool:
        raise ValueError("assessment.uses_mps must be a bool")
    if type(mps_available) is not bool:
        raise ValueError("assessment.mps_available must be a bool")
    if (
        not isinstance(device, str)
        or not device.strip()
        or "test" in device.lower()
    ):
        raise ValueError("MPS assessment requires a device label")
    if fallback_reason is _MISSING:
        raise ValueError("assessment.fallback_reason is missing")
    if fallback_reason is not None and (
        not isinstance(fallback_reason, str) or not fallback_reason.strip()
    ):
        raise ValueError("assessment fallback_reason must be a nonempty string or None")
    batch_size = batch_key.batch_size
    if cpu is _MISSING:
        raise ValueError("assessment.cpu_measurement is missing")
    _validate_timing(
        cpu,
        "cpu_measurement",
        expected_batch_size=batch_size,
        expected_backend="cpu",
    )

    if not cpu.finite:
        if uses_mps or not isinstance(fallback_reason, str) or not fallback_reason.strip():
            raise ValueError("nonfinite CPU output requires an explicit CPU fallback")
        return "nonfinite_output"

    if not mps_available:
        if mps is not None or parity_report is not None or speedup is not None:
            raise ValueError("unavailable MPS cannot include MPS evidence")
        if uses_mps or not isinstance(fallback_reason, str) or not fallback_reason.strip():
            raise ValueError("unavailable MPS requires a CPU fallback reason")
        return "mps_unavailable"

    if device.strip().lower() in {"unavailable", "cpu"}:
        raise ValueError("available MPS requires a non-CPU device label")

    if mps is _MISSING:
        raise ValueError("assessment.mps_measurement is missing")
    _validate_timing(
        mps,
        "mps_measurement",
        expected_batch_size=batch_size,
        expected_backend="mps",
    )
    if not mps.finite:
        reason = "nonfinite_output"
        expected_uses_mps = False
    else:
        if parity_report is _MISSING:
            raise ValueError("assessment.parity_report is missing")
        report = _validate_parity_shape(parity_report, config, batch_key)
        if cpu.candidate_coordinates_sha256 != mps.candidate_coordinates_sha256:
            reason = "candidate_coordinates_mismatch"
            expected_uses_mps = False
        elif not _candidate_indices_match(report, config):
            reason = "candidate_coordinates_mismatch"
            expected_uses_mps = False
        elif not report.qualification_match:
            reason = "qualification_mismatch"
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
            if speedup is _MISSING:
                raise ValueError("assessment.speedup is missing")
            supplied_speedup = speedup
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

    if uses_mps != expected_uses_mps:
        raise ValueError("MPS assessment selection disagrees with all frozen gates")
    if expected_uses_mps:
        if fallback_reason is not None:
            raise ValueError("eligible MPS assessment cannot have a fallback reason")
    elif not isinstance(fallback_reason, str) or not fallback_reason.strip():
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
                key = _validate_batch_key(getattr(item, "batch_key", None), "assessment.batch_key")
                items.append((key.batch_size, item))
            elif isinstance(item, BackendPolicy):
                item_backend = getattr(item, "backend", _MISSING)
                item_assessment = getattr(item, "assessment", _MISSING)
                if item_backend not in {"cpu", "mps"} or not isinstance(item_assessment, MpsAssessment):
                    raise ValueError("measurement policy is malformed")
                key = _validate_batch_key(getattr(item_assessment, "batch_key", None), "assessment.batch_key")
                items.append((key.batch_size, item))
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
            policy_backend = getattr(value, "backend", _MISSING)
            assessment = getattr(value, "assessment", _MISSING)
            if policy_backend not in {"cpu", "mps"} or not isinstance(assessment, MpsAssessment):
                raise ValueError("measurement policy contains an invalid backend")
        else:
            assessment = value
        if not isinstance(assessment, MpsAssessment):
            raise ValueError("measurements must contain MpsAssessment records")
        key = _validate_batch_key(getattr(assessment, "batch_key", None), "assessment.batch_key")
        if key.batch_size != batch_size:
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
    fail_closed = (not any(policy.backend == "mps" for policy in policies.values())) or any(
        reason != "eligible" for reason in reasons.values()
    )
    return BackendPolicyBundle(policies=policies, reasons=reasons, fail_closed=fail_closed)


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
