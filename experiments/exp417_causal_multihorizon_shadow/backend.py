"""Synthetic-only parity evidence and fail-closed accelerated-backend gates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
import copy

import numpy as np
import torch
from torch import Tensor, nn

from .config import FROZEN_CONFIG
from .contracts import BackendAudit
from .scoring import longest_true_run_fraction

try:  # Numba is optional outside the controlled experiment environment.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only on environments without numba
    njit = None


@dataclass(frozen=True)
class TorchBenchmarkMeasurement:
    """A complete, injectable device measurement used by the adoption gate."""

    candidate_identity: bool
    max_normalized_error: float
    spearman: float
    speedup: float
    finite_outputs: bool
    fallback: bool
    environment: str
    available: bool = True

    def __post_init__(self) -> None:
        if self.environment not in {"host", "isolated"}:
            raise ValueError("benchmark environment must be host or isolated")
        if any(type(value) is not bool for value in (
            self.candidate_identity, self.finite_outputs, self.fallback, self.available
        )):
            raise ValueError("benchmark flags must be bool")
        if any(not isinstance(value, (int, float)) or not isfinite(float(value)) for value in (
            self.max_normalized_error, self.spearman, self.speedup
        )):
            raise ValueError("benchmark metrics must be finite")


@dataclass(frozen=True)
class MPSProbe:
    """Host capability evidence, intentionally separate from adoption evidence."""

    environment: str
    built: bool
    available: bool
    forward_backward_synchronized: bool
    finite: bool
    fallback: bool
    reason: str | None = None


def persistence_numpy(mask: np.ndarray) -> float:
    """Return the frozen persistence statistic using its canonical NumPy reference."""
    return longest_true_run_fraction(mask)


if njit is not None:

    @njit(cache=True)
    def _numba_persistence(mask: np.ndarray) -> float:
        if mask.size == 0:
            return 0.0
        longest = 0
        current = 0
        for value in mask:
            if value:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 0
        return longest / mask.size


def persistence_numba(mask: np.ndarray) -> float:
    """Return the same frozen statistic from Numba; never change the definition."""
    _validate_mask(mask)
    if njit is None:
        raise RuntimeError("numba is unavailable; CPU fallback is required")
    return float(_numba_persistence(mask))


def benchmark_scoring_backend(
    masks: Sequence[np.ndarray], *, candidate_masks: Sequence[np.ndarray] | None = None, repeats: int = 3
) -> BackendAudit:
    """Measure NumPy/Numba persistence parity on caller-provided synthetic masks only."""
    frozen_masks = _masks(masks, "masks")
    frozen_candidates = frozen_masks if candidate_masks is None else _masks(candidate_masks, "candidate_masks")
    if len(frozen_masks) != len(frozen_candidates):
        raise ValueError("candidate_masks must align exactly with masks")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if njit is None:
        return BackendAudit("cpu", True, 0.0, 1.0, 1.0, True, "numba_unavailable")

    # Compile before timing: compilation is not scoring work and would otherwise
    # make a short synthetic smoke look unlike the exact repeated score stage.
    persistence_numba(frozen_masks[0])
    numpy_values, numpy_seconds = _timed_persistence(persistence_numpy, frozen_masks, repeats)
    numba_values, numba_seconds = _timed_persistence(persistence_numba, frozen_masks, repeats)
    candidate_identity = all(np.array_equal(mask, candidate) for mask, candidate in zip(frozen_masks, frozen_candidates))
    error = _max_normalized_error(numpy_values, numba_values)
    return BackendAudit(
        backend="numba",
        candidate_identity=candidate_identity,
        max_normalized_error=error,
        spearman=_spearman(numpy_values, numba_values),
        speedup=numpy_seconds / numba_seconds if numba_seconds > 0.0 else 0.0,
        fallback=False,
    )


def probe_mps_host() -> MPSProbe:
    """Run one synchronized host-only Conv1d forward/backward capability smoke."""
    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    if not (built and available):
        return MPSProbe("host", built, available, False, False, True, "mps_unavailable")
    try:
        device = torch.device("mps")
        network = nn.Conv1d(2, 2, kernel_size=3).to(device)
        values = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 2, 32).to(device)
        if not _device_matches(values.device, device) or any(
            not _device_matches(parameter.device, device) for parameter in network.parameters()
        ):
            return MPSProbe("host", built, available, False, False, True, "device_fallback")
        output = network(values)
        output.square().mean().backward()
        torch.mps.synchronize()
        finite = bool(torch.isfinite(output).all() and all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in network.parameters()
        ))
        return MPSProbe("host", built, available, finite, finite, not finite, None if finite else "nonfinite_outputs")
    except RuntimeError:
        return MPSProbe("host", built, available, False, False, True, "mps_runtime_error")


def benchmark_torch_devices(
    model: nn.Module | None = None,
    inputs: Tensor | None = None,
    *,
    candidate_selector: Callable[[Mapping[int, np.ndarray]], Sequence[int]] | None = None,
    repeats: int = 3,
    measurement: TorchBenchmarkMeasurement | None = None,
) -> BackendAudit:
    """Benchmark synchronized CPU/MPS forward-backward work and apply all frozen gates."""
    if measurement is None:
        if not isinstance(model, nn.Module) or not isinstance(inputs, Tensor):
            raise ValueError("model and inputs are required when no measurement is injected")
        if candidate_selector is None:
            raise ValueError("candidate_selector must use the frozen reference scoring path")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        probe = probe_mps_host()
        if not probe.available:
            measurement = TorchBenchmarkMeasurement(True, 0.0, 1.0, 1.0, False, True, "host", False)
        elif not probe.forward_backward_synchronized or probe.fallback:
            measurement = TorchBenchmarkMeasurement(True, 0.0, 1.0, 1.0, False, True, "host")
        else:
            measurement = _measure_torch_devices(model, inputs, candidate_selector, repeats)
    return _audit_torch_measurement(measurement)


def select_canonical_backend(audits: Sequence[BackendAudit]) -> str:
    """Adopt MPS only from explicit passing evidence; all other outcomes stay CPU."""
    frozen = tuple(audits)
    if not frozen or any(not isinstance(audit, BackendAudit) for audit in frozen):
        raise ValueError("MPS BackendAudit evidence is required")
    return "mps" if any(audit.backend == "mps" for audit in frozen) else "cpu"


def _audit_torch_measurement(measurement: TorchBenchmarkMeasurement) -> BackendAudit:
    if not measurement.available:
        reason = "mps_unavailable" if measurement.environment == "host" else "isolated_mps_unavailable"
    elif measurement.fallback:
        reason = "device_fallback"
    elif not measurement.finite_outputs:
        reason = "nonfinite_outputs"
    elif not measurement.candidate_identity:
        reason = "candidate_identity"
    elif measurement.max_normalized_error > FROZEN_CONFIG.mps_max_error:
        reason = "max_normalized_error"
    elif measurement.spearman < FROZEN_CONFIG.mps_min_spearman:
        reason = "spearman"
    elif measurement.speedup < FROZEN_CONFIG.mps_min_speedup:
        reason = "speedup"
    else:
        return BackendAudit("mps", True, measurement.max_normalized_error, measurement.spearman, measurement.speedup, False)
    return BackendAudit("cpu", measurement.candidate_identity, measurement.max_normalized_error, measurement.spearman, measurement.speedup, True, reason)


def _measure_torch_devices(
    model: nn.Module,
    inputs: Tensor,
    candidate_selector: Callable[[Mapping[int, np.ndarray]], Sequence[int]],
    repeats: int,
) -> TorchBenchmarkMeasurement:
    if inputs.device.type != "cpu" or not torch.isfinite(inputs).all():
        raise ValueError("synthetic benchmark inputs must be finite CPU tensors")
    cpu_values, cpu_candidates, cpu_seconds, cpu_fallback = _run_torch_stage(model, inputs, torch.device("cpu"), candidate_selector, repeats)
    mps_values, mps_candidates, mps_seconds, mps_fallback = _run_torch_stage(model, inputs, torch.device("mps"), candidate_selector, repeats)
    finite = bool(np.isfinite(cpu_values).all() and np.isfinite(mps_values).all())
    return TorchBenchmarkMeasurement(
        candidate_identity=cpu_candidates == mps_candidates,
        max_normalized_error=_max_normalized_error(cpu_values, mps_values),
        spearman=_spearman(cpu_values, mps_values),
        speedup=cpu_seconds / mps_seconds if mps_seconds > 0.0 else 0.0,
        finite_outputs=finite,
        fallback=cpu_fallback or mps_fallback,
        environment="host",
    )


def _run_torch_stage(
    model: nn.Module,
    inputs: Tensor,
    device: torch.device,
    candidate_selector: Callable[[Mapping[int, np.ndarray]], Sequence[int]],
    repeats: int,
) -> tuple[np.ndarray, tuple[int, ...], float, bool]:
    network = copy.deepcopy(model).to(device)
    fallback = any(not _device_matches(parameter.device, device) for parameter in network.parameters())
    final_outputs: Mapping[int, np.ndarray] | None = None
    elapsed = 0.0
    for _ in range(repeats):
        network.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.synchronize()
        started = perf_counter()
        values = inputs.detach().clone().to(device).requires_grad_(True)
        fallback = fallback or not _device_matches(values.device, device)
        output = network(values)
        fallback = fallback or any(not _device_matches(value.device, device) for value in output.values())
        sum(value.square().mean() for value in output.values()).backward()
        materialized = {
            int(horizon): value.detach().to("cpu").numpy().copy()
            for horizon, value in sorted(output.items())
        }
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed += perf_counter() - started
        final_outputs = materialized
    if final_outputs is None:  # pragma: no cover - repeats validation makes this unreachable
        raise AssertionError("torch benchmark did not run")
    selected = tuple(candidate_selector(final_outputs))
    if any(type(index) is not int for index in selected):
        raise ValueError("candidate_selector must return exact integer identities")
    flattened = np.concatenate([final_outputs[horizon].reshape(-1) for horizon in sorted(final_outputs)])
    return flattened, selected, elapsed, fallback


def _validate_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.dtype(bool):
        raise ValueError("mask must be a one-dimensional boolean ndarray")


def _masks(values: Sequence[np.ndarray], name: str) -> tuple[np.ndarray, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence")
    frozen = tuple(values)
    if not frozen:
        raise ValueError(f"{name} must be non-empty")
    for mask in frozen:
        _validate_mask(mask)
    return frozen


def _timed_persistence(
    function: Callable[[np.ndarray], float], masks: Sequence[np.ndarray], repeats: int
) -> tuple[np.ndarray, float]:
    started = perf_counter()
    values = np.asarray([function(mask) for _ in range(repeats) for mask in masks], dtype=np.float64)
    return values[: len(masks)], perf_counter() - started


def _max_normalized_error(reference: np.ndarray, observed: np.ndarray) -> float:
    if reference.shape != observed.shape or not np.isfinite(reference).all() or not np.isfinite(observed).all():
        return 1.0
    denominator = max(1.0, float(np.max(np.abs(reference), initial=0.0)))
    return float(np.max(np.abs(reference - observed), initial=0.0) / denominator)


def _spearman(reference: np.ndarray, observed: np.ndarray) -> float:
    if reference.shape != observed.shape or not np.isfinite(reference).all() or not np.isfinite(observed).all():
        return -1.0
    if np.array_equal(reference, observed):
        return 1.0
    left = _average_ranks(reference)
    right = _average_ranks(observed)
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _device_matches(actual: torch.device, requested: torch.device) -> bool:
    """Treat ``mps`` and its allocated ``mps:0`` representation as one device."""
    if actual.type != requested.type:
        return False
    if requested.type == "mps":
        return actual.index in {None, 0} and requested.index in {None, 0}
    return actual == requested


__all__ = [
    "MPSProbe",
    "TorchBenchmarkMeasurement",
    "benchmark_scoring_backend",
    "benchmark_torch_devices",
    "persistence_numba",
    "persistence_numpy",
    "probe_mps_host",
    "select_canonical_backend",
]
