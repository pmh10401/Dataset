"""Synthetic-only backend parity evidence and fail-closed MPS adoption gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from time import perf_counter
import copy

import numpy as np
import torch
from torch import Tensor, nn

from .config import FROZEN_CONFIG
from .contracts import BackendAudit, ProcessedGroup, RawTrainDataset
from .model import CausalMultiHorizonCNN
from .preprocessing import build_horizons
from .scoring import longest_true_run_fraction, score_horizon

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


_MPS_FAILURE_PRECEDENCE = (
    "mps_unavailable", "mps_runtime_error", "device_fallback", "nonfinite_outputs",
    "candidate_identity", "max_normalized_error", "spearman", "speedup",
)


@dataclass(frozen=True)
class BenchmarkTrainGroup:
    """One benchmarkable exact-length group, bound only to normal TRAIN metadata."""

    train: RawTrainDataset
    processed: ProcessedGroup

    def __post_init__(self) -> None:
        if not isinstance(self.train, RawTrainDataset):
            raise ValueError("benchmark metadata must be a RawTrainDataset")
        if not isinstance(self.processed, ProcessedGroup):
            raise ValueError("benchmark group must be a ProcessedGroup")
        if self.processed.status not in {"candidate_eligible", "diagnostic_only"}:
            raise ValueError("benchmark group must retain usable TRAIN channels")
        raw = {item.instance_index: item for item in self.train.series}
        if any(raw.get(item.instance_index) != item for item in self.processed.group.series):
            raise ValueError("processed group must bind exactly to its RawTrainDataset")

    @property
    def dataset(self) -> str:
        return self.train.dataset

    @property
    def length(self) -> int:
        return self.processed.group.length

    @property
    def normal_count(self) -> int:
        return len(self.processed.group.series)


@dataclass(frozen=True)
class TorchBenchmarkMeasurement:
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
        if any(type(value) is not bool for value in (self.candidate_identity, self.finite_outputs, self.fallback, self.available)):
            raise ValueError("benchmark flags must be bool")
        if any(not isinstance(value, (int, float)) or not isfinite(float(value)) for value in (self.max_normalized_error, self.spearman, self.speedup)):
            raise ValueError("benchmark metrics must be finite")


@dataclass(frozen=True)
class MPSProbe:
    environment: str
    built: bool
    available: bool
    forward_backward_synchronized: bool
    finite: bool
    fallback: bool
    reason: str | None = None


def select_benchmark_strata(groups: Sequence[BenchmarkTrainGroup]) -> tuple[BenchmarkTrainGroup, ...]:
    """Choose at most one TRAIN-only representative per count×length tertile."""
    frozen = tuple(groups)
    if not frozen or any(not isinstance(group, BenchmarkTrainGroup) for group in frozen):
        raise ValueError("benchmark strata require non-empty BenchmarkTrainGroup records")
    identities = tuple((group.dataset, group.length) for group in frozen)
    if len(identities) != len(set(identities)):
        raise ValueError("benchmark groups must have unique dataset/length identities")
    count_bins = _tertiles(frozen, lambda group: group.normal_count)
    length_bins = _tertiles(frozen, lambda group: group.length)
    selected: dict[tuple[int, int], BenchmarkTrainGroup] = {}
    for group in frozen:
        cell = (count_bins[(group.dataset, group.length)], length_bins[(group.dataset, group.length)])
        winner = selected.get(cell)
        if winner is None or (group.dataset, group.length) < (winner.dataset, winner.length):
            selected[cell] = group
    return tuple(sorted(selected.values(), key=lambda group: (group.dataset, group.length)))


def persistence_numpy(mask: np.ndarray) -> float:
    return longest_true_run_fraction(mask)


if njit is not None:
    @njit(cache=True)
    def _numba_persistence(mask: np.ndarray) -> float:
        longest = 0
        current = 0
        for value in mask:
            if value:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 0
        return longest / mask.size if mask.size else 0.0

    @njit(cache=True)
    def _numba_quantile(values: np.ndarray, fraction: float) -> float:
        ordered = np.sort(values.copy())
        position = (ordered.size - 1) * fraction
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    @njit(cache=True)
    def _numba_horizon_score(oof: np.ndarray, query: np.ndarray) -> tuple[float, float, float]:
        pooled = oof.reshape(oof.size)
        center = _numba_quantile(pooled, 0.5)
        scale = 1.4826 * _numba_quantile(np.abs(pooled - center), 0.5)
        normalized = (oof - center) / scale
        query_normalized = (query - center) / scale
        threshold = _numba_quantile(normalized.reshape(normalized.size), 0.99)
        scores = np.empty(oof.shape[0])
        for row in range(oof.shape[0]):
            values = normalized[row]
            scores[row] = _numba_quantile(values, 0.95) + 2.0 * _numba_persistence(values >= threshold)
        query_score = _numba_quantile(query_normalized, 0.95) + 2.0 * _numba_persistence(query_normalized >= threshold)
        return query_score, (1.0 + np.sum(scores >= query_score)) / (1.0 + scores.size), _numba_persistence(query_normalized >= threshold)


def persistence_numba(mask: np.ndarray) -> float:
    _validate_mask(mask)
    if njit is None:
        raise RuntimeError("numba is unavailable; CPU fallback is required")
    return float(_numba_persistence(mask))


def benchmark_scoring_backend(
    oof_residuals: Mapping[int, Sequence[np.ndarray]], query_residuals: Mapping[int, Mapping[int, np.ndarray]], *, repeats: int = 3
) -> BackendAudit:
    """Compare independent NumPy/Numba frozen score-to-candidate paths."""
    oof, queries = _scoring_inputs(oof_residuals, query_residuals)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if njit is None:
        return BackendAudit("cpu", True, 0.0, 1.0, 1.0, True, "numba_unavailable")
    _numba_candidate_coordinates(oof, queries)  # compile before timing
    numpy_coordinates, numpy_scores, numpy_seconds = _timed_scores(_numpy_candidate_coordinates, oof, queries, repeats)
    numba_coordinates, numba_scores, numba_seconds = _timed_scores(_numba_candidate_coordinates, oof, queries, repeats)
    return BackendAudit("numba", numpy_coordinates == numba_coordinates,
        _max_normalized_error(numpy_scores, numba_scores), _spearman(numpy_scores, numba_scores),
        numpy_seconds / numba_seconds if numba_seconds else 0.0, False)


def probe_mps_host() -> MPSProbe:
    built, available = bool(torch.backends.mps.is_built()), bool(torch.backends.mps.is_available())
    if not (built and available):
        return MPSProbe("host", built, available, False, False, True, "mps_unavailable")
    try:
        device = torch.device("mps")
        network = nn.Conv1d(2, 2, 3).to(device)
        values = torch.linspace(-1.0, 1.0, 64).reshape(1, 2, 32).to(device)
        if not _on_device(values, device) or any(not _on_device(item, device) for item in network.parameters()):
            return MPSProbe("host", built, available, False, False, True, "device_fallback")
        output = network(values); output.square().mean().backward(); torch.mps.synchronize()
        finite = bool(torch.isfinite(output).all() and all(item.grad is not None and torch.isfinite(item.grad).all() for item in network.parameters()))
        return MPSProbe("host", built, available, finite, finite, not finite, None if finite else "nonfinite_outputs")
    except RuntimeError:
        return MPSProbe("host", built, available, False, False, True, "mps_runtime_error")


def benchmark_torch_devices(
    groups: Sequence[BenchmarkTrainGroup] | None = None, *, repeats: int = 3, measurement: TorchBenchmarkMeasurement | None = None
) -> BackendAudit:
    """Benchmark all selected synthetic TRAIN representatives, never opaque tensors."""
    if measurement is None:
        selected = select_benchmark_strata(tuple(groups or ()))
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        probe = probe_mps_host()
        if not probe.available:
            measurement = TorchBenchmarkMeasurement(True, 0.0, 1.0, 1.0, False, True, "host", False)
        elif probe.fallback:
            measurement = TorchBenchmarkMeasurement(True, 0.0, 1.0, 1.0, False, True, "host")
        else:
            measurement = _measure_selected_strata(selected, repeats)
    return _audit_torch_measurement(measurement)


def canonical_mps_fallback_reason(audits: Sequence[BackendAudit]) -> str | None:
    frozen = _audits(audits)
    reasons = {audit.fallback_reason for audit in frozen if audit.backend == "cpu"}
    for reason in _MPS_FAILURE_PRECEDENCE:
        if reason in reasons:
            return reason
    return None if any(audit.backend == "mps" for audit in frozen) else "no_host_mps_audit"


def select_canonical_backend(audits: Sequence[BackendAudit]) -> str:
    """Adopt only if required host MPS evidence is uniformly passing."""
    frozen = _audits(audits)
    failure = canonical_mps_fallback_reason(frozen)
    return "mps" if failure is None and any(audit.backend == "mps" for audit in frozen) else "cpu"


def _audit_torch_measurement(measurement: TorchBenchmarkMeasurement) -> BackendAudit:
    if measurement.environment == "isolated":
        reason = "isolated_mps_unavailable" if not measurement.available else "isolated_environment"
    elif not measurement.available: reason = "mps_unavailable"
    elif measurement.fallback: reason = "device_fallback"
    elif not measurement.finite_outputs: reason = "nonfinite_outputs"
    elif not measurement.candidate_identity: reason = "candidate_identity"
    elif measurement.max_normalized_error > FROZEN_CONFIG.mps_max_error: reason = "max_normalized_error"
    elif measurement.spearman < FROZEN_CONFIG.mps_min_spearman: reason = "spearman"
    elif measurement.speedup < FROZEN_CONFIG.mps_min_speedup: reason = "speedup"
    else:
        return BackendAudit("mps", True, measurement.max_normalized_error, measurement.spearman, measurement.speedup, False)
    return BackendAudit("cpu", measurement.candidate_identity, measurement.max_normalized_error, measurement.spearman, measurement.speedup, True, reason)


def _measure_selected_strata(groups: Sequence[BenchmarkTrainGroup], repeats: int) -> TorchBenchmarkMeasurement:
    cpu_values: list[np.ndarray] = []; mps_values: list[np.ndarray] = []
    cpu_coordinates: list[tuple[str, int]] = []; mps_coordinates: list[tuple[str, int]] = []
    cpu_time = mps_time = 0.0; fallback = False
    for group in groups:
        indices = tuple(item.instance_index for item in group.processed.group.series)
        inputs = torch.from_numpy(np.stack([group.processed.channels[index] for index in indices]).copy())
        torch.manual_seed(FROZEN_CONFIG.model_seeds[0])
        model = CausalMultiHorizonCNN(build_horizons(group.length))
        cpu, cpu_ids, elapsed, cpu_fallback = _run_torch_stage(model, inputs, torch.device("cpu"), group.dataset, indices, repeats)
        mps, mps_ids, elapsed_mps, mps_fallback = _run_torch_stage(model, inputs, torch.device("mps"), group.dataset, indices, repeats)
        cpu_values.append(cpu); mps_values.append(mps); cpu_coordinates.extend(cpu_ids); mps_coordinates.extend(mps_ids)
        cpu_time += elapsed; mps_time += elapsed_mps; fallback = fallback or cpu_fallback or mps_fallback
    total_coordinates = sum(group.normal_count for group in groups)
    if not cpu_coordinates or not mps_coordinates or len(cpu_coordinates) >= total_coordinates or len(mps_coordinates) >= total_coordinates:
        raise ValueError("benchmark scoring calibration requires candidate and noncandidate coordinates")
    left, right = np.concatenate(cpu_values), np.concatenate(mps_values)
    return TorchBenchmarkMeasurement(tuple(cpu_coordinates) == tuple(mps_coordinates), _max_normalized_error(left, right), _spearman(left, right), cpu_time / mps_time if mps_time else 0.0, bool(np.isfinite(left).all() and np.isfinite(right).all()), fallback, "host")


def _run_torch_stage(model: nn.Module, inputs: Tensor, device: torch.device, dataset: str, indices: tuple[int, ...], repeats: int) -> tuple[np.ndarray, tuple[tuple[str, int], ...], float, bool]:
    network = copy.deepcopy(model).to(device); fallback = any(not _on_device(item, device) for item in network.parameters())
    outputs: Mapping[int, np.ndarray] | None = None; elapsed = 0.0
    for _ in range(repeats):
        network.zero_grad(set_to_none=True)
        if device.type == "mps": torch.mps.synchronize()
        started = perf_counter(); values = inputs.detach().clone().to(device).requires_grad_(True)
        output = network(values); fallback = fallback or not _on_device(values, device) or any(not _on_device(value, device) for value in output.values())
        sum(value.square().mean() for value in output.values()).backward()
        outputs = {int(h): value.detach().cpu().numpy().copy() for h, value in sorted(output.items())}
        if device.type == "mps": torch.mps.synchronize()
        elapsed += perf_counter() - started
    assert outputs is not None
    residuals = {index: {h: np.abs(outputs[h][position, 0] - inputs[position, 0, h:].numpy()) for h in outputs} for position, index in enumerate(indices)}
    coordinates = tuple((dataset, index) for index in indices if _is_candidate(residuals, index))
    return np.concatenate([outputs[h].reshape(-1) for h in sorted(outputs)]), coordinates, elapsed, fallback


def _is_candidate(residuals: Mapping[int, Mapping[int, np.ndarray]], query_index: int) -> bool:
    scores = []
    for horizon, query in residuals[query_index].items():
        try: score = score_horizon(tuple(values[horizon] for index, values in residuals.items() if index != query_index), query, horizon=horizon)
        except ValueError: return False
        scores.append(score.p_value <= FROZEN_CONFIG.candidate_alpha and score.persistence >= FROZEN_CONFIG.persistence_fraction)
    return sum(scores) >= 2


def _numpy_candidate_coordinates(oof: Mapping[int, tuple[np.ndarray, ...]], queries: Mapping[int, Mapping[int, np.ndarray]]) -> tuple[tuple[int, ...], np.ndarray]:
    coordinates = []; scores = []
    for coordinate, by_horizon in sorted(queries.items()):
        accepted = 0
        for horizon in sorted(oof):
            item = score_horizon(oof[horizon], by_horizon[horizon], horizon=horizon)
            scores.extend((item.score, item.p_value, item.persistence))
            accepted += item.p_value <= FROZEN_CONFIG.candidate_alpha and item.persistence >= FROZEN_CONFIG.persistence_fraction
        if accepted >= 2: coordinates.append(coordinate)
    return tuple(coordinates), np.asarray(scores, dtype=np.float64)


def _numba_candidate_coordinates(oof: Mapping[int, tuple[np.ndarray, ...]], queries: Mapping[int, Mapping[int, np.ndarray]]) -> tuple[tuple[int, ...], np.ndarray]:
    if njit is None: raise RuntimeError("numba is unavailable")
    coordinates = []; scores = []
    for coordinate, by_horizon in sorted(queries.items()):
        accepted = 0
        for horizon in sorted(oof):
            score, p_value, persistence = _numba_horizon_score(np.ascontiguousarray(np.stack(oof[horizon])), np.ascontiguousarray(by_horizon[horizon]))
            scores.extend((score, p_value, persistence)); accepted += p_value <= FROZEN_CONFIG.candidate_alpha and persistence >= FROZEN_CONFIG.persistence_fraction
        if accepted >= 2: coordinates.append(coordinate)
    return tuple(coordinates), np.asarray(scores, dtype=np.float64)


def _scoring_inputs(oof: Mapping[int, Sequence[np.ndarray]], queries: Mapping[int, Mapping[int, np.ndarray]]) -> tuple[dict[int, tuple[np.ndarray, ...]], dict[int, dict[int, np.ndarray]]]:
    if not isinstance(oof, Mapping) or not isinstance(queries, Mapping) or not oof or not queries: raise ValueError("scoring inputs must be non-empty mappings")
    horizons = tuple(sorted(oof))
    if len(horizons) != 3 or any(type(h) is not int or h < 1 for h in horizons): raise ValueError("scoring requires three positive horizons")
    frozen_oof = {h: tuple(np.asarray(value, dtype=np.float64) for value in oof[h]) for h in horizons}
    for h, values in frozen_oof.items():
        if len(values) < 2 or any(value.ndim != 1 or value.size == 0 or not np.isfinite(value).all() or np.any(value < 0) for value in values) or len({value.size for value in values}) != 1: raise ValueError("OOF residuals must be finite aligned vectors")
    frozen_queries = {coordinate: {h: np.asarray(value, dtype=np.float64) for h, value in by_horizon.items()} for coordinate, by_horizon in queries.items()}
    if any(type(coordinate) is not int or tuple(sorted(by_horizon)) != horizons or any(value.ndim != 1 or value.size != frozen_oof[h][0].size or not np.isfinite(value).all() or np.any(value < 0) for h, value in by_horizon.items()) for coordinate, by_horizon in frozen_queries.items()): raise ValueError("query residuals must align with every horizon")
    return frozen_oof, frozen_queries


def _timed_scores(function, oof, queries, repeats):
    started = perf_counter(); result = function(oof, queries)
    for _ in range(repeats - 1): function(oof, queries)
    return result[0], result[1], perf_counter() - started


def _tertiles(groups: Sequence[BenchmarkTrainGroup], value) -> dict[tuple[str, int], int]:
    ordered = sorted(groups, key=lambda group: (value(group), group.dataset, group.length)); width = ceil(len(ordered) / 3)
    return {(group.dataset, group.length): min(position // width, 2) for position, group in enumerate(ordered)}


def _audits(audits: Sequence[BackendAudit]) -> tuple[BackendAudit, ...]:
    frozen = tuple(audits)
    if not frozen or any(not isinstance(audit, BackendAudit) for audit in frozen): raise ValueError("MPS BackendAudit evidence is required")
    return frozen


def _validate_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.dtype(bool): raise ValueError("mask must be a one-dimensional boolean ndarray")


def _max_normalized_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all(): return 1.0
    return float(np.max(np.abs(left - right), initial=0.0) / max(1.0, float(np.max(np.abs(left), initial=0.0))))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all(): return -1.0
    if np.array_equal(left, right): return 1.0
    def ranks(values):
        order = np.argsort(values, kind="stable"); result = np.empty(values.size); start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]: end += 1
            result[order[start:end]] = (start + end - 1) / 2; start = end
        return result
    a, b = ranks(left), ranks(right)
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _on_device(actual: Tensor, requested: torch.device) -> bool:
    return actual.device.type == requested.type and (requested.type != "mps" or actual.device.index in {None, 0})


__all__ = ["BenchmarkTrainGroup", "MPSProbe", "TorchBenchmarkMeasurement", "benchmark_scoring_backend", "benchmark_torch_devices", "canonical_mps_fallback_reason", "persistence_numba", "persistence_numpy", "probe_mps_host", "select_benchmark_strata", "select_canonical_backend"]
