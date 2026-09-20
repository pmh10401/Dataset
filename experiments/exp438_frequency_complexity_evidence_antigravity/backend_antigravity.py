"""Deterministic Exp438 backend readiness evidence and selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendBatchEvidence, BackendDecision


@dataclass(frozen=True)
class BackendFixtureAntigravity:
    values: NDArray[np.float32]
    thresholds: NDArray[np.float32]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        thresholds = np.asarray(self.thresholds, dtype=np.float32)
        if values.ndim != 1 or values.size == 0 or thresholds.shape != values.shape:
            raise ValueError("backend fixture shape is invalid")
        if not np.isfinite(values).all() or not np.isfinite(thresholds).all():
            raise ValueError("backend fixture must be finite")
        values = np.ascontiguousarray(values).copy()
        thresholds = np.ascontiguousarray(thresholds).copy()
        values.flags.writeable = False
        thresholds.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "thresholds", thresholds)


def _output_antigravity(value: object, expected_shape: tuple[int, ...]) -> NDArray[np.float32]:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("backend output is invalid") from exc
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise ValueError("backend output is invalid")
    return np.ascontiguousarray(result)


def benchmark_candidate_antigravity(
    backend: str,
    fixture: BackendFixtureAntigravity,
    cpu_runner: Callable[[BackendFixtureAntigravity, int], object],
    device_runner: Callable[[BackendFixtureAntigravity, int], object],
    *,
    batch_sizes: Iterable[int] = FROZEN_CONFIG.batch_sizes,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] = lambda: None,
) -> tuple[BackendBatchEvidence, ...]:
    if backend not in {"numba", "mlx", "mps"}:
        raise ValueError("benchmark backend is invalid")
    if type(fixture) is not BackendFixtureAntigravity:
        raise ValueError("benchmark fixture is invalid")
    sizes = tuple(batch_sizes)
    if not sizes or any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("benchmark batch sizes are invalid")
    evidence: list[BackendBatchEvidence] = []
    expected_shape = fixture.values.shape
    for batch_size in sizes:
        failure: str | None = None
        cpu_times: list[float] = []
        device_times: list[float] = []
        maximum_error = math.inf
        finite = False
        parity = False
        try:
            for _ in range(FROZEN_CONFIG.warmup_runs):
                _output_antigravity(device_runner(fixture, batch_size), expected_shape)
                synchronize()
            maximum_error = 0.0
            finite = True
            parity = True
            for _ in range(FROZEN_CONFIG.timed_runs):
                start = float(clock())
                cpu_output = _output_antigravity(cpu_runner(fixture, batch_size), expected_shape)
                cpu_elapsed = float(clock()) - start
                start = float(clock())
                device_output = _output_antigravity(device_runner(fixture, batch_size), expected_shape)
                synchronize()
                device_elapsed = float(clock()) - start
                if not (math.isfinite(cpu_elapsed) and cpu_elapsed > 0.0 and math.isfinite(device_elapsed) and device_elapsed > 0.0):
                    raise ValueError("backend timing is invalid")
                cpu_times.append(cpu_elapsed)
                device_times.append(device_elapsed)
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(cpu_output.astype(np.float64) - device_output.astype(np.float64)))),
                )
                cpu_candidates = tuple(int(index) for index in np.flatnonzero(cpu_output > fixture.thresholds))
                device_candidates = tuple(int(index) for index in np.flatnonzero(device_output > fixture.thresholds))
                parity = parity and cpu_candidates == device_candidates
        except (RuntimeError, ValueError, TypeError) as exc:
            failure = str(exc) or "backend_runtime_failure"
            finite = False
            parity = False
        if failure is not None:
            cpu_median = 1.0
            device_median = 1.0
            speedup = 1.0
            maximum_error = 1e30 if not math.isfinite(maximum_error) else maximum_error
            passed = False
            reason = failure
        else:
            cpu_median = float(np.median(np.asarray(cpu_times, dtype=np.float64)))
            device_median = float(np.median(np.asarray(device_times, dtype=np.float64)))
            speedup = cpu_median / device_median
            passed = (
                finite
                and parity
                and maximum_error <= FROZEN_CONFIG.parity_tolerance
                and speedup >= FROZEN_CONFIG.minimum_speedup
            )
            failures = []
            if not finite:
                failures.append("nonfinite")
            if not parity:
                failures.append("candidate_mismatch")
            if maximum_error > FROZEN_CONFIG.parity_tolerance:
                failures.append("score_tolerance")
            if speedup < FROZEN_CONFIG.minimum_speedup:
                failures.append("speedup_below_1.1")
            reason = None if passed else ",".join(failures)
        evidence.append(
            BackendBatchEvidence(
                backend=backend,
                batch_size=batch_size,
                finite=finite,
                candidate_parity=parity,
                maximum_absolute_error=maximum_error,
                cpu_median_seconds=cpu_median,
                device_median_seconds=device_median,
                speedup=speedup,
                passed=passed,
                reason=reason,
            )
        )
    return tuple(evidence)


def select_backend_antigravity(
    evidence: Iterable[BackendBatchEvidence],
) -> BackendDecision:
    items = tuple(evidence)
    if any(type(item) is not BackendBatchEvidence for item in items):
        raise ValueError("backend selection evidence is invalid")
    passing = tuple(item for item in items if item.passed)
    if not passing:
        return BackendDecision(
            selected_backend="cpu",
            selected_batch_size=None,
            evidence=items,
            fallback_reason="no candidate passed parity and speed gates",
        )
    selected = min(
        passing,
        key=lambda item: (
            item.device_median_seconds,
            item.backend,
            item.batch_size,
        ),
    )
    return BackendDecision(
        selected_backend=selected.backend,
        selected_batch_size=selected.batch_size,
        evidence=items,
        fallback_reason=None,
    )


__all__ = [
    "BackendFixtureAntigravity",
    "benchmark_candidate_antigravity",
    "select_backend_antigravity",
]
