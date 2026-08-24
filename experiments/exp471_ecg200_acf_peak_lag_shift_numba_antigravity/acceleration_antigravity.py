"""Task 3 runtime assessment: Python parity, Numba determinism, and MLX eligibility."""

from __future__ import annotations

from dataclasses import dataclass
import math
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import llvmlite
import mlx.core as mx
import numba
import numpy as np
import numpy as np

from .acf_peak_lag_antigravity import (
    score_acf_peak_lag_numba_antigravity,
)
from .python_reference_antigravity import score_acf_peak_lag_python_reference_antigravity

__all__ = (
    "BackendAssessmentAntigravity",
    "RuntimeAssessmentAntigravity",
    "assess_runtime_antigravity",
    "fixed_reference_rows_antigravity",
    "manifest_payload_antigravity",
)

_PERIODS_ANTIGRAVITY = ((8.0, 8.0), (9.0, 9.0), (7.0, 11.0), (8.0, 10.0), (10.0, 10.0), (12.0, 12.0))


def fixed_reference_rows_antigravity() -> np.ndarray:
    positions = np.arange(140, dtype=np.float64)
    rows = [np.sin(2.0 * math.pi * positions / 9.0) for _ in range(6)]
    for row, (early_period, late_period) in zip(rows, _PERIODS_ANTIGRAVITY, strict=True):
        row[0:56] = np.sin(2.0 * math.pi * np.arange(56, dtype=np.float64) / early_period)
        row[84:140] = np.sin(2.0 * math.pi * np.arange(56, dtype=np.float64) / late_period)
    fixture = np.ascontiguousarray(np.stack(rows), dtype=np.float64)
    fixture.flags.writeable = False
    return fixture


def max_abs_raw_error_antigravity(actual, expected):
    fit_error = max(
        abs(float(left) - float(right))
        for left, right in zip(actual.fit_raw_shift, expected.fit_raw_shift)
    )
    query_error = max(
        abs(float(left) - float(right))
        for left, right in zip(actual.query_raw_shift, expected.query_raw_shift)
    )
    maximum_error = abs(
        float(actual.fit_normal_max_raw_shift)
        - float(expected.fit_normal_max_raw_shift)
    )
    return max(fit_error, query_error, maximum_error)


def exact_discrete_antigravity(actual, expected):
    return all(
        (
            tuple(actual.fit_early_peak_lags) == tuple(expected.fit_early_peak_lags),
            tuple(actual.fit_late_peak_lags) == tuple(expected.fit_late_peak_lags),
            actual.fit_peak_presence.tolist() == expected.fit_peak_presence.tolist(),
            tuple(actual.query_early_peak_lags) == tuple(expected.query_early_peak_lags),
            tuple(actual.query_late_peak_lags) == tuple(expected.query_late_peak_lags),
            actual.query_peak_presence.tolist() == expected.query_peak_presence.tolist(),
            tuple(actual.fit_shifts) == tuple(expected.fit_shifts),
            tuple(actual.query_shifts) == tuple(expected.query_shifts),
            tuple(actual.query_ranks) == tuple(expected.query_ranks),
            actual.confirmation_mask.tolist() == expected.confirmation_mask.tolist(),
        )
    )


@dataclass(frozen=True, slots=True)
class BackendAssessmentAntigravity:
    backend: str
    available: bool
    finite: bool
    python_reference_passed: bool | None
    parity_passed: bool
    exact_candidate_indices: bool
    max_raw_abs_error: float | None
    deterministic: bool
    cold_seconds: float | None
    warm_median_seconds: float | None
    warm_iqr_seconds: float | None
    speedup: float | None
    eligible: bool
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if type(self.backend) is not str or not self.backend:
            raise ValueError("backend must be a non-empty string")
        if type(self.available) is not bool:
            raise ValueError("available must be bool")
        for name in ("finite", "parity_passed", "exact_candidate_indices", "deterministic", "eligible"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise ValueError(name + " must be bool")
        if self.python_reference_passed is not None and type(self.python_reference_passed) is not bool:
            raise ValueError("python_reference_passed must be bool or None")
        numeric_fields = (
            "max_raw_abs_error",
            "cold_seconds",
            "warm_median_seconds",
            "warm_iqr_seconds",
            "speedup",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(name + " must be a finite number or None")
        if type(self.fallback_reason) is not str and self.fallback_reason is not None:
            raise ValueError("fallback_reason must be string or None")


@dataclass(frozen=True, slots=True)
class RuntimeAssessmentAntigravity:
    selected_backend: str
    backends: Mapping[str, BackendAssessmentAntigravity]
    fixture_shape: tuple[int, int]
    required_mlx_speedup: float

    @classmethod
    def from_measurements(
        cls,
        first_result,
        *,
        numba_cold_seconds,
        numba_warm_median_seconds,
        numba_warm_iqr_seconds,
        reference_error,
        reference_exact,
        reference_passed,
        mlx_assessment,
        required_mlx_speedup,
        input_shape,
    ):
        finite = bool(
            np.all(np.isfinite(first_result.fit_raw_shift))
            and np.all(np.isfinite(first_result.query_raw_shift))
        )
        parity = bool(reference_error <= 1e-12 and reference_exact and reference_passed)
        eligible = finite and parity
        numba_backend = BackendAssessmentAntigravity(
            backend="numba",
            available=True,
            finite=finite,
            python_reference_passed=bool(reference_passed),
            parity_passed=parity,
            exact_candidate_indices=bool(reference_exact),
            max_raw_abs_error=float(reference_error),
            deterministic=True,
            cold_seconds=float(numba_cold_seconds),
            warm_median_seconds=float(numba_warm_median_seconds),
            warm_iqr_seconds=float(numba_warm_iqr_seconds),
            speedup=None,
            eligible=eligible,
            fallback_reason=None if eligible else "reference_or_finite_gate_failed",
        )
        backends = {"numba": numba_backend, "mlx": mlx_assessment}
        return cls(
            "numba",
            MappingProxyType(backends),
            tuple(int(value) for value in input_shape),
            float(required_mlx_speedup),
        )

    def __post_init__(self) -> None:
        if self.selected_backend != "numba":
            raise ValueError("Exp471 production selection remains fixed to Numba")
        if not isinstance(self.backends, Mapping) or set(self.backends) != {"numba", "mlx"}:
            raise ValueError("runtime assessment records exactly Numba and MLX")
        if type(self.fixture_shape) is not tuple or len(self.fixture_shape) != 2:
            raise ValueError("fixture_shape must be an input shape tuple")
        if type(self.required_mlx_speedup) is not float or not math.isfinite(
            self.required_mlx_speedup
        ) or self.required_mlx_speedup <= 0.0:
            raise ValueError("required_mlx_speedup must be finite and positive")


def _timing_summary_antigravity(timings: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(float(value) for value in timings)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return median, q3 - q1


def _mlx_available_antigravity() -> bool:
    try:
        mx.eval(mx.array([1.0], dtype=mx.float32))
    except Exception:
        return False
    return True


def _mlx_device_name_antigravity() -> str:
    try:
        device = mx.default_device()
    except Exception:
        return "unavailable"
    rendered = str(device).removeprefix("Device(").removesuffix(")").strip()
    if rendered.startswith("mlx.") or rendered.startswith("gpu.") or rendered.startswith("cpu."):
        rendered = rendered.split(".", 1)[1]
    return rendered or str(device)


def _peak_windows_antigravity(batch):
    outputs = []
    for start in (0, 84):
        window = batch[:, start : start + 56]
        center = mx.median(window, axis=1, keepdims=True)
        deviations = mx.abs(window - center)
        mad = mx.median(deviations, axis=1, keepdims=True)
        scale = 1.4826 * mad
        z = (window - center) / scale
        denominator = mx.sum(z * z, axis=1)
        candidate_values = []
        for lag in range(2, 25):
            numerator = mx.sum(z[:, :56 - lag] * z[:, lag:56], axis=1)
            candidate_values.append((numerator / denominator).reshape((batch.shape[0], 1)))
        acfs = mx.concatenate(candidate_values, axis=1)
        previous = mx.concatenate(
            (
                mx.full((batch.shape[0], 1), -mx.inf),
                acfs[:, :-1],
            ),
            axis=1,
        )
        following = mx.roll(acfs, -1, axis=1)
        candidates = mx.where(
            (acfs > 0.0) & (acfs > previous) & (acfs >= following),
            acfs,
            mx.full(acfs.shape, -mx.inf),
        )
        best_zero_based = mx.argmax(candidates, axis=1)
        best_value = mx.max(candidates, axis=1)
        outputs.append(mx.where(best_value > 0.0, best_zero_based + 2, 0).reshape((batch.shape[0], 1)))
    return mx.concatenate(outputs, axis=1)


def _ineligible_mlx_assessment_antigravity(reason):
    return BackendAssessmentAntigravity(
        backend="mlx",
        available=False,
        finite=False,
        python_reference_passed=None,
        parity_passed=False,
        exact_candidate_indices=False,
        max_raw_abs_error=None,
        deterministic=False,
        cold_seconds=None,
        warm_median_seconds=None,
        warm_iqr_seconds=None,
        speedup=None,
        eligible=False,
        fallback_reason=reason,
    )


def assess_mlx_smoke_antigravity(
    fit_rows,
    query_rows,
    numba_result,
    numba_warm_median_seconds,
    required_speedup,
    repetitions,
):
    if not _mlx_available_antigravity():
        return _ineligible_mlx_assessment_antigravity("mlx_unavailable")

    started = time.perf_counter()
    try:
        mlx_fit64 = mx.array(np.ascontiguousarray(fit_rows), dtype=mx.float64)
        mlx_query64 = mx.array(np.ascontiguousarray(query_rows), dtype=mx.float64)
        mx.eval(mlx_fit64, mlx_query64)
        transfer_seconds = time.perf_counter() - started

        def shifts_for(batch):
            peaks = _peak_windows_antigravity(batch)
            return peaks[:, 1:] - peaks[:, :1]

        fit_shifts_mx = shifts_for(mlx_fit64)
        query_shifts_mx = shifts_for(mlx_query64)
        mx.eval(fit_shifts_mx, query_shifts_mx)
        fit_shifts = np.asarray(fit_shifts_mx, dtype=np.int64).reshape(-1)
        query_shifts = np.asarray(query_shifts_mx, dtype=np.int64).reshape(-1)
    except (ValueError, RuntimeError) as error:
        reason = str(error)
        if "float64" in reason:
            reason = "float64_unsupported_on_" + _mlx_device_name_antigravity()
        return _ineligible_mlx_assessment_antigravity(reason)

    fit_shift_values = fit_shifts.astype(np.float64)
    query_shift_values = query_shifts.astype(np.float64)
    fit_raw = np.empty(fit_shift_values.size, dtype=np.float64)
    for excluded in range(fit_shift_values.size):
        others = np.delete(fit_shift_values, excluded, axis=0)
        fit_raw[excluded] = abs(float(fit_shift_values[excluded]) - float(np.median(others)))
    fit_center = float(np.median(fit_shift_values))
    maximum = float(np.max(fit_raw))
    query_raw = np.abs(query_shift_values - fit_center)
    ranks = (1.0 + np.sum(fit_raw[None, :] <= query_raw[:, None], axis=1).astype(np.float64)) / (float(fit_raw.size) + 1.0)
    confirmed = (query_raw > 0.0) & (query_raw > maximum)

    finite = bool(
        np.all(np.isfinite(fit_raw))
        and np.all(np.isfinite(query_raw))
        and np.all(np.isfinite(ranks))
    )
    exact_indices = bool(
        tuple(fit_shifts.tolist()) == tuple(numba_result.fit_shifts)
        and tuple(query_shifts.tolist()) == tuple(numba_result.query_shifts)
        and tuple(ranks.tolist()) == tuple(numba_result.query_ranks)
        and confirmed.tolist() == numba_result.confirmation_mask.tolist()
    )
    raw_error = float(
        max(
            float(np.max(np.abs(fit_raw - np.asarray(numba_result.fit_raw_shift, dtype=np.float64)))) if fit_raw.size else 0.0,
            float(np.max(np.abs(query_raw - np.asarray(numba_result.query_raw_shift, dtype=np.float64)))) if query_raw.size else 0.0,
            abs(maximum - float(numba_result.fit_normal_max_raw_shift)),
        )
    )

    timing_samples = []
    for _ in range(repetitions):
        repetition_started = time.perf_counter()
        repeated_shifts = shifts_for(mlx_fit64)
        mx.eval(repeated_shifts)
        timing_samples.append(time.perf_counter() - repetition_started)
    warm_median, warm_iqr = _timing_summary_antigravity(timing_samples)
    total_elapsed = transfer_seconds + warm_median
    speedup = float(numba_warm_median_seconds / total_elapsed) if total_elapsed > 0.0 else math.inf

    reasons = []
    if not _mlx_available_antigravity():
        reasons.append("mlx_unavailable")
    if mlx_fit64.dtype != mx.float64:
        reasons.append("float64_unsupported")
    if not finite:
        reasons.append("nonfinite_output")
    if raw_error > 1e-12:
        reasons.append("raw_parity_tolerance_exceeded")
    if not exact_indices:
        reasons.append("candidate_index_mismatch")
    if speedup < required_speedup:
        reasons.append(f"speedup_below_{required_speedup:.1f}x")
    eligible = not reasons
    return BackendAssessmentAntigravity(
        backend="mlx",
        available=_mlx_available_antigravity(),
        finite=finite,
        python_reference_passed=None,
        parity_passed=raw_error <= 1e-12 and exact_indices,
        exact_candidate_indices=exact_indices,
        max_raw_abs_error=raw_error,
        deterministic=True,
        cold_seconds=None,
        warm_median_seconds=warm_median,
        warm_iqr_seconds=warm_iqr,
        speedup=speedup if math.isfinite(speedup) else 1e300,
        eligible=eligible,
        fallback_reason=";".join(reasons) if reasons else None,
    )


def assess_runtime_antigravity(
    fit_rows,
    query_rows,
    repetitions=21,
    reference_tolerance=1e-12,
    required_mlx_speedup=1.1,
):
    _require_strict_runtime_arguments_antigravity(
        repetitions, reference_tolerance, required_mlx_speedup
    )
    input_shape = tuple(np.asarray(fit_rows).shape)
    started = time.perf_counter()
    first = score_acf_peak_lag_numba_antigravity(fit_rows, query_rows)
    cold_seconds = time.perf_counter() - started
    second = score_acf_peak_lag_numba_antigravity(fit_rows, query_rows)
    if score_bytes_antigravity(first) != score_bytes_antigravity(second):
        raise RuntimeError("Numba determinism check failed")

    fixture = fixed_reference_rows_antigravity()
    expected = score_acf_peak_lag_python_reference_antigravity(fixture[:3], fixture[3:])
    observed = score_acf_peak_lag_numba_antigravity(fixture[:3], fixture[3:])
    error = max_abs_raw_error_antigravity(observed, expected)
    discrete_equal = exact_discrete_antigravity(observed, expected)
    reference_passed = error <= reference_tolerance and discrete_equal
    if not reference_passed:
        raise RuntimeError("Python reference check failed")

    timings = []
    for _ in range(repetitions):
        repetition_started = time.perf_counter()
        score_acf_peak_lag_numba_antigravity(fit_rows, query_rows)
        timings.append(time.perf_counter() - repetition_started)
    warm_median, warm_iqr = _timing_summary_antigravity(timings)
    mlx_assessment = assess_mlx_smoke_antigravity(
        fit_rows,
        query_rows,
        first,
        warm_median,
        required_mlx_speedup,
        repetitions,
    )
    return RuntimeAssessmentAntigravity.from_measurements(
        first,
        numba_cold_seconds=cold_seconds,
        numba_warm_median_seconds=warm_median,
        numba_warm_iqr_seconds=warm_iqr,
        reference_error=error,
        reference_exact=discrete_equal,
        reference_passed=reference_passed,
        mlx_assessment=mlx_assessment,
        required_mlx_speedup=float(required_mlx_speedup),
        input_shape=input_shape,
    )


def score_bytes_antigravity(result):
    fields = (
        result.fit_early_peak_lags, result.fit_late_peak_lags, result.fit_peak_presence,
        result.query_early_peak_lags, result.query_late_peak_lags, result.query_peak_presence,
        result.fit_shifts, result.query_shifts, result.fit_raw_shift,
        result.query_raw_shift, result.query_ranks, result.confirmation_mask,
        result.backend.encode("ascii"),
        np.asarray(result.fit_normal_max_raw_shift, dtype=np.float64).tobytes(order="C"),
    )
    return b"".join(
        field if isinstance(field, bytes) else np.asarray(field, order="C").tobytes()
        for field in fields
    )


def _require_strict_runtime_arguments_antigravity(
    repetitions,
    reference_tolerance,
    required_mlx_speedup,
):
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if type(reference_tolerance) not in (int, float) or not math.isfinite(
        float(reference_tolerance)
    ):
        raise ValueError("reference_tolerance must be finite")
    if type(required_mlx_speedup) not in (int, float) or not math.isfinite(
        float(required_mlx_speedup)
    ) or float(required_mlx_speedup) <= 0.0:
        raise ValueError("required_mlx_speedup must be finite and positive")


def manifest_payload_antigravity(result):
    return {
        "assessment": {
            "selected_backend": result.selected_backend,
            "fixture_shape": list(result.fixture_shape),
        },
        "environment": {
            "numba_version": numba.__version__,
            "numpy_version": np.__version__,
            "llvmlite_version": llvmlite.__version__,
            "mlx_version": mx.__version__,
            "device": _mlx_device_name_antigravity(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "timing": {
            "backend": {
                name: {
                    "cold_seconds": backend.cold_seconds,
                    "warm_median_seconds": backend.warm_median_seconds,
                    "warm_iqr_seconds": backend.warm_iqr_seconds,
                    "speedup": backend.speedup,
                }
                for name, backend in result.backends.items()
            },
        },
        "parity": {
            name: {
                "finite": backend.finite,
                "python_reference_passed": backend.python_reference_passed,
                "parity_passed": backend.parity_passed,
                "exact_candidate_indices": backend.exact_candidate_indices,
                "deterministic": backend.deterministic,
                "eligible": backend.eligible,
                "fallback_reason": backend.fallback_reason,
            }
            for name, backend in result.backends.items()
        },
        "selected_backend": result.selected_backend,
        "required_mlx_speedup": result.required_mlx_speedup,
        "private_runs": 0,
        "official_runs": 0,
        "test_fn_access": 0,
        "routing_runs": 0,
    }


def main_antigravity():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scratch/exp471_acceleration_assessment_antigravity.json")
    arguments = parser.parse_args()
    fixture = fixed_reference_rows_antigravity()
    assessment = assess_runtime_antigravity(fixture[:3], fixture[3:], repetitions=21)
    payload = manifest_payload_antigravity(assessment)
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main_antigravity()
