"""Task 3 runtime assessment: Python parity, Numba determinism, and MLX eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import math
import platform
import statistics
import struct
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import llvmlite
import mlx.core as mx
import numba
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
        fixture_shape,
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
            tuple(int(value) for value in fixture_shape),
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


def _package_version_antigravity(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _dominant_peaks_from_acf_mlx_antigravity(acf):
    candidate_values = mx.where(
        (acf[:, 1:24] > 0.0)
        & (acf[:, 1:24] > acf[:, 0:23])
        & (acf[:, 1:24] >= acf[:, 2:25]),
        acf[:, 1:24],
        mx.full((acf.shape[0], 23), -mx.inf),
    )
    best_zero_based = mx.argmax(candidate_values, axis=1)
    best_value = mx.max(candidate_values, axis=1)
    present = mx.isfinite(best_value)
    return mx.where(present, best_zero_based + 2, 0), present


def _peak_windows_antigravity(batch):
    peak_columns = []
    presence_columns = []
    for start in (0, 84):
        window = batch[:, start : start + 56]
        center = mx.median(window, axis=1, keepdims=True)
        mad = mx.median(mx.abs(window - center), axis=1, keepdims=True)
        z = (window - center) / (1.4826 * mad)
        denominator = mx.sum(z * z, axis=1)
        acf_columns = []
        for lag in range(1, 26):
            numerator = mx.sum(z[:, : 56 - lag] * z[:, lag:56], axis=1)
            acf_columns.append((numerator / denominator).reshape((batch.shape[0], 1)))
        acf = mx.concatenate(acf_columns, axis=1)
        peaks, present = _dominant_peaks_from_acf_mlx_antigravity(acf)
        peak_columns.append(peaks.reshape((batch.shape[0], 1)))
        presence_columns.append(present.reshape((batch.shape[0], 1)))
    return (
        mx.concatenate(peak_columns, axis=1),
        mx.concatenate(presence_columns, axis=1),
    )


def _complete_mlx_outputs_antigravity(fit_rows, query_rows):
    fit = mx.array(np.ascontiguousarray(fit_rows), dtype=mx.float64)
    query = mx.array(np.ascontiguousarray(query_rows), dtype=mx.float64)
    fit_peaks, fit_presence = _peak_windows_antigravity(fit)
    query_peaks, query_presence = _peak_windows_antigravity(query)
    fit_shifts = fit_peaks[:, 1].astype(mx.int32) - fit_peaks[:, 0].astype(mx.int32)
    query_shifts = query_peaks[:, 1].astype(mx.int32) - query_peaks[:, 0].astype(mx.int32)
    fit_shift_float = fit_shifts.astype(mx.float64)
    query_shift_float = query_shifts.astype(mx.float64)

    fit_raw_parts = []
    for excluded in range(int(fit_rows.shape[0])):
        others = mx.concatenate(
            (fit_shift_float[:excluded], fit_shift_float[excluded + 1 :]), axis=0
        )
        fit_raw_parts.append(
            mx.abs(fit_shift_float[excluded] - mx.median(others)).reshape((1,))
        )
    fit_raw = mx.concatenate(fit_raw_parts, axis=0)
    fit_center = mx.median(fit_shift_float)
    fit_maximum = mx.max(fit_raw)
    query_raw = mx.abs(query_shift_float - fit_center)
    counts = mx.sum(
        fit_raw.reshape((1, -1)) <= query_raw.reshape((-1, 1)), axis=1
    ).astype(mx.float64)
    ranks = (1.0 + counts) / (float(fit_rows.shape[0]) + 1.0)
    confirmed = (query_raw > 0.0) & (query_raw > fit_maximum)
    return (
        fit_peaks,
        fit_presence,
        query_peaks,
        query_presence,
        fit_shifts,
        query_shifts,
        fit_raw,
        query_raw,
        ranks,
        confirmed,
        fit_maximum,
    )


def _host_mlx_outputs_antigravity(outputs):
    mx.eval(*outputs)
    return tuple(np.asarray(value) for value in outputs)


def _mlx_output_bytes_antigravity(host_outputs):
    chunks = []
    for value in host_outputs:
        array = np.ascontiguousarray(np.asarray(value))
        metadata_bytes = (array.dtype.str + ":" + repr(array.shape)).encode("ascii")
        data = array.tobytes(order="C")
        chunks.append(struct.pack("<Q", len(metadata_bytes)) + metadata_bytes)
        chunks.append(struct.pack("<Q", len(data)) + data)
    return b"".join(chunks)


def _ineligible_mlx_assessment_antigravity(reason, *, available):
    return BackendAssessmentAntigravity(
        backend="mlx",
        available=available,
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
        return _ineligible_mlx_assessment_antigravity(
            "mlx_unavailable", available=False
        )

    try:
        cold_started = time.perf_counter()
        first_host = _host_mlx_outputs_antigravity(
            _complete_mlx_outputs_antigravity(fit_rows, query_rows)
        )
        cold_seconds = time.perf_counter() - cold_started
    except Exception as error:
        reason = str(error)
        if "float64" in reason:
            reason = "float64_unsupported_on_" + _mlx_device_name_antigravity()
        return _ineligible_mlx_assessment_antigravity(reason, available=True)

    (
        fit_peaks,
        fit_presence,
        query_peaks,
        query_presence,
        fit_shifts,
        query_shifts,
        fit_raw,
        query_raw,
        ranks,
        confirmed,
        maximum,
    ) = first_host
    fit_raw = np.asarray(fit_raw, dtype=np.float64).reshape(-1)
    query_raw = np.asarray(query_raw, dtype=np.float64).reshape(-1)
    ranks = np.asarray(ranks, dtype=np.float64).reshape(-1)
    confirmed = np.asarray(confirmed, dtype=np.bool_).reshape(-1)
    maximum_value = float(np.asarray(maximum, dtype=np.float64).reshape(()))
    finite = bool(
        np.all(np.isfinite(fit_raw))
        and np.all(np.isfinite(query_raw))
        and np.all(np.isfinite(ranks))
        and math.isfinite(maximum_value)
    )
    exact_indices = bool(
        np.asarray(fit_peaks[:, 0]).tolist()
        == numba_result.fit_early_peak_lags.tolist()
        and np.asarray(fit_peaks[:, 1]).tolist()
        == numba_result.fit_late_peak_lags.tolist()
        and np.asarray(fit_presence, dtype=np.bool_).tolist()
        == numba_result.fit_peak_presence.tolist()
        and np.asarray(query_peaks[:, 0]).tolist()
        == numba_result.query_early_peak_lags.tolist()
        and np.asarray(query_peaks[:, 1]).tolist()
        == numba_result.query_late_peak_lags.tolist()
        and np.asarray(query_presence, dtype=np.bool_).tolist()
        == numba_result.query_peak_presence.tolist()
        and np.asarray(fit_shifts).reshape(-1).tolist()
        == numba_result.fit_shifts.tolist()
        and np.asarray(query_shifts).reshape(-1).tolist()
        == numba_result.query_shifts.tolist()
        and ranks.tolist() == numba_result.query_ranks.tolist()
        and confirmed.tolist() == numba_result.confirmation_mask.tolist()
    )
    raw_error = float(
        max(
            float(np.max(np.abs(fit_raw - numba_result.fit_raw_shift)))
            if fit_raw.size
            else 0.0,
            float(np.max(np.abs(query_raw - numba_result.query_raw_shift)))
            if query_raw.size
            else 0.0,
            abs(maximum_value - float(numba_result.fit_normal_max_raw_shift)),
        )
    )

    timing_samples = []
    deterministic = True
    first_bytes = _mlx_output_bytes_antigravity(first_host)
    for _ in range(repetitions):
        repetition_started = time.perf_counter()
        repeated_host = _host_mlx_outputs_antigravity(
            _complete_mlx_outputs_antigravity(fit_rows, query_rows)
        )
        timing_samples.append(time.perf_counter() - repetition_started)
        deterministic = deterministic and (
            _mlx_output_bytes_antigravity(repeated_host) == first_bytes
        )
    warm_median, warm_iqr = _timing_summary_antigravity(timing_samples)
    speedup = (
        float(numba_warm_median_seconds / warm_median)
        if warm_median > 0.0
        else math.inf
    )

    reasons = []
    if not finite:
        reasons.append("nonfinite_output")
    if raw_error > 1e-12:
        reasons.append("raw_parity_tolerance_exceeded")
    if not exact_indices:
        reasons.append("candidate_index_mismatch")
    if not deterministic:
        reasons.append("nondeterministic_output")
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
        deterministic=deterministic,
        cold_seconds=cold_seconds,
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
        fixture_shape=fixture.shape,
    )


def score_bytes_antigravity(result):
    arrays = (
        result.fit_early_peak_lags, result.fit_late_peak_lags, result.fit_peak_presence,
        result.query_early_peak_lags, result.query_late_peak_lags, result.query_peak_presence,
        result.fit_shifts, result.query_shifts, result.fit_raw_shift,
        result.query_raw_shift, result.query_ranks, result.confirmation_mask,
    )
    chunks = []
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        description = (array.dtype.str + ":" + repr(array.shape)).encode("ascii")
        data = array.tobytes(order="C")
        chunks.extend(
            (
                struct.pack("<Q", len(description)),
                description,
                struct.pack("<Q", len(data)),
                data,
            )
        )
    backend = result.backend.encode("utf-8")
    scalar = struct.pack("<d", float(result.fit_normal_max_raw_shift))
    chunks.extend(
        (
            struct.pack("<Q", len(backend)),
            backend,
            struct.pack("<Q", len(scalar)),
            scalar,
        )
    )
    return b"".join(chunks)


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
            "mlx_metal_version": _package_version_antigravity("mlx-metal"),
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
                "available": backend.available,
                "finite": backend.finite,
                "python_reference_passed": backend.python_reference_passed,
                "parity_passed": backend.parity_passed,
                "exact_candidate_indices": backend.exact_candidate_indices,
                "deterministic": backend.deterministic,
                "eligible": backend.eligible,
                "fallback_reason": backend.fallback_reason,
                "max_raw_abs_error": backend.max_raw_abs_error,
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
