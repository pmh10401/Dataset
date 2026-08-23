"""Sequential backend parity and cold acceleration gate for Exp468 S10."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .local_stft_antigravity import (
    SpectralScoresAntigravity,
    _score_from_power_numpy_antigravity,
    score_local_stft_antigravity,
)


FloatArrayAntigravity = NDArray[np.float64]
REQUIRED_SPEEDUP_ANTIGRAVITY = 1.1


def _owned_rows_antigravity(value: object, *, name: str) -> FloatArrayAntigravity:
    rows = np.asarray(value)
    if rows.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have exact float64 dtype")
    if rows.ndim != 2 or rows.shape[1:] != (140,):
        raise ValueError(f"{name} must have shape (n, 140)")
    if rows.shape[0] < 1 or not np.isfinite(rows).all():
        raise ValueError(f"{name} must be nonempty and finite")
    owned = np.array(rows, dtype=np.float64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class AccelerationFoldAntigravity:
    fit_rows: FloatArrayAntigravity
    query_scopes: tuple[FloatArrayAntigravity, ...]

    def __post_init__(self) -> None:
        fit = _owned_rows_antigravity(self.fit_rows, name="fit_rows")
        if fit.shape[0] < 2:
            raise ValueError("fit_rows must contain at least two rows")
        if type(self.query_scopes) is not tuple or not self.query_scopes:
            raise TypeError("query_scopes must be a nonempty tuple")
        scopes = tuple(
            _owned_rows_antigravity(scope, name=f"query_scopes[{index}]")
            for index, scope in enumerate(self.query_scopes)
        )
        object.__setattr__(self, "fit_rows", fit)
        object.__setattr__(self, "query_scopes", scopes)


@dataclass(frozen=True, slots=True)
class AccelerationWorkloadAntigravity:
    folds: tuple[AccelerationFoldAntigravity, ...]
    numpy_repetitions: int = 21
    warm_repetitions: int = 21

    def __post_init__(self) -> None:
        if type(self.folds) is not tuple or not self.folds:
            raise TypeError("folds must be a nonempty tuple")
        if not all(type(fold) is AccelerationFoldAntigravity for fold in self.folds):
            raise TypeError("folds must contain AccelerationFoldAntigravity values")
        if type(self.numpy_repetitions) is not int or self.numpy_repetitions < 1:
            raise ValueError("numpy_repetitions must be a positive int")
        if type(self.warm_repetitions) is not int or self.warm_repetitions < 1:
            raise ValueError("warm_repetitions must be a positive int")


@dataclass(frozen=True, slots=True)
class BackendAssessmentAntigravity:
    backend: str
    available: bool
    parity_passed: bool
    finite: bool
    max_raw_abs_error: float | None
    rank_exact: bool
    confirmation_exact: bool
    deterministic: bool
    cold_total_seconds: float | None
    cold_compile_seconds: float | None
    warm_median_seconds: float | None
    warm_iqr_seconds: float | None
    gate_seconds: float | None
    speedup: float | None
    eligible: bool
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class AccelerationAssessmentAntigravity:
    required_speedup: float
    workload_counts: Mapping[str, int]
    backends: Mapping[str, BackendAssessmentAntigravity]
    selected_backend: str
    gate_passed: bool

    def __post_init__(self) -> None:
        if self.required_speedup != REQUIRED_SPEEDUP_ANTIGRAVITY:
            raise ValueError("required_speedup differs from frozen 1.1x gate")
        counts = dict(self.workload_counts)
        if set(counts) != {"fit_raw", "query_raw", "transformed_rows", "frames"}:
            raise ValueError("workload_counts keys differ")
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("workload_counts values must be nonnegative ints")
        backends = dict(self.backends)
        if set(backends) != {"numpy", "numba", "mlx"}:
            raise ValueError("assessment must record numpy, numba, and mlx")
        if self.selected_backend not in backends:
            raise ValueError("selected_backend is not assessed")
        if self.gate_passed != (self.selected_backend != "numpy"):
            raise ValueError("gate_passed and selected_backend disagree")
        object.__setattr__(self, "workload_counts", MappingProxyType(counts))
        object.__setattr__(self, "backends", MappingProxyType(backends))


try:
    import numba as _numba_antigravity
except ImportError:  # pragma: no cover - exercised only in dependency-missing environments
    _numba_antigravity = None


if _numba_antigravity is not None:
    _HANN_ANTIGRAVITY = np.hanning(28).astype(np.float64)
    _COS_ANTIGRAVITY = np.array(
        [
            [np.cos(2.0 * np.pi * frequency * n / 28.0) for n in range(28)]
            for frequency in range(1, 15)
        ],
        dtype=np.float64,
    )
    _SIN_ANTIGRAVITY = np.array(
        [
            [np.sin(2.0 * np.pi * frequency * n / 28.0) for n in range(28)]
            for frequency in range(1, 15)
        ],
        dtype=np.float64,
    )

    @_numba_antigravity.njit(cache=False, fastmath=False)
    def _power_numba_antigravity(rows: FloatArrayAntigravity) -> FloatArrayAntigravity:
        output = np.empty((rows.shape[0], 5, 14), dtype=np.float64)
        starts = (0, 14, 28, 42, 56)
        weighted = np.empty(28, dtype=np.float64)
        for row_index in range(rows.shape[0]):
            for frame_index in range(5):
                base = 56 + starts[frame_index]
                mean = 0.0
                for n in range(28):
                    mean += rows[row_index, base + n]
                mean /= 28.0
                for n in range(28):
                    weighted[n] = (rows[row_index, base + n] - mean) * _HANN_ANTIGRAVITY[n]
                total = 0.0
                for bin_index in range(14):
                    real = 0.0
                    imag = 0.0
                    for n in range(28):
                        real += weighted[n] * _COS_ANTIGRAVITY[bin_index, n]
                        imag -= weighted[n] * _SIN_ANTIGRAVITY[bin_index, n]
                    value = real * real + imag * imag + 1e-12
                    output[row_index, frame_index, bin_index] = value
                    total += value
                for bin_index in range(14):
                    output[row_index, frame_index, bin_index] /= total
        return output

    @_numba_antigravity.njit(cache=False, fastmath=False)
    def _js_numba_antigravity(p: FloatArrayAntigravity, q: FloatArrayAntigravity) -> float:
        divergence = 0.0
        for index in range(14):
            midpoint = (p[index] + q[index]) / 2.0
            divergence += 0.5 * p[index] * np.log(p[index] / midpoint)
            divergence += 0.5 * q[index] * np.log(q[index] / midpoint)
        if np.abs(divergence) <= 1e-18:
            return 0.0
        if divergence < 0.0:
            return 0.0
        return divergence

    @_numba_antigravity.njit(cache=False, fastmath=False)
    def _score_arrays_numba_antigravity(
        fit_rows: FloatArrayAntigravity,
        query_rows: FloatArrayAntigravity,
    ) -> tuple[FloatArrayAntigravity, FloatArrayAntigravity]:
        fit_power = _power_numba_antigravity(fit_rows)
        query_power = _power_numba_antigravity(query_rows)
        fit_raw = np.empty(fit_rows.shape[0], dtype=np.float64)
        query_raw = np.empty(query_rows.shape[0], dtype=np.float64)
        reference = np.empty((5, 14), dtype=np.float64)
        scratch = np.empty(fit_rows.shape[0], dtype=np.float64)
        frame_js = np.empty(5, dtype=np.float64)

        for excluded in range(fit_rows.shape[0]):
            for frame in range(5):
                reference_sum = 0.0
                for bin_index in range(14):
                    position = 0
                    for row_index in range(fit_rows.shape[0]):
                        if row_index != excluded:
                            scratch[position] = fit_power[row_index, frame, bin_index]
                            position += 1
                    value = np.median(scratch[:position])
                    reference[frame, bin_index] = value
                    reference_sum += value
                for bin_index in range(14):
                    reference[frame, bin_index] /= reference_sum
                frame_js[frame] = _js_numba_antigravity(
                    fit_power[excluded, frame], reference[frame],
                )
            fit_raw[excluded] = np.median(frame_js)

        for frame in range(5):
            reference_sum = 0.0
            for bin_index in range(14):
                for row_index in range(fit_rows.shape[0]):
                    scratch[row_index] = fit_power[row_index, frame, bin_index]
                value = np.median(scratch[: fit_rows.shape[0]])
                reference[frame, bin_index] = value
                reference_sum += value
            for bin_index in range(14):
                reference[frame, bin_index] /= reference_sum

        for row_index in range(query_rows.shape[0]):
            for frame in range(5):
                frame_js[frame] = _js_numba_antigravity(
                    query_power[row_index, frame], reference[frame],
                )
            query_raw[row_index] = np.median(frame_js)
        return fit_raw, query_raw


def _score_local_stft_numba_antigravity(
    fit_rows: FloatArrayAntigravity,
    query_rows: FloatArrayAntigravity,
) -> SpectralScoresAntigravity:
    if _numba_antigravity is None:
        raise RuntimeError("Numba is not installed")
    # Profile-guided second path: compile only the direct DFT transform. NumPy's
    # vectorized reference/JS reduction is already faster and remains the oracle.
    fit_power = _power_numba_antigravity(fit_rows)
    query_power = _power_numba_antigravity(query_rows)
    return _score_from_power_numpy_antigravity(
        fit_power, query_power, backend="numba",
    )


def _score_local_stft_mlx_antigravity(
    fit_rows: FloatArrayAntigravity,
    query_rows: FloatArrayAntigravity,
) -> SpectralScoresAntigravity:
    del query_rows
    try:
        import mlx.core as mx
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("MLX is not installed") from error
    try:
        probe = mx.array(fit_rows[:1, 56:84], dtype=mx.float64)
        transformed = mx.fft.rfft(probe)
        mx.eval(transformed)
        transformed.tolist()
    except Exception as error:
        raise RuntimeError(
            f"MLX float64 RFFT cannot preserve the frozen oracle: {type(error).__name__}: {error}",
        ) from error
    raise RuntimeError("MLX float64 full scoring backend is unavailable")


def _split_combined_scores_antigravity(
    combined: SpectralScoresAntigravity,
    sizes: tuple[int, ...],
) -> tuple[SpectralScoresAntigravity, ...]:
    results = []
    start = 0
    for size in sizes:
        stop = start + size
        results.append(
            SpectralScoresAntigravity(
                fit_raw_js=combined.fit_raw_js,
                query_raw_js=combined.query_raw_js[start:stop],
                query_ranks=combined.query_ranks[start:stop],
                confirmation_mask=combined.confirmation_mask[start:stop],
                fit_normal_max_raw_js=combined.fit_normal_max_raw_js,
                backend=combined.backend,
            ),
        )
        start = stop
    return tuple(results)


def run_acceleration_workload_antigravity(
    workload: AccelerationWorkloadAntigravity,
    *,
    backend: str,
) -> tuple[tuple[SpectralScoresAntigravity, ...], ...]:
    if type(workload) is not AccelerationWorkloadAntigravity:
        raise TypeError("workload must be AccelerationWorkloadAntigravity")
    folds = []
    for fold in workload.folds:
        sizes = tuple(scope.shape[0] for scope in fold.query_scopes)
        combined_query = np.concatenate(fold.query_scopes, axis=0)
        combined = score_local_stft_antigravity(
            fold.fit_rows, combined_query, backend=backend,
        )
        folds.append(_split_combined_scores_antigravity(combined, sizes))
    return tuple(folds)


def _timing_summary_antigravity(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    q25, q75 = np.percentile(array, (25.0, 75.0))
    return median, float(q75 - q25)


def _parity_antigravity(
    expected: tuple[tuple[SpectralScoresAntigravity, ...], ...],
    actual: tuple[tuple[SpectralScoresAntigravity, ...], ...],
) -> tuple[bool, bool, float, bool, bool]:
    finite = True
    max_error = 0.0
    rank_exact = True
    confirmation_exact = True
    if len(expected) != len(actual):
        return False, False, float("inf"), False, False
    for expected_fold, actual_fold in zip(expected, actual, strict=True):
        if len(expected_fold) != len(actual_fold):
            return False, False, float("inf"), False, False
        for oracle, candidate in zip(expected_fold, actual_fold, strict=True):
            finite = finite and bool(
                np.isfinite(candidate.fit_raw_js).all()
                and np.isfinite(candidate.query_raw_js).all()
            )
            max_error = max(
                max_error,
                float(np.max(np.abs(oracle.fit_raw_js - candidate.fit_raw_js))),
                float(np.max(np.abs(oracle.query_raw_js - candidate.query_raw_js))),
            )
            rank_exact = rank_exact and bool(
                np.array_equal(oracle.query_ranks, candidate.query_ranks),
            )
            confirmation_exact = confirmation_exact and bool(
                np.array_equal(oracle.confirmation_mask, candidate.confirmation_mask),
            )
    passed = finite and max_error <= 1e-12 and rank_exact and confirmation_exact
    return passed, finite, max_error, rank_exact, confirmation_exact


def _exact_outputs_antigravity(
    left: tuple[tuple[SpectralScoresAntigravity, ...], ...],
    right: tuple[tuple[SpectralScoresAntigravity, ...], ...],
) -> bool:
    if len(left) != len(right):
        return False
    for left_fold, right_fold in zip(left, right, strict=True):
        if len(left_fold) != len(right_fold):
            return False
        for a, b in zip(left_fold, right_fold, strict=True):
            if not (
                np.array_equal(a.fit_raw_js, b.fit_raw_js)
                and np.array_equal(a.query_raw_js, b.query_raw_js)
                and np.array_equal(a.query_ranks, b.query_ranks)
                and np.array_equal(a.confirmation_mask, b.confirmation_mask)
            ):
                return False
    return True


def _counts_antigravity(workload: AccelerationWorkloadAntigravity) -> dict[str, int]:
    fit_raw = sum(fold.fit_rows.shape[0] for fold in workload.folds)
    query_raw = sum(
        scope.shape[0] for fold in workload.folds for scope in fold.query_scopes
    )
    transformed = fit_raw + query_raw
    return {
        "fit_raw": fit_raw,
        "query_raw": query_raw,
        "transformed_rows": transformed,
        "frames": transformed * 5,
    }


def assess_acceleration_antigravity(
    workload: AccelerationWorkloadAntigravity,
) -> AccelerationAssessmentAntigravity:
    if type(workload) is not AccelerationWorkloadAntigravity:
        raise TypeError("workload must be AccelerationWorkloadAntigravity")

    numpy_times: list[float] = []
    numpy_reference = None
    numpy_deterministic = True
    for _ in range(workload.numpy_repetitions):
        started = perf_counter()
        result = run_acceleration_workload_antigravity(workload, backend="numpy")
        numpy_times.append(perf_counter() - started)
        if numpy_reference is None:
            numpy_reference = result
        else:
            numpy_deterministic = numpy_deterministic and _exact_outputs_antigravity(
                numpy_reference, result,
            )
    assert numpy_reference is not None
    numpy_median, numpy_iqr = _timing_summary_antigravity(numpy_times)
    numpy_backend = BackendAssessmentAntigravity(
        backend="numpy",
        available=True,
        parity_passed=True,
        finite=True,
        max_raw_abs_error=0.0,
        rank_exact=True,
        confirmation_exact=True,
        deterministic=numpy_deterministic,
        cold_total_seconds=None,
        cold_compile_seconds=None,
        warm_median_seconds=numpy_median,
        warm_iqr_seconds=numpy_iqr,
        gate_seconds=numpy_median,
        speedup=1.0,
        eligible=False,
        fallback_reason=None,
    )

    if _numba_antigravity is None:
        numba_backend = BackendAssessmentAntigravity(
            "numba", False, False, False, None, False, False, False,
            None, None, None, None, None, None, False, "Numba is not installed",
        )
    else:
        try:
            started = perf_counter()
            numba_first = run_acceleration_workload_antigravity(workload, backend="numba")
            cold_total = perf_counter() - started
            warm_times: list[float] = []
            deterministic = True
            for _ in range(workload.warm_repetitions):
                started = perf_counter()
                candidate = run_acceleration_workload_antigravity(workload, backend="numba")
                warm_times.append(perf_counter() - started)
                deterministic = deterministic and _exact_outputs_antigravity(
                    numba_first, candidate,
                )
            warm_median, warm_iqr = _timing_summary_antigravity(warm_times)
            cold_compile = max(0.0, cold_total - warm_median)
            gate_seconds = cold_compile + warm_median
            parity, finite, error, rank_exact, confirmation_exact = _parity_antigravity(
                numpy_reference, numba_first,
            )
            speedup = numpy_median / gate_seconds
            eligible = (
                parity and deterministic and speedup >= REQUIRED_SPEEDUP_ANTIGRAVITY
            )
            reason = None
            if not parity:
                reason = "Numba output failed finite/raw/rank/confirmation parity"
            elif not deterministic:
                reason = "Numba repeated output was not deterministic"
            elif speedup < REQUIRED_SPEEDUP_ANTIGRAVITY:
                reason = "Numba cold compile plus one warm execution missed the 1.1x gate"
            numba_backend = BackendAssessmentAntigravity(
                "numba", True, parity, finite, error, rank_exact, confirmation_exact,
                deterministic, cold_total, cold_compile, warm_median, warm_iqr,
                gate_seconds, speedup, eligible, reason,
            )
        except Exception as error:  # pragma: no cover - defensive environment record
            numba_backend = BackendAssessmentAntigravity(
                "numba", True, False, False, None, False, False, False,
                None, None, None, None, None, None, False,
                f"{type(error).__name__}: {error}",
            )

    try:
        started = perf_counter()
        mlx_result = run_acceleration_workload_antigravity(workload, backend="mlx")
        mlx_cold = perf_counter() - started
        parity, finite, error, rank_exact, confirmation_exact = _parity_antigravity(
            numpy_reference, mlx_result,
        )
        speedup = numpy_median / mlx_cold
        eligible = parity and speedup >= REQUIRED_SPEEDUP_ANTIGRAVITY
        mlx_backend = BackendAssessmentAntigravity(
            "mlx", True, parity, finite, error, rank_exact, confirmation_exact,
            True, mlx_cold, 0.0, mlx_cold, 0.0, mlx_cold, speedup, eligible,
            None if eligible else "MLX missed parity or the 1.1x gate",
        )
    except ImportError as error:  # pragma: no cover - environment dependent
        mlx_backend = BackendAssessmentAntigravity(
            "mlx", False, False, False, None, False, False, False,
            None, None, None, None, None, None, False, str(error),
        )
    except Exception as error:
        mlx_backend = BackendAssessmentAntigravity(
            "mlx", True, False, False, None, False, False, False,
            None, None, None, None, None, None, False,
            f"{type(error).__name__}: {error}",
        )

    eligible = [backend for backend in (numba_backend, mlx_backend) if backend.eligible]
    if eligible:
        selected = min(
            eligible,
            key=lambda backend: float(backend.gate_seconds),
        ).backend
    else:
        selected = "numpy"
    return AccelerationAssessmentAntigravity(
        required_speedup=REQUIRED_SPEEDUP_ANTIGRAVITY,
        workload_counts=_counts_antigravity(workload),
        backends={
            "numpy": numpy_backend,
            "numba": numba_backend,
            "mlx": mlx_backend,
        },
        selected_backend=selected,
        gate_passed=selected != "numpy",
    )


__all__ = (
    "AccelerationAssessmentAntigravity",
    "AccelerationFoldAntigravity",
    "AccelerationWorkloadAntigravity",
    "BackendAssessmentAntigravity",
    "assess_acceleration_antigravity",
    "run_acceleration_workload_antigravity",
)
