"""Runtime evidence for Exp470's fixed Numba spectral-entropy scorer.

This module only measures the four-row public smoke fixture and caller-provided
unlabelled arrays.  It neither loads ECG200 nor performs qualification.  The
selected backend is deliberately always Numba; MLX is observed only to satisfy
the project acceleration policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
from time import perf_counter
from types import MappingProxyType
from typing import Mapping, Sequence

import llvmlite
import numba
import numpy as np

from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.python_reference_antigravity import (
    score_spectral_entropy_python_reference_antigravity,
)
from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.spectral_entropy_antigravity import (
    EntropyScoresAntigravity,
    score_spectral_entropy_numba_antigravity,
)


_FIXTURE_SHA256_ANTIGRAVITY = "054bfadb84cd5e4fb8e123207a25c089c6e4cd45366895e74c5e39707068187c"
_MLX_BACKEND_ANTIGRAVITY = "mlx"
_NUMBA_BACKEND_ANTIGRAVITY = "numba"


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


@dataclass(frozen=True, slots=True)
class RuntimeAssessmentAntigravity:
    selected_backend: str
    required_mlx_speedup: float
    backends: Mapping[str, BackendAssessmentAntigravity]
    fixture_sha256: str
    python_version: str
    numba_version: str
    llvmlite_version: str
    mlx_version: str | None
    mlx_metal_version: str | None
    device: str
    cpu: str


@dataclass(frozen=True, slots=True)
class _MlxScoresAntigravity:
    fit_raw_entropy: tuple[float, ...]
    query_raw_entropy: tuple[float, ...]
    query_ranks: tuple[float, ...]
    confirmation_mask: tuple[bool, ...]
    fit_normal_max_raw_entropy: float


def fixed_reference_rows_antigravity() -> np.ndarray:
    """Return the frozen exact 4×140 fixture as an owned, readonly C array.

    NumPy is used here only for the explicitly permitted ndarray fixture
    container, never as a score backend.
    """
    x = np.arange(140, dtype=np.float64)
    rows = np.stack(
        (
            np.ones(140, dtype=np.float64),
            np.sin(2.0 * np.pi * x / 14.0),
            x / 139.0,
            np.sin(2.0 * np.pi * x / 7.0) + 0.25 * np.sin(2.0 * np.pi * x / 4.0),
        )
    )
    owned = np.ascontiguousarray(rows, dtype=np.float64).copy()
    owned.setflags(write=False)
    return owned


def score_bytes_antigravity(scores: EntropyScoresAntigravity) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    """Return canonical bytes for the five discrete/numeric scorer outputs."""
    return (
        scores.fit_raw_entropy.tobytes(order="C"),
        scores.query_raw_entropy.tobytes(order="C"),
        scores.query_ranks.tobytes(order="C"),
        scores.confirmation_mask.tobytes(order="C"),
        struct.pack("<d", scores.fit_normal_max_raw_entropy),
    )


def scores_finite_antigravity(scores: EntropyScoresAntigravity) -> bool:
    """Check every exported floating scorer field without a NumPy score path."""
    return (
        all(math.isfinite(float(value)) for value in scores.fit_raw_entropy)
        and all(math.isfinite(float(value)) for value in scores.query_raw_entropy)
        and all(math.isfinite(float(value)) for value in scores.query_ranks)
        and math.isfinite(scores.fit_normal_max_raw_entropy)
    )


def timing_summary_antigravity(values: Sequence[float]) -> tuple[float, float]:
    """Return median and nearest-rank IQR of finite, positive timings.

    Nearest-rank quartiles avoid platform-specific interpolation choices in the
    tiny fixed smoke timing sample.
    """
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timings must be positive finite values")
    ordered = tuple(sorted(float(value) for value in values))
    count = len(ordered)
    lower = ordered[max(0, math.ceil(count * 0.25) - 1)]
    upper = ordered[max(0, math.ceil(count * 0.75) - 1)]
    return float(statistics.median(ordered)), float(upper - lower)


def _fixture_digest_antigravity(rows: np.ndarray) -> str:
    if rows.dtype != np.float64 or rows.shape != (4, 140) or not rows.flags.c_contiguous:
        raise RuntimeError("fixed fixture lost canonical float64 C-order representation")
    digest = hashlib.sha256(rows.tobytes(order="C")).hexdigest()
    if digest != _FIXTURE_SHA256_ANTIGRAVITY:
        raise RuntimeError("fixed fixture digest mismatch")
    return digest


def _python_rows_antigravity(rows: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in rows)


def _max_raw_error_antigravity(
    observed: EntropyScoresAntigravity,
    expected: object,
) -> float:
    errors = [
        *(abs(float(actual) - reference) for actual, reference in zip(observed.fit_raw_entropy, expected.fit_raw_entropy)),
        *(abs(float(actual) - reference) for actual, reference in zip(observed.query_raw_entropy, expected.query_raw_entropy)),
        abs(observed.fit_normal_max_raw_entropy - expected.fit_normal_max_raw_entropy),
    ]
    return float(max(errors))


def _exact_discrete_outputs_antigravity(observed: EntropyScoresAntigravity, expected: object) -> bool:
    return (
        tuple(float(value) for value in observed.query_ranks) == expected.query_ranks
        and tuple(bool(value) for value in observed.confirmation_mask) == expected.confirmation_mask
    )


def _require_runtime_arguments_antigravity(
    repetitions: int,
    reference_tolerance: float,
    required_mlx_speedup: float,
) -> None:
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if not math.isfinite(reference_tolerance):
        raise ValueError("reference_tolerance must be finite")
    if not math.isfinite(required_mlx_speedup) or required_mlx_speedup <= 0.0:
        raise ValueError("required_mlx_speedup must be finite and positive")


def _median_mlx_antigravity(mx: object, values: object, axis: int) -> object:
    ordered = mx.sort(values, axis=axis)
    length = values.shape[axis]
    if length % 2:
        return mx.take(ordered, length // 2, axis=axis)
    return (mx.take(ordered, length // 2 - 1, axis=axis) + mx.take(ordered, length // 2, axis=axis)) / 2.0


def _score_mlx_antigravity(mx: object, fit_rows: np.ndarray, query_rows: np.ndarray) -> _MlxScoresAntigravity:
    """Compute the frozen formula through MLX only for the policy smoke."""
    fit = mx.array(fit_rows, dtype=mx.float64)
    query = mx.array(query_rows, dtype=mx.float64)
    positions = mx.arange(28, dtype=mx.float64)
    bins = mx.arange(1, 15, dtype=mx.float64)
    window = 0.5 - 0.5 * mx.cos(2.0 * math.pi * positions / 27.0)
    phase = 2.0 * math.pi * bins[:, None] * positions[None, :] / 28.0
    cosine = mx.cos(phase)
    sine = mx.sin(phase)

    def entropies(rows: object) -> object:
        per_window = []
        for start in (0, 14, 28, 42, 56):
            segment = rows[:, 56 + start : 56 + start + 28]
            centered = segment - mx.mean(segment, axis=1, keepdims=True)
            weighted = centered * window[None, :]
            real = mx.sum(weighted[:, None, :] * cosine[None, :, :], axis=2)
            imaginary = mx.sum(-weighted[:, None, :] * sine[None, :, :], axis=2)
            powers = real * real + imaginary * imaginary + 1e-12
            probability = powers / mx.sum(powers, axis=1, keepdims=True)
            per_window.append(-mx.sum(probability * mx.log(probability), axis=1) / math.log(14.0))
        return _median_mlx_antigravity(mx, mx.stack(per_window, axis=1), axis=1)

    fit_entropy = entropies(fit)
    query_entropy = entropies(query)
    loo_raw = []
    for index in range(fit.shape[0]):
        remainder = mx.concatenate((fit_entropy[:index], fit_entropy[index + 1 :]))
        loo_raw.append(mx.abs(fit_entropy[index] - _median_mlx_antigravity(mx, remainder, axis=0)))
    fit_raw = mx.stack(loo_raw)
    fit_median = _median_mlx_antigravity(mx, fit_entropy, axis=0)
    query_raw = mx.abs(query_entropy - fit_median)
    ranks = mx.stack(
        tuple((1.0 + mx.sum(fit_raw <= raw)) / (fit_raw.shape[0] + 1.0) for raw in query_raw)
    )
    confirmation = (query_raw > 0.0) & (query_raw > mx.max(fit_raw))
    mx.eval(fit_raw, query_raw, ranks, confirmation)
    return _MlxScoresAntigravity(
        fit_raw_entropy=tuple(float(value) for value in fit_raw.tolist()),
        query_raw_entropy=tuple(float(value) for value in query_raw.tolist()),
        query_ranks=tuple(float(value) for value in ranks.tolist()),
        confirmation_mask=tuple(bool(value) for value in confirmation.tolist()),
        fit_normal_max_raw_entropy=float(mx.max(fit_raw).item()),
    )


def _mlx_error_antigravity(observed: _MlxScoresAntigravity, expected: EntropyScoresAntigravity) -> float:
    return float(
        max(
            *(abs(actual - float(reference)) for actual, reference in zip(observed.fit_raw_entropy, expected.fit_raw_entropy)),
            *(abs(actual - float(reference)) for actual, reference in zip(observed.query_raw_entropy, expected.query_raw_entropy)),
            abs(observed.fit_normal_max_raw_entropy - expected.fit_normal_max_raw_entropy),
        )
    )


def assess_mlx_smoke_antigravity(
    fit_rows: np.ndarray,
    query_rows: np.ndarray,
    numba_scores: EntropyScoresAntigravity,
    numba_warm_median: float,
    required_speedup: float,
    repetitions: int = 21,
    reference_tolerance: float = 1e-12,
) -> BackendAssessmentAntigravity:
    """Observe MLX parity/speed, retaining a detailed unavailable/failure record."""
    try:
        import mlx.core as mx
    except Exception as error:  # capability is an expected environment outcome
        return BackendAssessmentAntigravity(
            backend=_MLX_BACKEND_ANTIGRAVITY, available=False, finite=False,
            python_reference_passed=None, parity_passed=False, exact_candidate_indices=False,
            max_raw_abs_error=None, deterministic=False, cold_seconds=None,
            warm_median_seconds=None, warm_iqr_seconds=None, speedup=None, eligible=False,
            fallback_reason=f"MLX unavailable: {type(error).__name__}: {error}",
        )

    try:
        started = perf_counter()
        first = _score_mlx_antigravity(mx, fit_rows, query_rows)
        cold_seconds = perf_counter() - started
        started = perf_counter()
        second = _score_mlx_antigravity(mx, fit_rows, query_rows)
        second_seconds = perf_counter() - started
        timings = [second_seconds]
        for _ in range(repetitions - 1):
            started = perf_counter()
            _score_mlx_antigravity(mx, fit_rows, query_rows)
            timings.append(perf_counter() - started)
        warm_median, warm_iqr = timing_summary_antigravity(timings)
        raw_error = _mlx_error_antigravity(first, numba_scores)
        exact_discrete = (
            first.query_ranks == tuple(float(value) for value in numba_scores.query_ranks)
            and first.confirmation_mask == tuple(bool(value) for value in numba_scores.confirmation_mask)
        )
        finite = all(
            math.isfinite(value)
            for values in (first.fit_raw_entropy, first.query_raw_entropy, first.query_ranks)
            for value in values
        ) and math.isfinite(first.fit_normal_max_raw_entropy)
        deterministic = first == second
        parity = finite and raw_error <= reference_tolerance
        speedup = numba_warm_median / warm_median
        eligible = parity and exact_discrete and deterministic and speedup >= required_speedup
        failures = []
        if not finite:
            failures.append("nonfinite MLX score")
        if not parity:
            failures.append(f"raw parity error {raw_error:.17g} exceeds {reference_tolerance:.17g}")
        if not exact_discrete:
            failures.append("candidate ranks or confirmations differ")
        if not deterministic:
            failures.append("score outputs are not deterministic")
        if speedup < required_speedup:
            failures.append(f"speedup {speedup:.6g} below required {required_speedup:.6g}")
        return BackendAssessmentAntigravity(
            backend=_MLX_BACKEND_ANTIGRAVITY, available=True, finite=finite,
            python_reference_passed=None, parity_passed=parity,
            exact_candidate_indices=exact_discrete, max_raw_abs_error=raw_error,
            deterministic=deterministic, cold_seconds=cold_seconds,
            warm_median_seconds=warm_median, warm_iqr_seconds=warm_iqr,
            speedup=speedup, eligible=eligible,
            fallback_reason=None if eligible else "; ".join(failures),
        )
    except Exception as error:
        return BackendAssessmentAntigravity(
            backend=_MLX_BACKEND_ANTIGRAVITY, available=True, finite=False,
            python_reference_passed=None, parity_passed=False, exact_candidate_indices=False,
            max_raw_abs_error=None, deterministic=False, cold_seconds=None,
            warm_median_seconds=None, warm_iqr_seconds=None, speedup=None, eligible=False,
            fallback_reason=f"MLX smoke failed: {type(error).__name__}: {error}",
        )


def assess_runtime_antigravity(
    fit_rows: object,
    query_rows: object,
    repetitions: int = 21,
    reference_tolerance: float = 1e-12,
    required_mlx_speedup: float = 1.1,
) -> RuntimeAssessmentAntigravity:
    """Measure Numba and observe MLX; production selection remains Numba."""
    _require_runtime_arguments_antigravity(repetitions, reference_tolerance, required_mlx_speedup)
    started = perf_counter()
    first = score_spectral_entropy_numba_antigravity(fit_rows, query_rows)
    cold_seconds = perf_counter() - started
    if not scores_finite_antigravity(first):
        raise RuntimeError("Numba finite check failed")
    second = score_spectral_entropy_numba_antigravity(fit_rows, query_rows)
    if score_bytes_antigravity(first) != score_bytes_antigravity(second):
        raise RuntimeError("Numba determinism check failed")

    fixed_rows = fixed_reference_rows_antigravity()
    fixture_digest = _fixture_digest_antigravity(fixed_rows)
    expected = score_spectral_entropy_python_reference_antigravity(
        _python_rows_antigravity(fixed_rows[:3]), _python_rows_antigravity(fixed_rows[3:])
    )
    observed = score_spectral_entropy_numba_antigravity(fixed_rows[:3], fixed_rows[3:])
    raw_error = _max_raw_error_antigravity(observed, expected)
    exact_discrete = _exact_discrete_outputs_antigravity(observed, expected)
    if not scores_finite_antigravity(observed) or raw_error > reference_tolerance or not exact_discrete:
        raise RuntimeError("Python reference check failed")

    timings = []
    for _ in range(repetitions):
        started = perf_counter()
        score_spectral_entropy_numba_antigravity(fit_rows, query_rows)
        timings.append(perf_counter() - started)
    warm_median, warm_iqr = timing_summary_antigravity(timings)
    numba_assessment = BackendAssessmentAntigravity(
        backend=_NUMBA_BACKEND_ANTIGRAVITY, available=True, finite=True,
        python_reference_passed=True, parity_passed=True, exact_candidate_indices=True,
        max_raw_abs_error=raw_error, deterministic=True, cold_seconds=cold_seconds,
        warm_median_seconds=warm_median, warm_iqr_seconds=warm_iqr, speedup=1.0,
        eligible=True, fallback_reason=None,
    )
    mlx_assessment = assess_mlx_smoke_antigravity(
        fit_rows, query_rows, first, warm_median, required_mlx_speedup, repetitions,
        reference_tolerance,
    )
    mlx_version = None
    mlx_metal_version = None
    device = "MLX unavailable"
    if mlx_assessment.available:
        try:
            import mlx.core as mx

            mlx_version = getattr(mx, "__version__", None)
            mlx_metal_version = getattr(getattr(mx, "metal", None), "__version__", None)
            device = str(mx.default_device())
        except Exception:
            device = "MLX metadata unavailable"
    return RuntimeAssessmentAntigravity(
        selected_backend=_NUMBA_BACKEND_ANTIGRAVITY,
        required_mlx_speedup=required_mlx_speedup,
        backends=MappingProxyType({_NUMBA_BACKEND_ANTIGRAVITY: numba_assessment, _MLX_BACKEND_ANTIGRAVITY: mlx_assessment}),
        fixture_sha256=fixture_digest,
        python_version=platform.python_version(),
        numba_version=numba.__version__,
        llvmlite_version=llvmlite.__version__,
        mlx_version=mlx_version,
        mlx_metal_version=mlx_metal_version,
        device=device,
        cpu=platform.processor() or platform.machine(),
    )


def _manifest_payload_antigravity(assessment: RuntimeAssessmentAntigravity) -> dict[str, object]:
    return {
        "selected_backend": assessment.selected_backend,
        "required_mlx_speedup": assessment.required_mlx_speedup,
        "backends": {name: asdict(result) for name, result in assessment.backends.items()},
        "fixture_sha256": assessment.fixture_sha256,
        "python_version": assessment.python_version,
        "numba_version": assessment.numba_version,
        "llvmlite_version": assessment.llvmlite_version,
        "mlx_version": assessment.mlx_version,
        "mlx_metal_version": assessment.mlx_metal_version,
        "device": assessment.device,
        "cpu": assessment.cpu,
    }


def _validate_json_values_antigravity(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("assessment JSON cannot contain nonfinite values")
    if isinstance(value, dict):
        for nested in value.values():
            _validate_json_values_antigravity(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json_values_antigravity(nested)


def main_antigravity(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create Exp470 Numba/MLX runtime evidence.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=21)
    parsed = parser.parse_args(arguments)
    rows = fixed_reference_rows_antigravity()
    assessment = assess_runtime_antigravity(rows[:3], rows[3:], repetitions=parsed.repetitions)
    payload = _manifest_payload_antigravity(assessment)
    _validate_json_values_antigravity(payload)
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())


__all__ = (
    "BackendAssessmentAntigravity",
    "RuntimeAssessmentAntigravity",
    "assess_mlx_smoke_antigravity",
    "assess_runtime_antigravity",
    "fixed_reference_rows_antigravity",
    "main_antigravity",
    "score_bytes_antigravity",
    "scores_finite_antigravity",
    "timing_summary_antigravity",
)
