"""Project-owned FFT decomposition and TRAIN-only depth selection for Exp439."""

from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from .aspe_antigravity import (
    AspeResultAntigravity,
    amplitude_sensitive_permutation_entropy_antigravity,
)
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import DecompositionEvidence


def _finite_vector_antigravity(values: object, *, minimum_length: int = 1) -> NDArray[np.float32]:
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature input is invalid") from exc
    if vector.ndim != 1 or vector.size < minimum_length or not np.isfinite(vector).all():
        raise ValueError("feature input must be a finite vector")
    return np.ascontiguousarray(vector)


def maximum_depth_antigravity(raw_length: int) -> int:
    if type(raw_length) is not int:
        raise ValueError("raw length is invalid")
    if raw_length < FROZEN_CONFIG.minimum_raw_length:
        raise ValueError("raw length is below the minimum raw length")
    return min(
        FROZEN_CONFIG.maximum_depth,
        max(1, math.floor(math.log2(raw_length)) - 3),
    )


def fft_split_antigravity(
    values: object,
    depth: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    original = _finite_vector_antigravity(
        values,
        minimum_length=FROZEN_CONFIG.minimum_raw_length,
    )
    maximum = maximum_depth_antigravity(int(original.size))
    if type(depth) is not int or depth < 1 or depth > maximum:
        raise ValueError("decomposition depth is invalid for raw length")
    original64 = original.astype(np.float64, copy=False)
    spectrum = np.fft.rfft(original64)
    nyquist_bin = original.size // 2
    cutoff_bin = max(1, math.floor(nyquist_bin / (2**depth)))
    low_spectrum = np.zeros_like(spectrum)
    low_spectrum[: cutoff_bin + 1] = spectrum[: cutoff_bin + 1]
    low64 = np.fft.irfft(low_spectrum, n=original.size)
    high64 = original64 - low64
    low = np.ascontiguousarray(low64, dtype=np.float32)
    high = np.ascontiguousarray(high64, dtype=np.float32)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("decomposition output is nonfinite")
    error = float(np.max(np.abs(original - (low + high))))
    if not math.isfinite(error) or error > 1e-5:
        raise ValueError("decomposition reconstruction tolerance failed")
    low.flags.writeable = False
    high.flags.writeable = False
    return low, high


def amplitude_complexity_vector_antigravity(
    original: object,
    low: object,
    high: object,
    *,
    aspe_compute: Callable[[object, int, int], AspeResultAntigravity] = amplitude_sensitive_permutation_entropy_antigravity,
) -> tuple[NDArray[np.float32], tuple[bool, ...]]:
    source = _finite_vector_antigravity(original, minimum_length=FROZEN_CONFIG.minimum_raw_length)
    low_vector = _finite_vector_antigravity(low, minimum_length=FROZEN_CONFIG.minimum_raw_length)
    high_vector = _finite_vector_antigravity(high, minimum_length=FROZEN_CONFIG.minimum_raw_length)
    if source.shape != low_vector.shape or source.shape != high_vector.shape:
        raise ValueError("decomposition components do not share exact length")
    values: list[float] = []
    flags: list[bool] = []
    for component in (source, low_vector, high_vector):
        for delay in FROZEN_CONFIG.delays:
            result = aspe_compute(
                component,
                FROZEN_CONFIG.embedding_dimension,
                delay,
            )
            values.append(result.value)
            flags.append(result.flat_or_zero_weight)
    vector = np.ascontiguousarray(values, dtype=np.float32)
    if vector.shape != (9,) or not np.isfinite(vector).all():
        raise ValueError("amplitude-complexity feature is invalid")
    vector.flags.writeable = False
    return vector, tuple(flags)


def decompose_at_depth_antigravity(
    values: object,
    depth: int,
    *,
    aspe_compute: Callable[[object, int, int], AspeResultAntigravity] = amplitude_sensitive_permutation_entropy_antigravity,
) -> DecompositionEvidence:
    original = _finite_vector_antigravity(values, minimum_length=FROZEN_CONFIG.minimum_raw_length)
    low, high = fft_split_antigravity(original, depth)
    complexity, flags = amplitude_complexity_vector_antigravity(
        original, low, high, aspe_compute=aspe_compute
    )
    return DecompositionEvidence(
        depth=depth,
        low=low,
        high=high,
        amplitude_complexity=complexity,
        flat_or_zero_weight=flags,
    )


def _depth_score_antigravity(
    values: NDArray[np.float32],
    depth: int,
    aspe_compute: Callable[[object, int, int], AspeResultAntigravity],
) -> float:
    _, high = fft_split_antigravity(values, depth)
    scores = tuple(
        aspe_compute(
            high,
            FROZEN_CONFIG.embedding_dimension,
            delay,
        ).value
        for delay in FROZEN_CONFIG.delays
    )
    result = float(np.median(np.asarray(scores, dtype=np.float64)))
    if not math.isfinite(result):
        raise ValueError("depth ASPE score is nonfinite")
    return result


def preferred_depth_from_scores_antigravity(scores: Iterable[float]) -> int:
    values = tuple(float(item) for item in scores)
    if not values or len(values) > FROZEN_CONFIG.maximum_depth or any(not math.isfinite(item) for item in values):
        raise ValueError("depth scores are invalid")
    for index in range(1, len(values)):
        if values[index] >= values[index - 1] - 1e-6:
            return index
    return len(values)


def preferred_depth_antigravity(
    values: object,
    *,
    aspe_compute: Callable[[object, int, int], AspeResultAntigravity] = amplitude_sensitive_permutation_entropy_antigravity,
) -> int:
    vector = _finite_vector_antigravity(values, minimum_length=FROZEN_CONFIG.minimum_raw_length)
    maximum = maximum_depth_antigravity(int(vector.size))
    return preferred_depth_from_scores_antigravity(
        _depth_score_antigravity(vector, depth, aspe_compute)
        for depth in range(1, maximum + 1)
    )


def lower_median_depth_antigravity(depths: Iterable[int]) -> int:
    values = tuple(depths)
    if not values or any(type(value) is not int or value < 1 or value > FROZEN_CONFIG.maximum_depth for value in values):
        raise ValueError("preferred depths are invalid")
    ordered = tuple(sorted(values))
    return ordered[(len(ordered) - 1) // 2]


def select_train_depth_antigravity(
    matrix: object,
    *,
    excluded_row_index: int | None = None,
    aspe_compute: Callable[[object, int, int], AspeResultAntigravity] = amplitude_sensitive_permutation_entropy_antigravity,
) -> int:
    try:
        train = np.asarray(matrix, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("TRAIN depth matrix is invalid") from exc
    if train.ndim != 2 or train.shape[0] == 0 or train.shape[1] < FROZEN_CONFIG.minimum_raw_length or not np.isfinite(train).all():
        raise ValueError("TRAIN depth matrix is invalid")
    if excluded_row_index is not None:
        if type(excluded_row_index) is not int or excluded_row_index < 0 or excluded_row_index >= train.shape[0]:
            raise ValueError("excluded TRAIN row index is invalid")
        selected = np.delete(train, excluded_row_index, axis=0)
    else:
        selected = train
    if selected.shape[0] == 0:
        raise ValueError("TRAIN depth selection has no reference rows")
    return lower_median_depth_antigravity(
        preferred_depth_antigravity(row, aspe_compute=aspe_compute) for row in selected
    )


__all__ = [
    "amplitude_complexity_vector_antigravity",
    "decompose_at_depth_antigravity",
    "fft_split_antigravity",
    "lower_median_depth_antigravity",
    "maximum_depth_antigravity",
    "preferred_depth_antigravity",
    "preferred_depth_from_scores_antigravity",
    "select_train_depth_antigravity",
]
