"""Candidate Numba ASPE kernels for the Exp471 preprocessing spike."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import math
import os
from types import MappingProxyType

import numba
import numpy as np


_FLAT_EPSILON_ANTIGRAVITY = 1e-12


def _aspe_kernel_python_antigravity(
    values: np.ndarray, embedding_dimension: int, delay: int
) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if embedding_dimension != 3 or delay < 1 or vector.ndim != 1:
        return np.nan
    count = vector.size - 2 * delay
    if count < 1:
        return np.nan
    minimum = np.min(vector)
    peak = np.max(vector) - minimum
    if not np.isfinite(peak):
        return np.nan
    if peak <= _FLAT_EPSILON_ANTIGRAVITY:
        return 0.0
    normalized = (vector - minimum) / peak
    pattern_codes = np.empty(count, dtype=np.int64)
    window_weights = np.empty(count, dtype=np.float64)
    for start in numba.prange(count):
        local = np.asarray(
            (
                normalized[start],
                normalized[start + delay],
                normalized[start + 2 * delay],
            ),
            dtype=np.float64,
        )
        mean = (local[0] + local[1] + local[2]) / 3.0
        variance = (
            (local[0] - mean) ** 2
            + (local[1] - mean) ** 2
            + (local[2] - mean) ** 2
        ) / 2.0
        standard_deviation = math.sqrt(variance)
        weight = (
            0.0
            if mean <= _FLAT_EPSILON_ANTIGRAVITY
            and standard_deviation <= _FLAT_EPSILON_ANTIGRAVITY
            else standard_deviation / mean
        )
        indices = np.asarray((0, 1, 2), dtype=np.int64)
        for right in range(1, 3):
            current = indices[right]
            left = right - 1
            while left >= 0 and local[current] < local[indices[left]]:
                indices[left + 1] = indices[left]
                left -= 1
            indices[left + 1] = current
        pattern_codes[start] = indices[0] * 9 + indices[1] * 3 + indices[2]
        window_weights[start] = weight

    # Reduce in original window order so parallel scheduling cannot change sums.
    weights = np.zeros(27, dtype=np.float64)
    seen = np.zeros(27, dtype=np.uint8)
    first_seen = np.empty(27, dtype=np.int64)
    first_seen_count = 0
    total = 0.0
    for index in range(count):
        code = pattern_codes[index]
        weight = window_weights[index]
        if not np.isfinite(weight) or weight < 0.0:
            return np.nan
        if seen[code] == 0:
            seen[code] = 1
            first_seen[first_seen_count] = code
            first_seen_count += 1
        weights[code] += weight
        total += weight
    if not np.isfinite(total):
        return np.nan
    if total <= _FLAT_EPSILON_ANTIGRAVITY:
        return 0.0
    entropy = 0.0
    for index in range(first_seen_count):
        weight = weights[first_seen[index]]
        if weight > 0.0:
            probability = weight / total
            entropy -= probability * math.log2(probability)
    result = entropy / math.log2(6.0)
    if not np.isfinite(result):
        return np.nan
    return 0.0 if abs(result) <= _FLAT_EPSILON_ANTIGRAVITY else result


_SERIAL_SAFE_ANTIGRAVITY = numba.njit(
    cache=True, fastmath=False, nogil=True, parallel=False
)(_aspe_kernel_python_antigravity)
_PARALLEL_SAFE_ANTIGRAVITY = numba.njit(
    cache=True, fastmath=False, nogil=True, parallel=True
)(_aspe_kernel_python_antigravity)
_PARALLEL_FAST_ANTIGRAVITY = numba.njit(
    cache=True, fastmath=True, nogil=True, parallel=True
)(_aspe_kernel_python_antigravity)


def _checked_variant_antigravity(
    kernel: Callable[[np.ndarray, int, int], float]
) -> Callable[[object, int, int], float]:
    def compute_antigravity(
        values: object, embedding_dimension: int, delay: int
    ) -> float:
        if (
            type(embedding_dimension) is not int
            or embedding_dimension != 3
            or type(delay) is not int
            or delay < 1
        ):
            raise ValueError("Exp471 Numba ASPE parameters differ")
        vector = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Exp471 Numba ASPE input differs")
        value = float(kernel(vector, embedding_dimension, delay))
        if not math.isfinite(value):
            raise ValueError("Exp471 Numba ASPE output differs")
        return value

    return compute_antigravity


def create_aspe_variants_antigravity() -> Mapping[
    str, Callable[[object, int, int], float]
]:
    """Create the three preregistered safe/aggressive candidate variants."""

    logical_cpu_count = os.cpu_count() or 1
    numba.set_num_threads(max(1, logical_cpu_count - 1))
    return MappingProxyType(
        {
            "serial_safe": _checked_variant_antigravity(
                _SERIAL_SAFE_ANTIGRAVITY
            ),
            "parallel_safe": _checked_variant_antigravity(
                _PARALLEL_SAFE_ANTIGRAVITY
            ),
            "parallel_fast": _checked_variant_antigravity(
                _PARALLEL_FAST_ANTIGRAVITY
            ),
        }
    )


@contextmanager
def instrument_numba_aspe_antigravity(
    score: Callable[..., object],
    variant: Callable[[object, int, int], float],
):
    """Temporarily replace only the captured ASPE scalar and restore it."""

    scoring_globals = getattr(score, "__globals__", None)
    original = (
        scoring_globals.get("_aspe_paper")
        if isinstance(scoring_globals, dict)
        else None
    )
    if not callable(score) or not callable(variant) or not callable(original):
        raise ValueError("Exp471 captured ASPE boundary differs")
    scoring_globals["_aspe_paper"] = variant
    try:
        yield
    finally:
        observed = scoring_globals.get("_aspe_paper")
        scoring_globals["_aspe_paper"] = original
        if observed is not variant:
            raise ValueError("Exp471 captured ASPE callable changed")


__all__ = [
    "create_aspe_variants_antigravity",
    "instrument_numba_aspe_antigravity",
]
