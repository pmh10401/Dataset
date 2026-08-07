"""Optional, lazily-created Numba ASPE adapter for Exp439."""

from __future__ import annotations

import math
from typing import Callable

from .aspe_antigravity import AspeResultAntigravity


class NumbaUnavailableAntigravity(RuntimeError):
    pass


class NumbaAspeEngineAntigravity:
    backend = "numba"

    def __init__(self, kernel: Callable[[object, int, int], object], numba_version: str) -> None:
        if not callable(kernel) or type(numba_version) is not str or not numba_version:
            raise ValueError("Numba engine provenance is invalid")
        self._kernel = kernel
        self.numba_version = numba_version

    def compute(self, values: object, embedding_dimension: int, delay: int) -> AspeResultAntigravity:
        result = self._kernel(values, embedding_dimension, delay)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Numba ASPE output is invalid")
        value, flag = result
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("Numba ASPE output must be finite")
        if type(flag) is not bool:
            raise ValueError("Numba ASPE audit flag is invalid")
        return AspeResultAntigravity(float(value), flag)


def create_numba_aspe_engine_antigravity() -> NumbaAspeEngineAntigravity:
    try:
        import numba
        import numpy as np
    except (ImportError, OSError) as exc:
        raise NumbaUnavailableAntigravity("numba framework unavailable") from exc

    @numba.njit(cache=False)
    def kernel(values, embedding_dimension, delay):
        vector = np.asarray(values, dtype=np.float64)
        if embedding_dimension != 3 or delay < 1 or vector.ndim != 1:
            raise ValueError("Numba ASPE supports the frozen dimension only")
        count = vector.size - 2 * delay
        if count < 1:
            raise ValueError("Numba ASPE input is too short")
        minimum = np.min(vector)
        peak = np.max(vector) - minimum
        if not np.isfinite(peak) or peak <= 1e-12:
            return 0.0, True
        normalized = (vector - minimum) / peak
        weights = np.zeros(27, dtype=np.float64)
        total = 0.0
        for start in range(count):
            a = normalized[start]
            b = normalized[start + delay]
            c = normalized[start + 2 * delay]
            mean = (a + b + c) / 3.0
            variance = ((a - mean) ** 2 + (b - mean) ** 2 + (c - mean) ** 2) / 2.0
            standard_deviation = math.sqrt(variance)
            weight = 0.0 if mean <= 1e-12 and standard_deviation <= 1e-12 else standard_deviation / mean
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError("Numba ASPE weight is invalid")
            indices = np.asarray([0, 1, 2], dtype=np.int64)
            local = np.asarray([a, b, c], dtype=np.float64)
            for right in range(1, 3):
                current = indices[right]
                left = right - 1
                while left >= 0 and local[current] < local[indices[left]]:
                    indices[left + 1] = indices[left]
                    left -= 1
                indices[left + 1] = current
            code = indices[0] * 9 + indices[1] * 3 + indices[2]
            weights[code] += weight
            total += weight
        if total <= 1e-12:
            return 0.0, True
        entropy = 0.0
        for weight in weights:
            if weight > 0.0:
                probability = weight / total
                entropy -= probability * math.log2(probability)
        return entropy / math.log2(6.0), False

    return NumbaAspeEngineAntigravity(kernel, str(numba.__version__))


__all__ = [
    "NumbaAspeEngineAntigravity",
    "NumbaUnavailableAntigravity",
    "create_numba_aspe_engine_antigravity",
]
