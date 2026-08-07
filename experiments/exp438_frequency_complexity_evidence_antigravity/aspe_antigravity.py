"""Amplitude-sensitive permutation entropy for Exp438.

Reference semantics:
  https://github.com/Brioal/ASPE
  commit f17cf8d796c4d25bc1b5ace0e9a7ef7be2944a89
  entropy/aspe.m::aspe, Apache-2.0

This is an independently written Python implementation of the published
algorithm, not a line-for-line MATLAB copy. Exp438 adds one documented safety
extension: flat input or total finite weight <= 1e-12 returns 0.0 and records
``flat_or_zero_weight=True`` instead of propagating NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


_FLAT_EPSILON_ANTIGRAVITY = 1e-12


@dataclass(frozen=True)
class AspeResultAntigravity:
    value: float
    flat_or_zero_weight: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value < 0.0 or self.value > 1.0 + 1e-12:
            raise ValueError("ASPE value is invalid")
        if type(self.flat_or_zero_weight) is not bool:
            raise ValueError("ASPE audit flag is invalid")


def _finite_vector_antigravity(values: object) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ASPE input is invalid") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("ASPE input must be a finite nonempty vector")
    return np.ascontiguousarray(vector)


def ordinal_pattern_antigravity(values: Sequence[float] | np.ndarray) -> tuple[int, ...]:
    vector = _finite_vector_antigravity(values)
    return tuple(int(index) for index in np.argsort(vector, kind="stable"))


def amplitude_sensitive_permutation_entropy_antigravity(
    values: object,
    embedding_dimension: int,
    delay: int,
) -> AspeResultAntigravity:
    if type(embedding_dimension) is not int or embedding_dimension < 2:
        raise ValueError("ASPE embedding dimension is invalid")
    if type(delay) is not int or delay < 1:
        raise ValueError("ASPE delay is invalid")
    vector = _finite_vector_antigravity(values)
    window_count = vector.size - (embedding_dimension - 1) * delay
    if window_count < 1:
        raise ValueError("ASPE input is too short for the embedding")
    minimum = float(np.min(vector))
    peak_to_peak = float(np.max(vector) - minimum)
    if not math.isfinite(peak_to_peak):
        raise ValueError("ASPE range is nonfinite")
    if peak_to_peak <= _FLAT_EPSILON_ANTIGRAVITY:
        return AspeResultAntigravity(0.0, True)
    normalized = (vector - minimum) / peak_to_peak
    weighted_patterns: dict[tuple[int, ...], float] = {}
    total_weight = 0.0
    offsets = np.arange(embedding_dimension, dtype=np.int64) * delay
    for start in range(window_count):
        window = normalized[start + offsets]
        mean = float(np.mean(window, dtype=np.float64))
        standard_deviation = float(np.std(window, ddof=1, dtype=np.float64))
        if mean <= _FLAT_EPSILON_ANTIGRAVITY and standard_deviation <= _FLAT_EPSILON_ANTIGRAVITY:
            weight = 0.0
        else:
            weight = standard_deviation / mean
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("ASPE window weight is nonfinite")
        pattern = tuple(int(index) for index in np.argsort(window, kind="stable"))
        weighted_patterns[pattern] = weighted_patterns.get(pattern, 0.0) + weight
        total_weight += weight
    if not math.isfinite(total_weight):
        raise ValueError("ASPE total weight is nonfinite")
    if total_weight <= _FLAT_EPSILON_ANTIGRAVITY:
        return AspeResultAntigravity(0.0, True)
    entropy = 0.0
    for weight in weighted_patterns.values():
        if weight <= 0.0:
            continue
        probability = weight / total_weight
        entropy -= probability * math.log2(probability)
    normalized_entropy = entropy / math.log2(math.factorial(embedding_dimension))
    if not math.isfinite(normalized_entropy):
        raise ValueError("ASPE entropy is nonfinite")
    if abs(normalized_entropy) <= _FLAT_EPSILON_ANTIGRAVITY:
        normalized_entropy = 0.0
    return AspeResultAntigravity(float(normalized_entropy), False)


__all__ = [
    "AspeResultAntigravity",
    "amplitude_sensitive_permutation_entropy_antigravity",
    "ordinal_pattern_antigravity",
]
