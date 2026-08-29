"""Small four-row Python math oracle for Exp470 E11 scorer validation only."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


_ONSET_ANTIGRAVITY = 56
_WINDOW_LENGTH_ANTIGRAVITY = 28
_WINDOW_STARTS_ANTIGRAVITY = (0, 14, 28, 42, 56)
_DFT_BINS_ANTIGRAVITY = 14
_POWER_EPSILON_ANTIGRAVITY = 1e-12


@dataclass(frozen=True, slots=True)
class PythonReferenceScoresAntigravity:
    fit_raw_entropy: tuple[float, ...]
    query_raw_entropy: tuple[float, ...]
    query_ranks: tuple[float, ...]
    confirmation_mask: tuple[bool, ...]
    fit_normal_max_raw_entropy: float


def _entropy_row_python_reference_antigravity(row: tuple[float, ...]) -> float:
    entropies: list[float] = []
    for start in _WINDOW_STARTS_ANTIGRAVITY:
        base = _ONSET_ANTIGRAVITY + start
        window = tuple(row[base + index] for index in range(_WINDOW_LENGTH_ANTIGRAVITY))
        mean = math.fsum(window) / _WINDOW_LENGTH_ANTIGRAVITY
        powers: list[float] = []
        for bin_index in range(1, _DFT_BINS_ANTIGRAVITY + 1):
            real = math.fsum(
                (window[index] - mean)
                * (0.5 - 0.5 * math.cos(2.0 * math.pi * index / (_WINDOW_LENGTH_ANTIGRAVITY - 1)))
                * math.cos(2.0 * math.pi * bin_index * index / _WINDOW_LENGTH_ANTIGRAVITY)
                for index in range(_WINDOW_LENGTH_ANTIGRAVITY)
            )
            imaginary = math.fsum(
                -(window[index] - mean)
                * (0.5 - 0.5 * math.cos(2.0 * math.pi * index / (_WINDOW_LENGTH_ANTIGRAVITY - 1)))
                * math.sin(2.0 * math.pi * bin_index * index / _WINDOW_LENGTH_ANTIGRAVITY)
                for index in range(_WINDOW_LENGTH_ANTIGRAVITY)
            )
            powers.append(real * real + imaginary * imaginary + _POWER_EPSILON_ANTIGRAVITY)
        power_total = math.fsum(powers)
        entropies.append(
            -math.fsum((power / power_total) * math.log(power / power_total) for power in powers)
            / math.log(_DFT_BINS_ANTIGRAVITY)
        )
    return float(statistics.median(entropies))


def score_spectral_entropy_python_reference_antigravity(
    fit_rows: tuple[tuple[float, ...], ...],
    query_rows: tuple[tuple[float, ...], ...],
) -> PythonReferenceScoresAntigravity:
    """Return the frozen E11 oracle values without importing production scoring."""
    fit_entropy = tuple(_entropy_row_python_reference_antigravity(row) for row in fit_rows)
    query_entropy = tuple(_entropy_row_python_reference_antigravity(row) for row in query_rows)
    fit_raw = tuple(
        abs(entropy - float(statistics.median(fit_entropy[:index] + fit_entropy[index + 1 :])))
        for index, entropy in enumerate(fit_entropy)
    )
    fit_median = float(statistics.median(fit_entropy))
    query_raw = tuple(abs(entropy - fit_median) for entropy in query_entropy)
    fit_maximum = max(fit_raw)
    ranks = tuple(
        (1.0 + sum(1 for fit_value in fit_raw if fit_value <= raw)) / (len(fit_raw) + 1.0)
        for raw in query_raw
    )
    confirmation = tuple(raw > 0.0 and raw > fit_maximum for raw in query_raw)
    return PythonReferenceScoresAntigravity(
        fit_raw_entropy=fit_raw,
        query_raw_entropy=query_raw,
        query_ranks=ranks,
        confirmation_mask=confirmation,
        fit_normal_max_raw_entropy=fit_maximum,
    )


__all__ = (
    "PythonReferenceScoresAntigravity",
    "score_spectral_entropy_python_reference_antigravity",
)
