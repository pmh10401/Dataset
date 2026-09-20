"""Compiled direct-DFT spectral-entropy scorer for Exp470 E11."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numba
import numpy as np

from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
)


_ONSET_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.onset
_WINDOW_LENGTH_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.window_length
_WINDOW_STARTS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.window_starts
_DFT_BINS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.dft_bins
_POWER_EPSILON_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.power_epsilon
_WIDTH_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.interpolated_width


@dataclass(frozen=True, slots=True)
class EntropyScoresAntigravity:
    fit_raw_entropy: np.ndarray
    query_raw_entropy: np.ndarray
    query_ranks: np.ndarray
    confirmation_mask: np.ndarray
    fit_normal_max_raw_entropy: float
    backend: str


@numba.njit(cache=False, fastmath=False)
def _median_copy_numba_antigravity(values: np.ndarray) -> float:
    ordered = values.copy()
    length = ordered.shape[0]
    for index in range(1, length):
        current = ordered[index]
        cursor = index - 1
        while cursor >= 0 and ordered[cursor] > current:
            ordered[cursor + 1] = ordered[cursor]
            cursor -= 1
        ordered[cursor + 1] = current
    midpoint = length // 2
    if length % 2 == 1:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


@numba.njit(cache=False, fastmath=False)
def entropy_rows_numba_antigravity(rows: np.ndarray) -> np.ndarray:
    """Calculate the median five-window normalized entropy for every input row."""
    row_count = rows.shape[0]
    output = np.empty(row_count, dtype=np.float64)
    for row_index in range(row_count):
        window_entropies = np.empty(len(_WINDOW_STARTS_ANTIGRAVITY), dtype=np.float64)
        for window_index in range(len(_WINDOW_STARTS_ANTIGRAVITY)):
            base = _ONSET_ANTIGRAVITY + _WINDOW_STARTS_ANTIGRAVITY[window_index]
            mean = 0.0
            for sample_index in range(_WINDOW_LENGTH_ANTIGRAVITY):
                mean += rows[row_index, base + sample_index]
            mean /= _WINDOW_LENGTH_ANTIGRAVITY
            powers = np.empty(_DFT_BINS_ANTIGRAVITY, dtype=np.float64)
            total_power = 0.0
            for bin_index in range(1, _DFT_BINS_ANTIGRAVITY + 1):
                real = 0.0
                imaginary = 0.0
                for sample_index in range(_WINDOW_LENGTH_ANTIGRAVITY):
                    hann = 0.5 - 0.5 * math.cos(
                        2.0 * math.pi * sample_index / (_WINDOW_LENGTH_ANTIGRAVITY - 1)
                    )
                    sample = (rows[row_index, base + sample_index] - mean) * hann
                    angle = 2.0 * math.pi * bin_index * sample_index / _WINDOW_LENGTH_ANTIGRAVITY
                    real += sample * math.cos(angle)
                    imaginary -= sample * math.sin(angle)
                power = real * real + imaginary * imaginary + _POWER_EPSILON_ANTIGRAVITY
                powers[bin_index - 1] = power
                total_power += power
            entropy = 0.0
            for bin_index in range(_DFT_BINS_ANTIGRAVITY):
                probability = powers[bin_index] / total_power
                entropy -= probability * math.log(probability)
            window_entropies[window_index] = entropy / math.log(_DFT_BINS_ANTIGRAVITY)
        output[row_index] = _median_copy_numba_antigravity(window_entropies)
    return output


@numba.njit(cache=False, fastmath=False)
def strict_confirmation_numba_antigravity(
    query_raw_entropy: np.ndarray, fit_normal_max_raw_entropy: float
) -> np.ndarray:
    """Return strict, tie-rejecting confirmation decisions for query raw scores."""
    confirmation = np.empty(query_raw_entropy.shape[0], dtype=np.bool_)
    for index in range(query_raw_entropy.shape[0]):
        raw = query_raw_entropy[index]
        confirmation[index] = raw > 0.0 and raw > fit_normal_max_raw_entropy
    return confirmation


@numba.njit(cache=False, fastmath=False)
def assemble_raw_rank_numba_antigravity(
    fit_entropy: np.ndarray, query_entropy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Assemble LOO raw scores, query raw scores, deterministic ranks, and fit maximum."""
    fit_count = fit_entropy.shape[0]
    fit_raw = np.empty(fit_count, dtype=np.float64)
    for fit_index in range(fit_count):
        without_current = np.empty(fit_count - 1, dtype=np.float64)
        write_index = 0
        for candidate_index in range(fit_count):
            if candidate_index != fit_index:
                without_current[write_index] = fit_entropy[candidate_index]
                write_index += 1
        fit_raw[fit_index] = abs(fit_entropy[fit_index] - _median_copy_numba_antigravity(without_current))
    fit_median = _median_copy_numba_antigravity(fit_entropy)
    fit_maximum = fit_raw[0]
    for fit_index in range(1, fit_count):
        if fit_raw[fit_index] > fit_maximum:
            fit_maximum = fit_raw[fit_index]
    query_count = query_entropy.shape[0]
    query_raw = np.empty(query_count, dtype=np.float64)
    query_ranks = np.empty(query_count, dtype=np.float64)
    for query_index in range(query_count):
        raw = abs(query_entropy[query_index] - fit_median)
        query_raw[query_index] = raw
        less_than_or_equal_count = 0
        for fit_index in range(fit_count):
            if fit_raw[fit_index] <= raw:
                less_than_or_equal_count += 1
        query_ranks[query_index] = (1.0 + less_than_or_equal_count) / (fit_count + 1.0)
    return fit_raw, query_raw, query_ranks, fit_maximum


def _validated_rows_antigravity(rows: object, name: str, minimum_rows: int) -> np.ndarray:
    if type(rows) is not np.ndarray:
        raise TypeError(f"{name} must be an exact numpy.ndarray")
    if rows.dtype != np.float64:
        raise TypeError(f"{name} must have exact dtype float64")
    if rows.ndim != 2:
        raise ValueError(f"{name} must have rank 2")
    if rows.shape[1] != _WIDTH_ANTIGRAVITY:
        raise ValueError(f"{name} must have width {_WIDTH_ANTIGRAVITY}")
    if rows.shape[0] < minimum_rows:
        raise ValueError(f"{name} must contain at least {minimum_rows} rows")
    if not np.isfinite(rows).all():
        raise ValueError(f"{name} must contain only finite values")
    return rows


def _freeze_array_antigravity(values: np.ndarray) -> np.ndarray:
    owned = values.copy()
    owned.setflags(write=False)
    return owned


def _require_finite_outputs_antigravity(
    fit_entropy: np.ndarray,
    query_entropy: np.ndarray,
    fit_raw: np.ndarray,
    query_raw: np.ndarray,
    query_ranks: np.ndarray,
    fit_maximum: float,
) -> None:
    if not (
        np.isfinite(fit_entropy).all()
        and np.isfinite(query_entropy).all()
        and np.isfinite(fit_raw).all()
        and np.isfinite(query_raw).all()
        and np.isfinite(query_ranks).all()
        and math.isfinite(fit_maximum)
    ):
        raise ValueError("non-finite numerical output from compiled spectral entropy scorer")


def score_spectral_entropy_numba_antigravity(
    fit_rows: object, query_rows: object
) -> EntropyScoresAntigravity:
    """Validate arrays then execute all E11 numerical scoring inside compiled kernels."""
    validated_fit = _validated_rows_antigravity(fit_rows, "fit_rows", 2)
    validated_query = _validated_rows_antigravity(query_rows, "query_rows", 1)
    fit_entropy = entropy_rows_numba_antigravity(validated_fit)
    query_entropy = entropy_rows_numba_antigravity(validated_query)
    fit_raw, query_raw, query_ranks, fit_maximum = assemble_raw_rank_numba_antigravity(
        fit_entropy, query_entropy
    )
    _require_finite_outputs_antigravity(
        fit_entropy,
        query_entropy,
        fit_raw,
        query_raw,
        query_ranks,
        fit_maximum,
    )
    confirmation = strict_confirmation_numba_antigravity(query_raw, fit_maximum)
    return EntropyScoresAntigravity(
        fit_raw_entropy=_freeze_array_antigravity(fit_raw),
        query_raw_entropy=_freeze_array_antigravity(query_raw),
        query_ranks=_freeze_array_antigravity(query_ranks),
        confirmation_mask=_freeze_array_antigravity(confirmation),
        fit_normal_max_raw_entropy=float(fit_maximum),
        backend="numba",
    )


__all__ = (
    "EntropyScoresAntigravity",
    "assemble_raw_rank_numba_antigravity",
    "entropy_rows_numba_antigravity",
    "score_spectral_entropy_numba_antigravity",
    "strict_confirmation_numba_antigravity",
)
