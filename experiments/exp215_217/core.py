from __future__ import annotations

import numpy as np

from experiments.exp162.full_coverage_score_core import (
    robust_score_pair,
    shapelet_distance_features,
    z_normalize,
)


PROTOTYPE_LENGTHS = (8, 16, 32, 64)
PROTOTYPES_PER_LENGTH = 8


def _normalize_window(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return ((values - values.mean()) / (values.std() + 1e-6)).astype(np.float32)


def fixed_kernel_dictionary() -> list[np.ndarray]:
    """Return a data-independent analytic dictionary shared by every fold."""
    kernels: list[np.ndarray] = []
    for length in PROTOTYPE_LENGTHS:
        position = np.linspace(0.0, 1.0, length, dtype=np.float32)
        for number in range(PROTOTYPES_PER_LENGTH):
            frequency = 1 + number // 2
            phase = 0.0 if number % 2 == 0 else np.pi / 2.0
            wave = np.sin(2.0 * np.pi * frequency * position + phase)
            if number >= 4:
                wave *= 0.5 + position
            kernels.append(_normalize_window(wave))
    return kernels


def _signature(window: np.ndarray, bins: int = 16) -> np.ndarray:
    effective_bins = min(int(bins), len(window))
    edges = np.linspace(0, len(window), effective_bins + 1, dtype=int)
    compact = np.asarray(
        [window[edges[index]:edges[index + 1]].mean() for index in range(effective_bins)],
        dtype=np.float32,
    )
    return np.pad(compact, (0, bins - effective_bins), mode="edge")


def central_normal_medoid_dictionary(x_fit: np.ndarray) -> list[np.ndarray]:
    """Choose recurrent, central normal windows using only the fit fold."""
    values = z_normalize(np.asarray(x_fit, dtype=np.float32)).astype(np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("central medoid dictionary requires at least two TRAIN-normal rows")
    dictionary: list[np.ndarray] = []
    row_count, width = values.shape
    row_indices = np.unique(np.linspace(0, row_count - 1, min(row_count, 128), dtype=int))
    for length in PROTOTYPE_LENGTHS:
        if length > width:
            continue
        starts = np.unique(np.linspace(0, width - length, 8, dtype=int))
        candidates = [_normalize_window(values[row, start:start + length]) for row in row_indices for start in starts]
        signatures = np.vstack([_signature(candidate) for candidate in candidates])
        center = np.median(signatures, axis=0)
        distance = np.sum((signatures - center) ** 2, axis=1)
        order = np.lexsort((np.arange(len(candidates)), distance))
        for index in order[:PROTOTYPES_PER_LENGTH]:
            dictionary.append(candidates[int(index)])
    if not dictionary:
        raise ValueError("no valid normal windows for medoid dictionary")
    while len(dictionary) < len(PROTOTYPE_LENGTHS) * PROTOTYPES_PER_LENGTH:
        dictionary.append(dictionary[len(dictionary) % len(dictionary)].copy())
    return dictionary[: len(PROTOTYPE_LENGTHS) * PROTOTYPES_PER_LENGTH]


def fit_shape_source(
    mode: str, x_fit: np.ndarray, x_query: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a TRAIN-normal shape representation and score fit/query rows."""
    del seed  # Dictionaries are deterministic by contract.
    fit = z_normalize(np.asarray(x_fit, dtype=np.float32)).astype(np.float32)
    query = z_normalize(np.asarray(x_query, dtype=np.float32)).astype(np.float32)
    if mode == "fixed_dictionary_control":
        dictionary = [item for item in fixed_kernel_dictionary() if len(item) <= fit.shape[1]]
    elif mode == "normal_consensus_medoid":
        dictionary = central_normal_medoid_dictionary(fit)
    else:
        raise ValueError(f"unknown shape source mode: {mode}")
    fit_features = shapelet_distance_features(fit, dictionary)
    query_features = shapelet_distance_features(query, dictionary)
    return robust_score_pair(fit_features, query_features)
