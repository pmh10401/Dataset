from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.exp206_210.core import empirical_upper_pvalues


@dataclass(frozen=True)
class NormalModes:
    centers: np.ndarray
    assignments: np.ndarray


def fit_normal_modes(features: np.ndarray, *, clusters: int, seed: int) -> NormalModes:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("TRAIN features must be a finite 2D array")
    k = int(clusters)
    if k < 1 or k > len(values):
        raise ValueError("cluster count outside TRAIN range")
    rng = np.random.default_rng(int(seed))
    initial = np.sort(rng.choice(len(values), size=k, replace=False))
    centers = values[initial].copy()
    assignments = np.zeros(len(values), dtype=int)
    for _ in range(50):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_assignments = np.argmin(distances, axis=1)
        next_centers = centers.copy()
        for cluster in range(k):
            members = values[next_assignments == cluster]
            if len(members):
                next_centers[cluster] = np.mean(members, axis=0)
        if np.array_equal(next_assignments, assignments) and np.allclose(next_centers, centers):
            assignments = next_assignments
            centers = next_centers
            break
        assignments = next_assignments
        centers = next_centers
    order = np.lexsort(np.rot90(centers)) if centers.shape[1] > 1 else np.argsort(centers[:, 0])
    inverse = np.empty(k, dtype=int)
    inverse[order] = np.arange(k)
    return NormalModes(centers=centers[order], assignments=inverse[assignments])


def local_empirical_pvalues(
    train_scores: np.ndarray,
    train_modes: np.ndarray,
    query_scores: np.ndarray,
    query_modes: np.ndarray,
) -> np.ndarray:
    train = np.asarray(train_scores, dtype=float).reshape(-1)
    modes = np.asarray(train_modes, dtype=int).reshape(-1)
    query = np.asarray(query_scores, dtype=float).reshape(-1)
    assigned = np.asarray(query_modes, dtype=int).reshape(-1)
    if len(train) != len(modes) or len(query) != len(assigned):
        raise ValueError("score and mode vectors must align")
    output = np.ones(len(query), dtype=float)
    for mode in np.unique(assigned):
        reference = train[modes == mode]
        selected = np.flatnonzero(assigned == mode)
        if len(reference):
            output[selected] = empirical_upper_pvalues(reference, query[selected])
    return output

