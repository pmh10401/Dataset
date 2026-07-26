from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def rankdata_average(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("rank input must be finite")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    position = 0
    while position < len(array):
        end = position + 1
        while end < len(array) and array[order[end]] == array[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def spearman_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float | None:
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.size != right_array.size:
        raise ValueError("Spearman inputs must have equal length")
    if left_array.size < 2:
        return None
    left_ranks = rankdata_average(left_array)
    right_ranks = rankdata_average(right_array)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered))
        * np.sqrt(np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def robust_normalize(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("normalization input must be finite")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = mad if mad > 1e-12 else 1.0
    return (array - median) / scale


def collapse_correlated_sources(
    sources: Sequence[str],
    correlations: Mapping[tuple[str, str], float | None],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    ordered = tuple(sources)
    parent = {source: source for source in ordered}

    def find(source: str) -> str:
        while parent[source] != source:
            parent[source] = parent[parent[source]]
            source = parent[source]
        return source

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        left_index = ordered.index(left_root)
        right_index = ordered.index(right_root)
        if left_index <= right_index:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            value = correlations.get((left, right))
            if value is None:
                value = correlations.get((right, left))
            if value is not None and abs(float(value)) >= float(threshold):
                union(left, right)

    components: dict[str, list[str]] = {}
    for source in ordered:
        components.setdefault(find(source), []).append(source)
    return tuple(tuple(component) for component in components.values())
