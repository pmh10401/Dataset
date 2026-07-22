from __future__ import annotations

from collections.abc import Mapping, Set

import numpy as np


NONLEARNING_GROUPS = {
    "multiscale_patch_discord": "local_shape",
    "multiwindow_change_point": "transition",
    "wavelet_phase_coherence": "frequency",
}

FULL_SOURCE_GROUPS = {
    **NONLEARNING_GROUPS,
    "predictive_residual": "prediction",
    "contrastive_normal_memory": "representation",
    "deep_svdd_normal_boundary": "representation",
}


def independent_group_count(
    support: Set[str], groups: Mapping[str, str | Set[str]],
) -> int:
    if groups and all(isinstance(value, set) for value in groups.values()):
        return sum(bool(set(members) & set(support)) for members in groups.values())
    return len({str(groups[name]) for name in support if name in groups})


def fuse_indices(
    evidence: Mapping[str, Set[int]], *, exact_normal: Set[int], minimum_groups: int,
    excluded: Set[int] | None = None, groups: Mapping[str, str] = NONLEARNING_GROUPS,
) -> set[int]:
    candidates = set().union(*(set(indices) for indices in evidence.values())) if evidence else set()
    output = set()
    for index in candidates:
        support = {source for source, indices in evidence.items() if index in indices}
        if independent_group_count(support, groups) >= int(minimum_groups):
            output.add(int(index))
    return output - set(exact_normal) - set(excluded or set())


def fused_evidence_arrays(
    pvalues_by_source: Mapping[str, np.ndarray], *, attainable_by_source: Mapping[str, float],
) -> dict[str, list[float]]:
    names = sorted(pvalues_by_source)
    if not names:
        return {"bonferroni_p": [], "mean_e": []}
    matrix = np.vstack([np.asarray(pvalues_by_source[name], dtype=float) for name in names])
    bonferroni = np.minimum(1.0, len(names) * matrix.min(axis=0))
    evalues = []
    for row, name in zip(matrix, names):
        floor = max(float(attainable_by_source[name]), np.finfo(float).tiny)
        evalues.append(0.5 / np.sqrt(np.maximum(row, floor)))
    return {
        "bonferroni_p": bonferroni.tolist(),
        "mean_e": np.mean(np.vstack(evalues), axis=0).tolist(),
    }


def close_normal_guard_indices(
    pvalues_by_source: Mapping[str, np.ndarray], *, minimum_sources: int = 3,
    normal_p: float = 0.5, normal_fraction: float = 0.8,
) -> set[int]:
    if not pvalues_by_source:
        return set()
    matrix = np.vstack([np.asarray(values, dtype=float) for values in pvalues_by_source.values()])
    if len(matrix) < int(minimum_sources):
        return set()
    required = int(np.ceil(len(matrix) * float(normal_fraction)))
    return set(np.flatnonzero(np.count_nonzero(matrix >= float(normal_p), axis=0) >= required).astype(int).tolist())
