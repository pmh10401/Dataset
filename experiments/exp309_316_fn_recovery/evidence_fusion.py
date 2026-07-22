from __future__ import annotations

from collections.abc import Mapping, Set

import numpy as np


NONLEARNING_GROUPS = {
    "multiscale_patch_discord": "local_shape",
    "multiwindow_change_point": "transition",
    "wavelet_phase_coherence": "frequency",
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
