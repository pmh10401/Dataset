from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from experiments.exp293_294_train_only_moe_lite.routing import PRIMARY_EXPERTS


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def representative_train_score(train_variants: Mapping[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(values, dtype=float).reshape(-1) for _, values in sorted(train_variants.items())]
    if not arrays or len({len(values) for values in arrays}) != 1:
        raise ValueError("TRAIN variant length mismatch")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("TRAIN variants contain non-finite values")
    return np.mean(np.vstack([_rankdata(values) for values in arrays]), axis=0)


def _eligible(audits: Mapping[str, Mapping[str, object]]) -> list[str]:
    return sorted(
        expert
        for expert, row in audits.items()
        if expert in PRIMARY_EXPERTS
        and bool(row.get("available"))
        and bool(row.get("provenance_verified"))
    )


def _abs_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if float(np.std(left_rank)) <= 1e-12 or float(np.std(right_rank)) <= 1e-12:
        return 1.0
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return abs(value) if np.isfinite(value) else 1.0


def choose_diversity_pair(
    train_by_expert: Mapping[str, Mapping[str, np.ndarray]],
    audits: Mapping[str, Mapping[str, object]],
) -> list[str]:
    eligible = [expert for expert in _eligible(audits) if expert in train_by_expert]
    if len(eligible) < 2:
        return []
    representatives = {
        expert: representative_train_score(train_by_expert[expert])
        for expert in eligible
    }
    lengths = {len(values) for values in representatives.values()}
    if len(lengths) != 1:
        raise ValueError("expert TRAIN length mismatch")
    pairs = list(combinations(eligible, 2))
    selected = min(
        pairs,
        key=lambda pair: (
            _abs_spearman(representatives[pair[0]], representatives[pair[1]]),
            sum(float(audits[expert]["risk_upper_95"]) for expert in pair),
            pair,
        ),
    )
    return list(selected)


def choose_calibration_pair(audits: Mapping[str, Mapping[str, object]]) -> list[str]:
    eligible = _eligible(audits)
    if len(eligible) < 2:
        return []
    rank_sum = {expert: 0 for expert in eligible}
    orderings = (
        sorted(eligible, key=lambda expert: (float(audits[expert]["risk_upper_95"]), expert)),
        sorted(eligible, key=lambda expert: (float(audits[expert]["attainable_p"]), expert)),
        sorted(eligible, key=lambda expert: (float(audits[expert]["train_drift"]), expert)),
        sorted(eligible, key=lambda expert: (-float(audits[expert]["stability"]), expert)),
    )
    for ordering in orderings:
        for rank, expert in enumerate(ordering, start=1):
            rank_sum[expert] += rank
    return sorted(eligible, key=lambda expert: (rank_sum[expert], expert))[:2]
