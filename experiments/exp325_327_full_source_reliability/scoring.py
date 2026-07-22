from __future__ import annotations

import math
from collections.abc import Mapping

from .config import FIXED_PRIORS


def wilson_upper(false_alerts: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 1.0
    p = max(0, min(int(false_alerts), int(total))) / int(total)
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return min(1.0, (center + radius) / denominator)


def wilson_reliability(false_alerts: int, total: int) -> dict[str, float | int]:
    upper = wilson_upper(false_alerts, total)
    floor = 1.0 / (int(total) + 1) if total > 0 else 1.0
    reliability = max(0.0, -math.log10(max(upper, floor)))
    return {
        "train_exceed_count": int(false_alerts),
        "train_score_count": int(total),
        "wilson_upper": float(upper),
        "reliability": float(reliability),
    }


def adaptive_weights(reliabilities: Mapping[str, float], mode: str) -> dict[str, float]:
    if mode == "fixed_priors":
        return dict(FIXED_PRIORS)
    if mode == "prior_reliability":
        raw = {name: prior * (0.25 + max(0.0, float(reliabilities.get(name, 0.0))))
               for name, prior in FIXED_PRIORS.items()}
    elif mode == "pure_reliability":
        raw = {name: max(0.0, float(reliabilities.get(name, 0.0))) for name in FIXED_PRIORS}
        if not sum(raw.values()):
            return dict(FIXED_PRIORS)
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def choose_replacement(*, base_indices: set[int], weak_indices: set[int],
                       top_candidate: int | None, top_support: int,
                       top_best_rank: int, score_gain: float,
                       train_count: int) -> set[int]:
    selected = set(base_indices)
    if train_count <= 10 or len(base_indices) > 1 or not weak_indices or top_candidate is None:
        return selected
    if top_support >= 2 and top_best_rank <= 3 and score_gain >= 0.04:
        return (selected - weak_indices) | {int(top_candidate)}
    return selected
