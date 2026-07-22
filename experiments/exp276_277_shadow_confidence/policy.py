from __future__ import annotations

import math
from collections.abc import Sequence


ALPHA = 0.005
MIN_GROUPS = 2
DEEP_MARGIN = 0.5
MAX_RISK = 0.005
MAX_MARGIN_STD = 0.25


def normalized_tail_margin(pvalue: float, reference_count: int, alpha: float = ALPHA) -> float:
    floor = 1.0 / (int(reference_count) + 1.0)
    if reference_count <= 0 or not math.isfinite(pvalue) or pvalue > alpha + 1e-15:
        return 0.0
    denominator = float(alpha) - floor
    if denominator <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (float(alpha) - float(pvalue)) / denominator))


def confidence_record(margins: Sequence[float], risks: Sequence[float]) -> dict[str, object]:
    values = [float(value) for value in margins]
    risk_values = [float(value) for value in risks]
    if len(values) != len(risk_values) or any(not math.isfinite(x) for x in values + risk_values):
        raise ValueError("finite aligned margin and risk values are required")
    count = len(values)
    if not count:
        return {
            "continuous_group_count": 0, "minimum_margin": 0.0,
            "median_margin": 0.0, "mean_margin": 0.0, "margin_std": 0.0,
            "maximum_train_risk": 1.0, "confidence_score": 0.0,
            "guard_all_deep": False, "guard_median_deep": False,
            "guard_consistent_deep": False,
        }
    ordered = sorted(values)
    median = ordered[count // 2] if count % 2 else (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0
    mean = sum(values) / count
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / count)
    max_risk = max(risk_values)
    reliability = max(0.0, 1.0 - max_risk / 0.01)
    consistency = max(0.0, 1.0 - min(1.0, std / 0.5))
    eligible = count >= MIN_GROUPS and max_risk <= MAX_RISK
    return {
        "continuous_group_count": count,
        "minimum_margin": min(values), "median_margin": median,
        "mean_margin": mean, "margin_std": std,
        "maximum_train_risk": max_risk,
        "confidence_score": mean * reliability * consistency,
        "guard_all_deep": eligible and min(values) >= DEEP_MARGIN,
        "guard_median_deep": eligible and median >= DEEP_MARGIN,
        "guard_consistent_deep": eligible and mean >= DEEP_MARGIN and std <= MAX_MARGIN_STD,
    }
