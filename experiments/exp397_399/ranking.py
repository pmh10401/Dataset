from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from experiments.exp397_399.contracts import PRIMITIVE_SCORES, VARIANTS


def _enrich_candidate(
    row: Mapping[str, Any],
    alpha: float,
) -> dict[str, Any]:
    result = dict(row)
    p_values = {
        name: float(value)
        for name, value in dict(row.get("p_values", {})).items()
    }
    primitive = [p_values.get(name, 1.0) for name in PRIMITIVE_SCORES]
    support_count = sum(value <= alpha for value in primitive)
    composite_p = p_values.get("composite", 1.0)
    grammar_evidence = support_count > 0 or composite_p <= alpha
    original_tier = int(row["exp292_tier"])
    result.update(
        {
            "alpha": float(alpha),
            "grammar_evidence": bool(grammar_evidence),
            "grammar_support_count": int(support_count),
            "composite_p_value": float(composite_p),
            "primitive_min_p_value": float(min(primitive, default=1.0)),
            "effective_tier": original_tier,
        }
    )
    return result


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]],
    variant: str,
    alpha: float,
) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ranking variant: {variant}")

    rows = [_enrich_candidate(row, alpha) for row in candidates]
    if variant == "additive_grammar_family":
        for row in rows:
            if row["grammar_evidence"]:
                row["effective_tier"] = max(1, int(row["exp292_tier"]) - 1)

    def stable_tail(row: Mapping[str, Any]) -> tuple[str, int]:
        return str(row["dataset"]), int(row["candidate_index"])

    if variant == "exp292_baseline":
        key = lambda row: (int(row["exp292_tier"]), *stable_tail(row))
    elif variant == "tier_preserving_grammar":
        key = lambda row: (
            int(row["exp292_tier"]),
            -int(bool(row["grammar_evidence"])),
            -int(row["grammar_support_count"]),
            float(row["composite_p_value"]),
            float(row["primitive_min_p_value"]),
            *stable_tail(row),
        )
    elif variant == "additive_grammar_family":
        key = lambda row: (
            int(row["effective_tier"]),
            -int(bool(row["grammar_evidence"])),
            -int(row["grammar_support_count"]),
            float(row["composite_p_value"]),
            float(row["primitive_min_p_value"]),
            int(row["exp292_tier"]),
            *stable_tail(row),
        )
    else:
        key = lambda row: (
            -int(bool(row["grammar_evidence"])),
            -int(row["grammar_support_count"]),
            float(row["composite_p_value"]),
            float(row["primitive_min_p_value"]),
            *stable_tail(row),
        )

    ordered = sorted(rows, key=key)
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["variant"] = variant
    return ordered
