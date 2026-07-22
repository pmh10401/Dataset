from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

from experiments.exp293_294_train_only_moe_lite.routing import PRIMARY_EXPERTS


def _confidence_tier(selected_support_count: int) -> str:
    if selected_support_count == 2:
        return "high"
    if selected_support_count == 1:
        return "medium"
    return "needs_confirmation"


def build_hard_confidence_records(
    *,
    dataset: str,
    hard_indices: Sequence[int],
    source_support: Mapping[str, Set[int]],
    selected_experts: Sequence[str],
) -> list[dict[str, object]]:
    selected = tuple(map(str, selected_experts))
    if len(selected) not in (0, 2) or len(set(selected)) != len(selected):
        raise ValueError("exactly two selected experts or Gate abstention are required")
    if any(expert not in PRIMARY_EXPERTS for expert in selected):
        raise ValueError("selected expert is not provenance-verified")

    indices = list(map(int, hard_indices))
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate Hard index")

    records: list[dict[str, object]] = []
    for index in sorted(indices):
        supporting = sorted(
            expert
            for expert in PRIMARY_EXPERTS
            if index in source_support.get(expert, set())
        )
        selected_supporting = sorted(set(supporting) & set(selected))
        selected_count = len(selected_supporting)
        records.append(
            {
                "dataset": str(dataset),
                "candidate_index": index,
                "confidence_tier": _confidence_tier(selected_count),
                "selected_expert_support_count": selected_count,
                "verified_expert_support_count": len(supporting),
                "gate_available": len(selected) == 2,
                "selected_supporting_experts": selected_supporting,
                "supporting_experts": supporting,
            }
        )
    return records
