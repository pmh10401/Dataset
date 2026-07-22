from __future__ import annotations

from collections.abc import Mapping, Sequence

from experiments.exp293_294_train_only_moe_lite.routing import GATE_VARIANTS


CONSENSUS_TIERS = (
    "unanimous_high",
    "unanimous_supported",
    "partial_supported",
    "no_selected_support",
)


def _tier(*, supporting_gate_count: int, high_gate_count: int) -> str:
    if high_gate_count == len(GATE_VARIANTS):
        return "unanimous_high"
    if supporting_gate_count == len(GATE_VARIANTS):
        return "unanimous_supported"
    if supporting_gate_count > 0:
        return "partial_supported"
    return "no_selected_support"


def build_consensus_records(
    *, dataset: str, gate_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    by_variant = {str(row["gate_variant"]): row for row in gate_rows}
    if set(by_variant) != set(GATE_VARIANTS) or len(gate_rows) != len(GATE_VARIANTS):
        raise ValueError(f"Gate coverage mismatch: {dataset}")

    hard_sets = {
        variant: set(map(int, by_variant[variant]["automatic_hard_indices"]))
        for variant in GATE_VARIANTS
    }
    first = hard_sets[GATE_VARIANTS[0]]
    if any(values != first for values in hard_sets.values()):
        raise ValueError(f"Hard index mismatch between Gates: {dataset}")

    support_by_variant: dict[str, dict[int, int]] = {}
    for variant in GATE_VARIANTS:
        records = by_variant[variant]["hard_confidence_records"]
        support = {
            int(record["candidate_index"]): int(record["selected_expert_support_count"])
            for record in records
        }
        if set(support) != first:
            raise ValueError(f"Hard confidence record mismatch: {dataset}/{variant}")
        if any(count not in (0, 1, 2) for count in support.values()):
            raise ValueError(f"invalid selected expert support count: {dataset}/{variant}")
        support_by_variant[variant] = support

    output: list[dict[str, object]] = []
    for index in sorted(first):
        per_gate = {
            variant: support_by_variant[variant][index]
            for variant in GATE_VARIANTS
        }
        supporting_gate_count = sum(count >= 1 for count in per_gate.values())
        high_gate_count = sum(count == 2 for count in per_gate.values())
        output.append(
            {
                "dataset": str(dataset),
                "candidate_index": index,
                "consensus_tier": _tier(
                    supporting_gate_count=supporting_gate_count,
                    high_gate_count=high_gate_count,
                ),
                "supporting_gate_count": supporting_gate_count,
                "high_gate_count": high_gate_count,
                "per_gate_selected_support_count": per_gate,
                "abstained_gate_count": sum(
                    bool(by_variant[variant].get("gate_abstained", False))
                    for variant in GATE_VARIANTS
                ),
            }
        )
    return output
