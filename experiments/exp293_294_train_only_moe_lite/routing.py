from __future__ import annotations

from typing import Mapping, Sequence


PRIMARY_EXPERTS = ("statistical", "structural", "change_point", "official_multirocket")
GATE_VARIANTS = ("risk_first_top2", "stability_first_top2", "balanced_rank_top2")


def _eligible(audit_by_expert: Mapping[str, Mapping[str, object]]) -> list[str]:
    return sorted(
        name for name, row in audit_by_expert.items()
        if name in PRIMARY_EXPERTS and bool(row.get("available")) and bool(row.get("provenance_verified"))
    )


def gate_experts(audit_by_expert: Mapping[str, Mapping[str, object]], *, variant: str) -> list[str]:
    names = _eligible(audit_by_expert)
    if variant == "risk_first_top2":
        return sorted(names, key=lambda name: (float(audit_by_expert[name]["risk_upper_95"]), -float(audit_by_expert[name]["stability"]), name))
    if variant == "stability_first_top2":
        return sorted(names, key=lambda name: (-float(audit_by_expert[name]["stability"]), float(audit_by_expert[name]["risk_upper_95"]), name))
    if variant != "balanced_rank_top2":
        raise ValueError(f"unknown Gate variant: {variant}")
    rank_sum = {name: 0 for name in names}
    orderings = (
        sorted(names, key=lambda name: (float(audit_by_expert[name]["risk_upper_95"]), name)),
        sorted(names, key=lambda name: (-float(audit_by_expert[name]["stability"]), name)),
        sorted(names, key=lambda name: (float(audit_by_expert[name]["attainable_p"]), name)),
    )
    for ordering in orderings:
        for rank, name in enumerate(ordering, start=1):
            rank_sum[name] += rank
    return sorted(names, key=lambda name: (rank_sum[name], name))


def rank_standard_candidates(
    *, standard_indices: Sequence[int], support_by_index: Mapping[str, Sequence[str]],
    ordered_experts: Sequence[str],
) -> list[int]:
    expert_rank = {name: rank for rank, name in enumerate(ordered_experts)}
    selected = set(ordered_experts[:2])
    primary = set(PRIMARY_EXPERTS)

    def key(index: int) -> tuple[object, ...]:
        support = set(map(str, support_by_index.get(str(index), ()))) & primary
        selected_count = len(support & selected)
        rank_score = sum(expert_rank.get(name, len(expert_rank)) for name in support)
        return (-selected_count, -len(support), rank_score, int(index))

    values = sorted(set(map(int, standard_indices)), key=key)
    if set(values) != set(map(int, standard_indices)):
        raise ValueError("MoE ranking changed Standard candidate set")
    return values


def candidate_rank_records(
    *, dataset: str, standard_indices: Sequence[int],
    support_by_index: Mapping[str, Sequence[str]], ordered_experts: Sequence[str],
) -> list[dict[str, object]]:
    expert_rank = {name: rank for rank, name in enumerate(ordered_experts)}
    selected = set(ordered_experts[:2])
    primary = set(PRIMARY_EXPERTS)
    records = []
    for index in sorted(set(map(int, standard_indices))):
        support = sorted(set(map(str, support_by_index.get(str(index), ()))) & primary)
        records.append({
            "dataset": dataset,
            "candidate_index": index,
            "selected_expert_support_count": len(set(support) & selected),
            "verified_expert_support_count": len(support),
            "supporting_experts": support,
            "expert_rank_sum": sum(expert_rank.get(name, len(expert_rank)) for name in support),
        })
    records.sort(key=lambda row: (
        -int(row["selected_expert_support_count"]),
        -int(row["verified_expert_support_count"]),
        int(row["expert_rank_sum"]),
        str(row["dataset"]),
        int(row["candidate_index"]),
    ))
    return records

