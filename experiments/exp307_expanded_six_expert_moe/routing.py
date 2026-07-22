from __future__ import annotations

from collections.abc import Mapping, Sequence, Set


EXPERT_GROUPS = {
    "statistical": "distribution",
    "ordinal_pattern": "distribution",
    "structural": "structure",
    "change_point": "structure",
    "official_multirocket": "random_kernel",
    "masked_context": "reconstruction",
}
EXPERTS = tuple(EXPERT_GROUPS)
GATE_VARIANTS = (
    "risk_first_distinct3",
    "stability_first_distinct3",
    "balanced_distinct3",
)


def _eligible(audit: Mapping[str, Mapping[str, object]]) -> list[str]:
    return [
        name for name in EXPERTS
        if bool(audit.get(name, {}).get("available"))
        and bool(audit.get(name, {}).get("provenance_verified"))
    ]


def _ordered(audit: Mapping[str, Mapping[str, object]], variant: str) -> list[str]:
    names = _eligible(audit)
    if variant == "risk_first_distinct3":
        return sorted(names, key=lambda n: (float(audit[n]["risk_upper_95"]), n))
    if variant == "stability_first_distinct3":
        return sorted(names, key=lambda n: (
            not bool(audit[n].get("stability_observable", True)),
            -float(audit[n].get("stability", 0.0)),
            float(audit[n]["risk_upper_95"]), n,
        ))
    if variant != "balanced_distinct3":
        raise ValueError(f"unknown Gate variant: {variant}")
    risk_order = sorted(names, key=lambda n: (float(audit[n]["risk_upper_95"]), n))
    stability_order = sorted(names, key=lambda n: (
        not bool(audit[n].get("stability_observable", True)),
        -float(audit[n].get("stability", 0.0)), n,
    ))
    rank = {name: risk_order.index(name) + stability_order.index(name) for name in names}
    return sorted(names, key=lambda n: (rank[n], float(audit[n]["risk_upper_95"]), n))


def select_distinct_experts(
    audit: Mapping[str, Mapping[str, object]], *, variant: str, limit: int = 3,
) -> list[str]:
    selected: list[str] = []
    groups: set[str] = set()
    for name in _ordered(audit, variant):
        group = EXPERT_GROUPS[name]
        if group in groups:
            continue
        selected.append(name)
        groups.add(group)
        if len(selected) >= int(limit):
            break
    return selected


def order_eligible_experts(
    audit: Mapping[str, Mapping[str, object]], *, variant: str,
) -> list[str]:
    return _ordered(audit, variant)


def build_candidate_records(
    *, dataset: str, standard_indices: Sequence[int],
    support_by_index: Mapping[int, Set[str]], ordered_experts: Sequence[str],
    selected_experts: Sequence[str],
) -> list[dict[str, object]]:
    ordered = list(map(str, ordered_experts))
    selected = set(map(str, selected_experts))
    expert_rank = {name: rank for rank, name in enumerate(ordered)}
    records = []
    for index in sorted(set(map(int, standard_indices))):
        support = set(map(str, support_by_index.get(index, set()))) & set(EXPERTS)
        groups = {EXPERT_GROUPS[name] for name in support}
        records.append({
            "dataset": str(dataset),
            "candidate_index": index,
            "supporting_experts": sorted(support),
            "supporting_method_groups": sorted(groups),
            "selected_expert_support_count": len(support & selected),
            "verified_expert_support_count": len(support),
            "independent_group_support_count": len(groups),
            "expert_rank_sum": sum(expert_rank.get(name, len(ordered)) for name in support),
            "shadow_2groups": len(groups) >= 2,
            "shadow_3groups": len(groups) >= 3,
        })
    records.sort(key=lambda row: (
        -int(row["selected_expert_support_count"]),
        -int(row["independent_group_support_count"]),
        -int(row["verified_expert_support_count"]),
        int(row["expert_rank_sum"]), str(row["dataset"]), int(row["candidate_index"]),
    ))
    return records
