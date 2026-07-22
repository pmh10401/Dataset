from __future__ import annotations

from collections.abc import Mapping, Sequence

from experiments.exp293_294_train_only_moe_lite.routing import GATE_VARIANTS


EXPANSION_POLICIES = (
    "all_gates_two_experts",
    "all_gates_any_support",
    "majority_gates_any_support",
    "any_gate_any_support",
)


def expansion_policies_for_counts(selected_support_counts: Sequence[int]) -> list[str]:
    counts = tuple(map(int, selected_support_counts))
    if len(counts) != len(GATE_VARIANTS):
        raise ValueError("exactly three Gate support counts are required")
    if any(count < 0 or count > 2 for count in counts):
        raise ValueError("selected expert support count must be between zero and two")
    supported_gates = sum(count >= 1 for count in counts)
    policies: list[str] = []
    if all(count == 2 for count in counts):
        policies.append("all_gates_two_experts")
    if supported_gates == len(GATE_VARIANTS):
        policies.append("all_gates_any_support")
    if supported_gates >= 2:
        policies.append("majority_gates_any_support")
    if supported_gates >= 1:
        policies.append("any_gate_any_support")
    return policies


def build_dataset_expansion(gate_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_gate = {str(row["gate_variant"]): row for row in gate_rows}
    if set(by_gate) != set(GATE_VARIANTS) or len(gate_rows) != len(GATE_VARIANTS):
        raise ValueError("three unique Gate variants are required")
    rows = [by_gate[variant] for variant in GATE_VARIANTS]
    datasets = {str(row["dataset"]) for row in rows}
    if len(datasets) != 1:
        raise ValueError("Gate rows must belong to one dataset")
    dataset = next(iter(datasets))

    hard_sets = [set(map(int, row["automatic_hard_indices"])) for row in rows]
    standard_sets = [set(map(int, row["standard_review_indices"])) for row in rows]
    if any(values != hard_sets[0] for values in hard_sets[1:]):
        raise ValueError("Hard candidate mismatch across Gates")
    if any(values != standard_sets[0] for values in standard_sets[1:]):
        raise ValueError("Standard candidate mismatch across Gates")
    if hard_sets[0] & standard_sets[0]:
        raise ValueError("Hard and Standard candidates overlap")

    record_maps: list[dict[int, Mapping[str, object]]] = []
    for row, expected in zip(rows, standard_sets, strict=True):
        records = {
            int(record["candidate_index"]): record
            for record in row["candidate_rank_records"]
        }
        if set(records) != expected:
            raise ValueError("Standard rank-record coverage mismatch")
        record_maps.append(records)

    promoted = {policy: [] for policy in EXPANSION_POLICIES}
    candidate_records: list[dict[str, object]] = []
    for index in sorted(standard_sets[0]):
        counts = [int(records[index]["selected_expert_support_count"]) for records in record_maps]
        policies = expansion_policies_for_counts(counts)
        for policy in policies:
            promoted[policy].append(index)
        candidate_records.append(
            {
                "dataset": dataset,
                "candidate_index": index,
                "gate_selected_support_counts": {
                    variant: count for variant, count in zip(GATE_VARIANTS, counts, strict=True)
                },
                "supported_gate_count": sum(count >= 1 for count in counts),
                "two_expert_gate_count": sum(count == 2 for count in counts),
                "eligible_expansion_policies": policies,
            }
        )

    for narrow, broad in zip(EXPANSION_POLICIES, EXPANSION_POLICIES[1:], strict=False):
        if not set(promoted[narrow]) <= set(promoted[broad]):
            raise ValueError("expansion policies are not nested")
    hard = sorted(hard_sets[0])
    expanded = {
        policy: sorted(set(hard) | set(indices))
        for policy, indices in promoted.items()
    }
    return {
        "dataset": dataset,
        "n_test": int(rows[0]["n_test"]),
        "automatic_hard_indices": hard,
        "standard_review_indices": sorted(standard_sets[0]),
        "candidate_policy_records": candidate_records,
        "promoted_indices_by_policy": promoted,
        "expanded_hard_indices_by_policy": expanded,
    }
