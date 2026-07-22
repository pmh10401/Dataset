from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

from experiments.exp293_294_train_only_moe_lite.routing import PRIMARY_EXPERTS


POLICIES = (
    "verified_2of4",
    "verified_1of4",
    "five_source_3of5",
    "five_source_2of5",
    "five_source_1of5",
    "reconstruction_and_existing",
    "reconstruction_only_residual",
)


def policies_for_support(*, existing_support_count: int, reconstruction_support: bool) -> list[str]:
    existing = int(existing_support_count)
    if existing < 0 or existing > len(PRIMARY_EXPERTS):
        raise ValueError("existing support count must be between zero and four")
    total = existing + int(bool(reconstruction_support))
    policies: list[str] = []
    if existing >= 2:
        policies.append("verified_2of4")
    if existing >= 1:
        policies.append("verified_1of4")
    if total >= 3:
        policies.append("five_source_3of5")
    if total >= 2:
        policies.append("five_source_2of5")
    if total >= 1:
        policies.append("five_source_1of5")
    if reconstruction_support and existing >= 1:
        policies.append("reconstruction_and_existing")
    if reconstruction_support and existing == 0:
        policies.append("reconstruction_only_residual")
    return policies


def build_five_source_routing(
    *,
    dataset: str,
    hard_indices: Sequence[int],
    standard_indices: Sequence[int],
    existing_support_by_index: Mapping[int, Set[str]],
    reconstruction_support: Set[int],
) -> dict[str, object]:
    hard = sorted(set(map(int, hard_indices)))
    standard = sorted(set(map(int, standard_indices)))
    if len(hard) != len(list(hard_indices)) or len(standard) != len(list(standard_indices)):
        raise ValueError("duplicate candidate index")
    if set(hard) & set(standard):
        raise ValueError("Hard and Standard candidates overlap")
    normalized_support = {
        int(index): set(map(str, experts)) & set(PRIMARY_EXPERTS)
        for index, experts in existing_support_by_index.items()
    }
    if set(normalized_support) != set(standard):
        raise ValueError("existing support coverage mismatch")

    promoted = {policy: [] for policy in POLICIES}
    records: list[dict[str, object]] = []
    for index in standard:
        existing = normalized_support[index]
        reconstruction = index in reconstruction_support
        policies = policies_for_support(
            existing_support_count=len(existing),
            reconstruction_support=reconstruction,
        )
        for policy in policies:
            promoted[policy].append(index)
        records.append(
            {
                "dataset": str(dataset),
                "candidate_index": index,
                "existing_supporting_experts": sorted(existing),
                "existing_support_count": len(existing),
                "reconstruction_support": reconstruction,
                "five_source_support_count": len(existing) + int(reconstruction),
                "eligible_expansion_policies": policies,
            }
        )
    return {
        "dataset": str(dataset),
        "automatic_hard_indices": hard,
        "standard_review_indices": standard,
        "candidate_policy_records": records,
        "promoted_indices_by_policy": promoted,
        "expanded_hard_indices_by_policy": {
            policy: sorted(set(hard) | set(indices)) for policy, indices in promoted.items()
        },
    }
