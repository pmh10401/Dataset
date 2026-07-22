from __future__ import annotations

from collections.abc import Mapping, Sequence, Set


POLICIES = (
    "verified_1of4",
    "reconstruction_only_residual",
    "wavelet_only_residual",
    "reconstruction_wavelet_cross_confirmed",
    "existing_1of4_or_reconstruction_wavelet",
    "six_source_2of6",
)


def policies_for_support(
    *, existing_support_count: int, reconstruction_support: bool, wavelet_support: bool,
) -> list[str]:
    existing = int(existing_support_count)
    if existing < 0 or existing > 4:
        raise ValueError("existing support count must be between zero and four")
    reconstruction = bool(reconstruction_support)
    wavelet = bool(wavelet_support)
    policies: list[str] = []
    if existing >= 1:
        policies.append("verified_1of4")
    if existing == 0 and reconstruction:
        policies.append("reconstruction_only_residual")
    if existing == 0 and wavelet and not reconstruction:
        policies.append("wavelet_only_residual")
    if existing == 0 and reconstruction and wavelet:
        policies.append("reconstruction_wavelet_cross_confirmed")
    if existing >= 1 or (reconstruction and wavelet):
        policies.append("existing_1of4_or_reconstruction_wavelet")
    if existing + int(reconstruction) + int(wavelet) >= 2:
        policies.append("six_source_2of6")
    return policies


def build_routing(
    *,
    dataset: str,
    hard_indices: Sequence[int],
    standard_indices: Sequence[int],
    evidence_by_index: Mapping[int, Mapping[str, object]],
    wavelet_support: Set[int],
) -> dict[str, object]:
    hard = sorted(set(map(int, hard_indices)))
    standard = sorted(set(map(int, standard_indices)))
    if len(hard) != len(list(hard_indices)) or len(standard) != len(list(standard_indices)):
        raise ValueError("duplicate candidate index")
    if set(hard) & set(standard):
        raise ValueError("Hard and Standard candidates overlap")
    if set(map(int, evidence_by_index)) != set(standard):
        raise ValueError("candidate evidence coverage mismatch")

    promoted = {policy: [] for policy in POLICIES}
    records: list[dict[str, object]] = []
    for index in standard:
        evidence = evidence_by_index[index]
        existing = int(evidence["existing_support_count"])
        reconstruction = bool(evidence["reconstruction_support"])
        wavelet = index in wavelet_support
        policies = policies_for_support(
            existing_support_count=existing,
            reconstruction_support=reconstruction,
            wavelet_support=wavelet,
        )
        for policy in policies:
            promoted[policy].append(index)
        records.append({
            "dataset": str(dataset),
            "candidate_index": index,
            "existing_support_count": existing,
            "reconstruction_support": reconstruction,
            "wavelet_support": wavelet,
            "six_source_support_count": existing + int(reconstruction) + int(wavelet),
            "eligible_expansion_policies": policies,
        })
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
