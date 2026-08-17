"""Build the eight Exp456 routing inputs without consulting labels or confusion."""

from __future__ import annotations

from .contracts_antigravity import DatasetRoutingFeaturesAntigravity, FrozenRoutingAuthorityAntigravity


_TARGET_METHODS_ANTIGRAVITY = {
    "ECG5000": "hydra_local_gap",
    "ChlorineConcentration": "multirocket_local_gap",
}


def build_label_free_features_antigravity(
    authority: FrozenRoutingAuthorityAntigravity,
) -> tuple[DatasetRoutingFeaturesAntigravity, ...]:
    """Return frozen label-free rows in approved ECG5000 then ChlorineConcentration public order."""
    if type(authority) is not FrozenRoutingAuthorityAntigravity or authority.label_query_count != 0:
        raise ValueError("routing authority is invalid")
    selected = {family: [] for family in _TARGET_METHODS_ANTIGRAVITY}
    counts = {family: 0 for family in _TARGET_METHODS_ANTIGRAVITY}
    for row in authority.rows:
        expected_method = _TARGET_METHODS_ANTIGRAVITY.get(row.family)
        if expected_method is None:
            continue
        if row.method != expected_method or type(row.candidate_indices) is not frozenset:
            raise ValueError("frozen routing target differs")
        counts[row.family] += 1
        baseline = frozenset(row.baseline_indices)
        candidate = frozenset(row.candidate_indices)
        overlap = baseline & candidate
        union = baseline | candidate
        selected[row.family].append(DatasetRoutingFeaturesAntigravity(
            dataset=row.dataset,
            family=row.family,
            target_length=row.target_length,
            baseline_count=len(baseline),
            candidate_count=len(candidate),
            overlap_count=len(overlap),
            union_count=len(union),
            added_count=len(candidate - baseline),
            removed_count=len(baseline - candidate),
            overlap_ppm=(len(overlap) * 1_000_000) // max(1, len(union)),
        ))
    result = tuple(selected["ECG5000"] + selected["ChlorineConcentration"])
    if counts != {"ECG5000": 5, "ChlorineConcentration": 3} or len(result) != 8:
        raise ValueError("frozen routing target population differs")
    return result


__all__ = ["build_label_free_features_antigravity"]
