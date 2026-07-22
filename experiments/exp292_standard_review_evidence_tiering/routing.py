from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from math import ceil


def build_stable_family_support(
    rows: Sequence[Mapping[str, object]], *, expected_variants: Sequence[str]
) -> set[int]:
    by_variant = {str(row["variant"]): row for row in rows}
    if set(by_variant) != set(expected_variants) or len(by_variant) != len(rows):
        raise ValueError("variant coverage mismatch")
    votes: Counter[int] = Counter()
    for row in by_variant.values():
        votes.update(set(map(int, row.get("candidate_indices", []))))
    required = ceil(len(expected_variants) / 2)
    return {index for index, count in votes.items() if count >= required}


def tier_standard_review(
    exp291_row: Mapping[str, object],
    family_support: Mapping[str, set[int] | None],
) -> dict[str, object]:
    standard = sorted(set(map(int, exp291_row.get("standard_review_indices", []))))
    available = sorted(name for name, support in family_support.items() if support is not None)
    support_by_index = {
        str(index): sorted(name for name, support in family_support.items() if support is not None and index in support)
        for index in standard
    }
    p1 = [index for index in standard if len(support_by_index[str(index)]) >= 2]
    p2 = [index for index in standard if len(support_by_index[str(index)]) == 1]
    p3 = [index for index in standard if len(support_by_index[str(index)]) == 0]
    if set(p1) | set(p2) | set(p3) != set(standard) or set(p1) & set(p2) or set(p1) & set(p3) or set(p2) & set(p3):
        raise ValueError("review tier partition mismatch")
    row = dict(exp291_row)
    row.update(
        {
            "experiment_id": "experiment_292_standard_review_evidence_tiering",
            "review_priority_1_indices": p1,
            "review_priority_2_indices": p2,
            "review_priority_3_indices": p3,
            "review_support_by_index": support_by_index,
            "available_source_families": available,
            "unavailable_source_families": sorted(name for name, support in family_support.items() if support is None),
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "review_only": True,
            "automatic_policy_changed": False,
        }
    )
    return row

