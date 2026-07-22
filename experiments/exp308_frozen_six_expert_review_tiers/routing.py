from __future__ import annotations

from collections.abc import Sequence


TIER_LABELS_KO = {
    "strong_review": "강한 검토",
    "medium_review": "중간 검토",
    "standard_review": "일반 검토",
}


def review_rank_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        -int(row["selected_expert_support_count"]),
        -int(row["independent_group_support_count"]),
        -int(row["verified_expert_support_count"]),
        int(row["expert_rank_sum"]),
        str(row["dataset"]),
        int(row["candidate_index"]),
    )


def partition_review_records(
    rows: Sequence[dict[str, object]], *, strong_count: int, medium_count: int,
) -> list[dict[str, object]]:
    ordered = sorted((dict(row) for row in rows), key=review_rank_key)
    keys = [(str(row["dataset"]), int(row["candidate_index"])) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate dataset/candidate index in review universe")
    if strong_count < 0 or medium_count < 0 or strong_count + medium_count > len(ordered):
        raise ValueError("invalid frozen review tier counts")
    for rank, row in enumerate(ordered, start=1):
        if rank <= strong_count:
            tier = "strong_review"
        elif rank <= strong_count + medium_count:
            tier = "medium_review"
        else:
            tier = "standard_review"
        row["global_review_rank"] = rank
        row["review_tier"] = tier
        row["review_tier_ko"] = TIER_LABELS_KO[tier]
    return ordered
