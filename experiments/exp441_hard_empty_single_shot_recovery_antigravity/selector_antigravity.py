"""Pure Hard-empty, one-per-dataset, budget-30 selection for Exp441."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Iterable

from .contracts_antigravity import (
    CandidateEvidenceAntigravity,
    FrozenSourceSnapshotAntigravity,
    RankedCandidateAntigravity,
    SelectionResultAntigravity,
)


_GLOBAL_CANDIDATE_BUDGET_ANTIGRAVITY = 30


def _canonical_candidate_order_antigravity(
    candidate: CandidateEvidenceAntigravity,
) -> tuple[str, int]:
    return candidate.dataset, candidate.instance_index


def _validated_coordinates_antigravity(
    candidates: Iterable[object],
) -> tuple[tuple[str, int], ...]:
    try:
        values = tuple(candidates)
    except TypeError as error:
        raise ValueError("selected candidates are invalid") from error
    coordinates: list[tuple[str, int]] = []
    for item in values:
        if isinstance(item, (CandidateEvidenceAntigravity, RankedCandidateAntigravity)):
            coordinate = item.coordinate
        else:
            raise ValueError("selected candidates are invalid")
        dataset, index = coordinate
        if not isinstance(dataset, str) or not dataset or type(index) is not int or index < 0:
            raise ValueError("selected coordinate is invalid")
        coordinates.append((dataset, index))
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("selected coordinates are duplicated")
    return tuple(sorted(coordinates))


def selected_coordinate_identity_sha256_antigravity(candidates: Iterable[object]) -> str:
    """Hash selected coordinates as compact canonical JSONL with one newline per row."""
    coordinates = _validated_coordinates_antigravity(candidates)
    payload = b"".join(
        json.dumps(
            {"dataset": dataset, "instance_index": index},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        for dataset, index in coordinates
    )
    return sha256(payload).hexdigest()


def select_hard_empty_single_shot_antigravity(
    snapshot: FrozenSourceSnapshotAntigravity,
) -> SelectionResultAntigravity:
    """Select at most one provisional candidate per Hard-empty dataset.

    The source snapshot is label-free.  Eligibility is determined solely by the
    frozen Hard coordinates; agreement count is already part of candidate
    eligibility and is intentionally absent from every ranking key.
    """
    if not isinstance(snapshot, FrozenSourceSnapshotAntigravity):
        raise ValueError("source snapshot is invalid")
    source_candidates = tuple(snapshot.provisional_candidates)
    coordinates = tuple(item.coordinate for item in source_candidates)
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("provisional coordinates are duplicated")

    # Canonicalize the copied source population so worker/input completion order
    # cannot change payload bytes or any downstream rank assignment.
    provisional = tuple(sorted(source_candidates, key=_canonical_candidate_order_antigravity))
    hard_coordinates = frozenset(snapshot.hard_coordinates)
    hard_datasets = frozenset(dataset for dataset, _ in hard_coordinates)
    outside_hard = tuple(
        item for item in provisional if item.coordinate not in hard_coordinates
    )
    eligible = tuple(
        item for item in outside_hard if item.dataset not in hard_datasets
    )

    grouped: dict[str, list[CandidateEvidenceAntigravity]] = {}
    for item in eligible:
        grouped.setdefault(item.dataset, []).append(item)
    local_winners = tuple(
        min(grouped[dataset], key=lambda item: item.dataset_key)
        for dataset in sorted(grouped)
    )
    ordered_winners = tuple(sorted(local_winners, key=lambda item: item.global_key))
    winners = tuple(
        RankedCandidateAntigravity(
            candidate=item,
            global_rank=index,
            selected=index <= _GLOBAL_CANDIDATE_BUDGET_ANTIGRAVITY,
            selection_rank=index if index <= _GLOBAL_CANDIDATE_BUDGET_ANTIGRAVITY else None,
        )
        for index, item in enumerate(ordered_winners, start=1)
    )
    selected = tuple(item for item in winners if item.selected)
    return SelectionResultAntigravity(
        provisional_candidates=provisional,
        outside_hard_candidates=outside_hard,
        eligible_candidates=eligible,
        winners=winners,
        selected=selected,
        hard_coordinates=hard_coordinates,
    )


__all__ = [
    "select_hard_empty_single_shot_antigravity",
    "selected_coordinate_identity_sha256_antigravity",
]
