from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence


def _validated_indices(
    values: Iterable[int],
    *,
    n_test: int,
    label: str,
) -> set[int]:
    indices = list(values)
    if any(type(index) is not int or index < 0 or index >= n_test for index in indices):
        raise ValueError(f"{label} contains out-of-range indices")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} contains duplicate indices")
    return set(indices)


def _true_indices(labels: Sequence[int | str]) -> set[int]:
    return {index for index, label in enumerate(labels) if str(label) == "1"}


def classify_baseline(
    hard_indices: Iterable[int],
    labels: Sequence[int | str],
) -> str:
    hard = _validated_indices(
        hard_indices,
        n_test=len(labels),
        label="Hard lane",
    )
    if not hard:
        return "zero_no_hard"
    if hard & _true_indices(labels):
        return "hard_has_tp"
    return "zero_wrong_hard"


def build_vote_candidates(
    evidence: Mapping[str, Mapping[str, object]],
    *,
    vote_threshold: int,
    n_test: int,
    excluded_indices: Iterable[int],
) -> set[int]:
    if type(vote_threshold) is not int or vote_threshold < 1:
        raise ValueError("vote_threshold must be a positive integer")
    excluded = _validated_indices(
        excluded_indices,
        n_test=n_test,
        label="excluded_indices",
    )
    votes: Counter[int] = Counter()
    for source, payload in evidence.items():
        if payload.get("available") is not True:
            continue
        values = payload.get("indices")
        if not isinstance(values, list):
            raise ValueError(f"{source} indices must be a list")
        source_indices = _validated_indices(
            values,
            n_test=n_test,
            label=f"{source} evidence",
        )
        votes.update(source_indices - excluded)
    return {index for index, count in votes.items() if count >= vote_threshold}


def evaluate_recovery(
    *,
    candidate_indices: Iterable[int],
    hard_indices: Iterable[int],
    review_indices: Iterable[int],
    labels: Sequence[int | str],
) -> dict[str, int | float]:
    n_test = len(labels)
    candidates = _validated_indices(
        candidate_indices,
        n_test=n_test,
        label="candidate_indices",
    )
    hard = _validated_indices(hard_indices, n_test=n_test, label="hard_indices")
    review = _validated_indices(review_indices, n_test=n_test, label="review_indices")
    truth = _true_indices(labels)
    missed = truth - hard
    recovered = candidates & missed
    added_fp = candidates - truth
    review_overlap = candidates & review
    review_recovered = recovered & review
    return {
        "candidate_count": len(candidates),
        "recovered_fn": len(recovered),
        "added_fp": len(added_fp),
        "candidate_precision": (
            len(recovered) / len(candidates) if candidates else 0.0
        ),
        "review_overlap_count": len(review_overlap),
        "review_recovered_fn": len(review_recovered),
        "novel_recovered_fn": len(recovered - review),
    }
