from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.3f}".replace(".", "p")


def validated_indices(
    values: Iterable[int],
    *,
    n_test: int,
    label: str,
) -> set[int]:
    indices = list(values)
    if any(type(index) is not int or index < 0 or index >= n_test for index in indices):
        raise ValueError(f"{label} contains out-of-range indices")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{label} contains duplicate indices")
    return set(indices)


@dataclass(frozen=True, order=True)
class CombinationRule:
    operator: str
    sources: tuple[str, ...]
    alpha: float
    minimum_votes: int

    def __post_init__(self) -> None:
        if self.operator not in {"intersection", "union", "atleast2"}:
            raise ValueError(f"unsupported combination operator: {self.operator}")
        if tuple(sorted(self.sources)) != self.sources:
            raise ValueError("combination sources must be sorted")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("combination sources must be unique")
        if len(self.sources) not in (2, 3):
            raise ValueError("combination rules require two or three sources")
        if self.minimum_votes < 1 or self.minimum_votes > len(self.sources):
            raise ValueError("invalid minimum vote count")

    @property
    def rule_id(self) -> str:
        joined = "__".join(self.sources)
        return f"{self.operator}__{joined}__alpha_{alpha_slug(self.alpha)}"


def enumerate_combination_rules(
    sources: Sequence[str],
    *,
    alpha: float,
) -> tuple[CombinationRule, ...]:
    ordered = tuple(sorted(sources))
    if len(ordered) != len(set(ordered)):
        raise ValueError("source names must be unique")
    rules: list[CombinationRule] = []
    for size in (2, 3):
        for group in combinations(ordered, size):
            rules.append(CombinationRule("intersection", group, alpha, size))
            rules.append(CombinationRule("union", group, alpha, 1))
            if size == 3:
                rules.append(CombinationRule("atleast2", group, alpha, 2))
    return tuple(sorted(rules, key=lambda item: item.rule_id))


def candidate_indices_for_rule(
    rule: CombinationRule,
    source_indices: Mapping[str, Iterable[int]],
    *,
    excluded: Iterable[int],
    n_test: int,
) -> set[int]:
    excluded_set = validated_indices(
        excluded,
        n_test=n_test,
        label="excluded indices",
    )
    votes: Counter[int] = Counter()
    for source in rule.sources:
        if source not in source_indices:
            raise ValueError(f"missing source for rule: {source}")
        values = validated_indices(
            source_indices[source],
            n_test=n_test,
            label=f"{source} source",
        )
        votes.update(values - excluded_set)
    return {
        index
        for index, count in votes.items()
        if count >= rule.minimum_votes
    }


def evaluate_candidate_set(
    *,
    candidate_indices: Iterable[int],
    hard_indices: Iterable[int],
    review_indices: Iterable[int],
    labels: Sequence[int | str],
) -> dict[str, int | float]:
    n_test = len(labels)
    candidates = validated_indices(
        candidate_indices,
        n_test=n_test,
        label="candidate indices",
    )
    hard = validated_indices(hard_indices, n_test=n_test, label="Hard indices")
    review = validated_indices(
        review_indices,
        n_test=n_test,
        label="review indices",
    )
    truth = {index for index, label in enumerate(labels) if str(label) == "1"}
    missed = truth - hard
    recovered = candidates & missed
    added_fp = candidates - truth
    review_overlap = candidates & review
    review_recovered = recovered & review
    review_overlap_fp = review_overlap - truth
    return {
        "baseline_fn": len(missed),
        "candidate_count": len(candidates),
        "recovered_fn": len(recovered),
        "added_fp": len(added_fp),
        "candidate_precision": (
            len(recovered) / len(candidates) if candidates else 0.0
        ),
        "fn_recovery_rate": len(recovered) / len(missed) if missed else 0.0,
        "review_overlap_count": len(review_overlap),
        "review_recovered_fn": len(review_recovered),
        "review_overlap_fp": len(review_overlap_fp),
        "review_overlap_precision": (
            len(review_recovered) / len(review_overlap) if review_overlap else 0.0
        ),
        "novel_recovered_fn": len(recovered - review),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("p-values must be between zero and one")
    count = len(p_values)
    if count == 0:
        return []
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted_sorted = [0.0] * count
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        original_index, value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, value * count / rank)
        adjusted_sorted[reverse_index] = min(1.0, running)
    adjusted = [0.0] * count
    for sorted_index, (original_index, _) in enumerate(ordered):
        adjusted[original_index] = adjusted_sorted[sorted_index]
    return adjusted
