from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np


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


def bootstrap_intervals(
    recovered_fn: np.ndarray,
    added_fp: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, np.ndarray]:
    recovered = np.asarray(recovered_fn, dtype=np.float64)
    false_positives = np.asarray(added_fp, dtype=np.float64)
    if recovered.ndim != 2 or false_positives.shape != recovered.shape:
        raise ValueError("bootstrap metric arrays must have the same 2D shape")
    if recovered.shape[1] == 0:
        raise ValueError("bootstrap requires at least one dataset")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        recovered.shape[1],
        np.full(recovered.shape[1], 1.0 / recovered.shape[1]),
        size=iterations,
    ).T
    sampled_tp = recovered @ weights
    sampled_fp = false_positives @ weights
    sampled_candidates = sampled_tp + sampled_fp
    sampled_precision = np.divide(
        sampled_tp,
        sampled_candidates,
        out=np.zeros_like(sampled_tp),
        where=sampled_candidates > 0,
    )

    def interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low, high = np.quantile(values, (0.025, 0.975), axis=1)
        return low, high

    recovered_low, recovered_high = interval(sampled_tp)
    fp_low, fp_high = interval(sampled_fp)
    precision_low, precision_high = interval(sampled_precision)
    return {
        "recovered_fn_ci_low": recovered_low,
        "recovered_fn_ci_high": recovered_high,
        "added_fp_ci_low": fp_low,
        "added_fp_ci_high": fp_high,
        "precision_ci_low": precision_low,
        "precision_ci_high": precision_high,
    }


def concentration_diagnostics(
    recovered_fn: np.ndarray,
    *,
    lineages: Sequence[str],
) -> dict[str, np.ndarray]:
    recovered = np.asarray(recovered_fn, dtype=np.int64)
    if recovered.ndim != 2 or recovered.shape[1] != len(lineages):
        raise ValueError("lineage count must match recovery matrix datasets")
    unique_lineages = tuple(sorted(set(lineages)))
    membership = np.zeros(
        (len(lineages), len(unique_lineages)),
        dtype=np.int64,
    )
    lineage_index = {name: index for index, name in enumerate(unique_lineages)}
    for dataset_index, lineage in enumerate(lineages):
        membership[dataset_index, lineage_index[lineage]] = 1
    lineage_tp = recovered @ membership
    total_tp = lineage_tp.sum(axis=1)
    top_tp = lineage_tp.max(axis=1) if lineage_tp.shape[1] else total_tp
    top_share = np.divide(
        top_tp,
        total_tp,
        out=np.zeros_like(total_tp, dtype=np.float64),
        where=total_tp > 0,
    )
    contributing = np.count_nonzero(lineage_tp, axis=1)
    leave_one_out = total_tp[:, None] - lineage_tp
    return {
        "tp_contributing_lineages": contributing,
        "top_lineage_tp_share": top_share,
        "concentrated_signal": (total_tp > 0) & (top_share >= 0.8),
        "leave_one_lineage_out_min_recovered_fn": (
            leave_one_out.min(axis=1)
            if leave_one_out.shape[1]
            else np.zeros(recovered.shape[0], dtype=np.int64)
        ),
        "leave_one_lineage_out_max_recovered_fn": (
            leave_one_out.max(axis=1)
            if leave_one_out.shape[1]
            else np.zeros(recovered.shape[0], dtype=np.int64)
        ),
    }


def assign_research_label(
    *,
    recovered_fn: int,
    candidate_precision: float,
    tp_contributing_lineages: int,
    top_lineage_tp_share: float,
    precision_ci_low: float,
) -> str:
    if recovered_fn == 0 or candidate_precision < 0.30:
        return "unsupported"
    if (
        recovered_fn >= 10
        and candidate_precision >= 0.50
        and tp_contributing_lineages >= 5
        and top_lineage_tp_share < 0.80
        and precision_ci_low >= 0.35
    ):
        return "preregistration_candidate"
    return "research_signal"
