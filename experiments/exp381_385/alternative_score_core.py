from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


ALTERNATIVE_SOURCES = (
    "spectral_derivative",
    "shapelet_normal_prototype",
    "interval_quantile",
)


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    alpha: float
    n_train: int
    threshold: float | None
    threshold_rank: int | None
    distinct_train_scores: int
    indices: tuple[int, ...]
    test_tail_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    sources: tuple[str, ...]
    k: int


@dataclass(frozen=True)
class RuleResult:
    indices: tuple[int, ...]
    available_source_count: int
    selected_source_count: int
    fully_observable: bool
    resolution_blocked: bool
    unavailable_sources: tuple[str, ...]


def calibrate_local_empirical(
    train_scores: Sequence[float] | np.ndarray,
    test_scores: Sequence[float] | np.ndarray,
    alpha: float,
) -> CalibrationResult:
    train = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    test = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("score arrays must be finite")

    n_train = int(train.size)
    distinct = int(np.unique(train).size)
    if n_train < 5:
        return CalibrationResult(
            status="calibration_abstained",
            alpha=float(alpha),
            n_train=n_train,
            threshold=None,
            threshold_rank=None,
            distinct_train_scores=distinct,
            indices=(),
            test_tail_probabilities=(),
        )

    maximum_train_exceedances = (
        math.floor(float(alpha) * (n_train + 1) + 1e-12) - 1
    )
    if maximum_train_exceedances < 0:
        return CalibrationResult(
            status="resolution_blocked",
            alpha=float(alpha),
            n_train=n_train,
            threshold=None,
            threshold_rank=None,
            distinct_train_scores=distinct,
            indices=(),
            test_tail_probabilities=(),
        )

    descending = np.sort(train)[::-1]
    threshold_rank = min(maximum_train_exceedances, n_train - 1)
    threshold = float(descending[threshold_rank])
    tail_probabilities = tuple(
        float((1 + np.count_nonzero(train >= score)) / (n_train + 1))
        for score in test
    )
    indices = tuple(
        index
        for index, (score, probability) in enumerate(
            zip(test, tail_probabilities, strict=True)
        )
        if score > threshold
        and (
            probability < float(alpha)
            or math.isclose(probability, float(alpha), abs_tol=1e-12)
        )
    )
    return CalibrationResult(
        status="attainable",
        alpha=float(alpha),
        n_train=n_train,
        threshold=threshold,
        threshold_rank=threshold_rank,
        distinct_train_scores=distinct,
        indices=indices,
        test_tail_probabilities=tail_probabilities,
    )


def generate_rule_specs(
    sources: Sequence[str] = ALTERNATIVE_SOURCES,
) -> tuple[RuleSpec, ...]:
    ordered = tuple(sources)
    rules: list[RuleSpec] = []
    for size in range(1, len(ordered) + 1):
        for subset in combinations(ordered, size):
            for k in range(1, size + 1):
                rules.append(
                    RuleSpec(
                        rule_id=f"{'_'.join(subset)}__{k}of{size}",
                        sources=subset,
                        k=k,
                    )
                )
    return tuple(rules)


def evaluate_rule(
    rule: RuleSpec,
    *,
    source_indices: Mapping[str, Sequence[int]],
    source_available: Mapping[str, bool],
) -> RuleResult:
    unavailable = tuple(
        source for source in rule.sources if not source_available.get(source, False)
    )
    available_count = len(rule.sources) - len(unavailable)
    blocked = available_count < rule.k
    votes: Counter[int] = Counter()
    if not blocked:
        for source in rule.sources:
            if source_available.get(source, False):
                votes.update({int(index) for index in source_indices.get(source, ())})
    return RuleResult(
        indices=tuple(sorted(index for index, count in votes.items() if count >= rule.k)),
        available_source_count=available_count,
        selected_source_count=len(rule.sources),
        fully_observable=not unavailable,
        resolution_blocked=blocked,
        unavailable_sources=unavailable,
    )
