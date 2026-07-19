from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConfidenceLanes:
    rocket_stable: frozenset[int]
    independent_auxiliary: frozenset[int]
    confidence_a: frozenset[int]
    confidence_b: frozenset[int]
    confidence_c: frozenset[int]
    review_priority_1: frozenset[int]
    review_priority_2: frozenset[int]
    review_priority_3: frozenset[int]
    priority_unchanged: frozenset[int]


def _frozen(values: Iterable[int]) -> frozenset[int]:
    return frozenset(int(value) for value in values)


def build_confidence_lanes(
    *, hard: Iterable[int], standard: Iterable[int], priority: Iterable[int],
    rocket_256: Iterable[int], rocket_512: Iterable[int], rocket_1024: Iterable[int],
    rocket_256_seed2: Iterable[int], independent_auxiliary: Iterable[int],
) -> ConfidenceLanes:
    hard_set, standard_set, priority_set = _frozen(hard), _frozen(standard), _frozen(priority)
    stable = _frozen(rocket_256) & _frozen(rocket_512) & _frozen(rocket_1024) & _frozen(rocket_256_seed2)
    independent = _frozen(independent_auxiliary)
    both = stable & independent
    either = stable | independent
    confidence_a = hard_set & both
    confidence_b = (hard_set & either) - confidence_a
    confidence_c = hard_set - confidence_a - confidence_b
    review_1 = standard_set & both
    review_2 = (standard_set & either) - review_1
    review_3 = standard_set - review_1 - review_2
    if confidence_a | confidence_b | confidence_c != hard_set:
        raise AssertionError("Exp238 confidence partition changed frozen Hard lane")
    if review_1 | review_2 | review_3 != standard_set:
        raise AssertionError("Exp238 review partition changed frozen Standard lane")
    return ConfidenceLanes(
        rocket_stable=stable,
        independent_auxiliary=independent,
        confidence_a=confidence_a,
        confidence_b=confidence_b,
        confidence_c=confidence_c,
        review_priority_1=review_1,
        review_priority_2=review_2,
        review_priority_3=review_3,
        priority_unchanged=priority_set,
    )
