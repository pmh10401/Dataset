from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ConfirmationTiers:
    dual: frozenset[int]
    single: frozenset[int]
    unconfirmed: frozenset[int]


def confirmation_tiers(
    hard_indices: Iterable[int],
    temporal_hits: Iterable[int],
    canonical_hits: Iterable[int],
) -> ConfirmationTiers:
    hard = frozenset(int(value) for value in hard_indices)
    temporal = frozenset(int(value) for value in temporal_hits)
    canonical = frozenset(int(value) for value in canonical_hits)
    dual = hard & temporal & canonical
    single = hard & (temporal ^ canonical)
    unconfirmed = hard - dual - single
    if dual | single | unconfirmed != hard:
        raise AssertionError("confirmation tiers do not partition Exp137 Hard")
    return ConfirmationTiers(dual=dual, single=single, unconfirmed=unconfirmed)

