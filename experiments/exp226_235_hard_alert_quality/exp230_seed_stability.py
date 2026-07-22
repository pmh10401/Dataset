from __future__ import annotations

from collections.abc import Mapping, Set


def stability_support(
    candidates_by_repetition: Mapping[str, Set[int]],
) -> dict[int, int]:
    if not candidates_by_repetition:
        raise ValueError("at least one frozen repetition is required")
    support: dict[int, int] = {}
    for name in sorted(candidates_by_repetition):
        for index in set(int(value) for value in candidates_by_repetition[name]):
            support[index] = support.get(index, 0) + 1
    return dict(sorted(support.items()))

