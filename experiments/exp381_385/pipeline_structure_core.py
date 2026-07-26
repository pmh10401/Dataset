from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PipelineLanes:
    baseline_hard: tuple[int, ...]
    review_additions: tuple[int, ...]
    review_union: tuple[int, ...]
    hard_expansion: tuple[int, ...]
    baseline_confirmed: tuple[int, ...]
    baseline_unconfirmed: tuple[int, ...]
    replacement: tuple[int, ...]


def build_pipeline_lanes(
    *,
    rule_indices: Sequence[int],
    baseline_hard_indices: Sequence[int],
    exact_normal_indices: Sequence[int],
    existing_review_indices: Sequence[int],
) -> PipelineLanes:
    rule = {int(index) for index in rule_indices}
    baseline = {int(index) for index in baseline_hard_indices}
    exact_normal = {int(index) for index in exact_normal_indices}
    existing_review = {int(index) for index in existing_review_indices}
    guarded_rule = rule - exact_normal
    additions = guarded_rule - baseline
    review_union = (existing_review | additions) - baseline - exact_normal
    confirmed = baseline & guarded_rule
    return PipelineLanes(
        baseline_hard=tuple(sorted(baseline)),
        review_additions=tuple(sorted(additions)),
        review_union=tuple(sorted(review_union)),
        hard_expansion=tuple(sorted(baseline | additions)),
        baseline_confirmed=tuple(sorted(confirmed)),
        baseline_unconfirmed=tuple(sorted(baseline - confirmed)),
        replacement=tuple(sorted(guarded_rule)),
    )
