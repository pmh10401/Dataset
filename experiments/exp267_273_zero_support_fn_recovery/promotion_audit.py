"""Exp273 hypothetical promotion gate; never mutates autonomous predictions."""

from __future__ import annotations

import math

from .config import HARD_RISK_LIMIT, MIN_RETROSPECTIVE_PRECISION
from .contracts import PromotionAudit


def audit_promotion(
    *,
    independent_groups: int,
    train_u95: float,
    precision: float,
    overall_precision: float,
    invariant: bool,
) -> PromotionAudit:
    """Apply frozen gates to a report-only future-policy candidate."""
    if type(independent_groups) is not int or independent_groups < 0:
        raise ValueError("independent_groups must be a non-negative integer")
    values = (float(train_u95), float(precision), float(overall_precision))
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("promotion rates must be finite values in [0, 1]")
    eligible = bool(
        independent_groups >= 2
        and values[0] <= HARD_RISK_LIMIT
        and values[1] >= MIN_RETROSPECTIVE_PRECISION
        and values[2] >= MIN_RETROSPECTIVE_PRECISION
        and invariant
    )
    return PromotionAudit(
        independent_groups=independent_groups,
        train_u95=values[0],
        precision=values[1],
        overall_precision=values[2],
        invariant=bool(invariant),
        eligible_for_future_experiment=eligible,
    )


__all__ = ["audit_promotion"]
