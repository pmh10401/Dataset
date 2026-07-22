from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MIN_TRAIN_SCORES, RELIABILITY_SPLITS, WILSON_Z


@dataclass(frozen=True)
class ReliabilityAudit:
    available: bool
    reason: str
    split_false_alert_counts: tuple[int, ...] = ()
    split_audit_counts: tuple[int, ...] = ()
    split_wilson_upper: tuple[float, ...] = ()
    max_wilson_upper: float | None = None
    n_audit_min: int = 0
    weight: float = 0.0


def deterministic_reference_audit_splits(
    count: int, *, seed: int, splits: int = RELIABILITY_SPLITS,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if int(count) < 2:
        raise ValueError("reference/audit splits require at least two scores")
    output = []
    for split_index in range(int(splits)):
        indices = np.arange(int(count), dtype=int)
        np.random.default_rng(int(seed) + split_index).shuffle(indices)
        reference_count = len(indices) // 2
        reference = tuple(sorted(indices[:reference_count].tolist()))
        audit = tuple(sorted(indices[reference_count:].tolist()))
        output.append((reference, audit))
    return tuple(output)


def empirical_upper_pvalues(reference, audit) -> np.ndarray:
    reference_values = np.asarray(reference, dtype=np.float64).reshape(-1)
    audit_values = np.asarray(audit, dtype=np.float64).reshape(-1)
    if not len(reference_values):
        raise ValueError("empirical p-values require a non-empty reference")
    return np.asarray([
        (1 + np.count_nonzero(reference_values >= value)) / (len(reference_values) + 1)
        for value in audit_values
    ], dtype=np.float64)


def wilson_upper_one_sided(
    false_alerts: int, total: int, *, z: float = WILSON_Z,
) -> float:
    if int(total) <= 0:
        raise ValueError("Wilson bound requires a positive total")
    if int(false_alerts) < 0 or int(false_alerts) > int(total):
        raise ValueError("false_alerts must be between zero and total")
    probability = int(false_alerts) / int(total)
    denominator = 1.0 + float(z) ** 2 / int(total)
    center = probability + float(z) ** 2 / (2.0 * int(total))
    radius = float(z) * np.sqrt(
        probability * (1.0 - probability) / int(total)
        + float(z) ** 2 / (4.0 * int(total) ** 2)
    )
    return float(min(1.0, (center + radius) / denominator))


def reliability_weight(max_wilson_upper: float, *, n_audit_min: int) -> float:
    if int(n_audit_min) <= 0:
        return 0.0
    risk = max(float(max_wilson_upper), 1.0 / (int(n_audit_min) + 1))
    return float(max(0.0, -np.log10(risk)))


def audit_source_reliability(
    train_scores, *, alpha: float, seed: int,
    splits: int = RELIABILITY_SPLITS, minimum_scores: int = MIN_TRAIN_SCORES,
) -> ReliabilityAudit:
    scores = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    if len(scores) < int(minimum_scores):
        return ReliabilityAudit(False, "insufficient_train_reference")
    if not np.isfinite(scores).all():
        return ReliabilityAudit(False, "non_finite_train_scores")
    false_alert_counts = []
    audit_counts = []
    upper_bounds = []
    for reference_indices, audit_indices in deterministic_reference_audit_splits(
        len(scores), seed=int(seed), splits=int(splits),
    ):
        pvalues = empirical_upper_pvalues(
            scores[list(reference_indices)], scores[list(audit_indices)],
        )
        false_alerts = int(np.count_nonzero(pvalues <= float(alpha)))
        audit_count = len(audit_indices)
        false_alert_counts.append(false_alerts)
        audit_counts.append(audit_count)
        upper_bounds.append(wilson_upper_one_sided(false_alerts, audit_count))
    maximum = max(upper_bounds)
    minimum_audit = min(audit_counts)
    return ReliabilityAudit(
        True, "",
        tuple(false_alert_counts), tuple(audit_counts), tuple(upper_bounds),
        float(maximum), int(minimum_audit),
        reliability_weight(maximum, n_audit_min=minimum_audit),
    )
