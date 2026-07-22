from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from experiments.exp317_318_train_reliability_fusion.reliability import (
    deterministic_reference_audit_splits,
)

from .config import SEED, SPLITS


def calibration_stability(train_scores, test_score: float, alpha: float) -> float:
    scores = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    if len(scores) < 2 or not np.isfinite(scores).all() or not np.isfinite(test_score):
        return 0.0
    decisions = []
    for reference_indices, _ in deterministic_reference_audit_splits(
        len(scores), seed=SEED, splits=SPLITS,
    ):
        reference = scores[list(reference_indices)]
        pvalue = (1 + np.count_nonzero(reference >= float(test_score))) / (len(reference) + 1)
        decisions.append(float(pvalue <= float(alpha)))
    return float(np.mean(decisions))


def robust_margin_strength(train_scores, test_score: float, alpha: float) -> float:
    scores = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    if not len(scores) or not np.isfinite(scores).all() or not np.isfinite(test_score):
        return 0.0
    threshold = float(np.quantile(scores, 1.0 - float(alpha), method="higher"))
    q25, q75 = np.quantile(scores, [0.25, 0.75])
    scale = max(float(q75 - q25), float(np.std(scores)), 1e-12)
    excess = max(0.0, (float(test_score) - threshold) / scale)
    return float(np.tanh(excess))


@dataclass(frozen=True)
class CandidateConfidence:
    available_groups: int
    supporting_groups: int
    available_weight: float
    supporting_evidence: float
    confidence: float
    supporting_group_names: tuple[str, ...]


def candidate_confidence(
    weights: Mapping[str, float], source_strengths: Mapping[str, float], *,
    groups: Mapping[str, str],
) -> CandidateConfidence:
    available_by_group: dict[str, float] = {}
    evidence_by_group: dict[str, float] = {}
    for source, raw_weight in weights.items():
        weight = max(0.0, float(raw_weight))
        if weight <= 0.0:
            continue
        group = str(groups[source])
        available_by_group[group] = max(available_by_group.get(group, 0.0), weight)
        strength = min(1.0, max(0.0, float(source_strengths.get(source, 0.0))))
        if strength > 0.0:
            evidence_by_group[group] = max(evidence_by_group.get(group, 0.0), weight * strength)
    available_weight = float(sum(available_by_group.values()))
    evidence = float(sum(evidence_by_group.values()))
    return CandidateConfidence(
        available_groups=len(available_by_group),
        supporting_groups=len(evidence_by_group),
        available_weight=available_weight,
        supporting_evidence=evidence,
        confidence=evidence / available_weight if available_weight else 0.0,
        supporting_group_names=tuple(sorted(evidence_by_group)),
    )
