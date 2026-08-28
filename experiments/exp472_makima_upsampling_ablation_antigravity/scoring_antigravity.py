"""Thin, label-free condition-E scoring over Exp450's isolated aligner boundary."""

from __future__ import annotations

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.routing_paper import (
    build_hard_evidence_surface,
    build_policies,
)
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import (
    score_low_frequency,
)
from paper.equipment_anomaly_detection.sources_paper.stability_paper import (
    classify_stability,
)

from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450DatasetConditionResult,
)
from experiments.exp450_resampling_image_ablation_antigravity.scoring_antigravity import (
    score_dataset_condition_with_alignment_exp450_antigravity,
)

from .config_antigravity import CONDITION_E_ANTIGRAVITY
from .contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
    Exp472ScoreShardAntigravity,
)
from .transforms_antigravity import (
    align_rows_exp472_antigravity,
    Exp472AlignmentTraceAntigravity,
    test_identity_exp472_antigravity,
)


def score_dataset_condition_e_exp472_antigravity(
    values: RawDataset,
    backend: Exp472BackendDecisionAntigravity,
    *,
    alignment_trace: Exp472AlignmentTraceAntigravity | None = None,
) -> Exp450DatasetConditionResult:
    """Score one raw dataset with E's Makima-only alignment substitution."""
    try:
        result = score_dataset_condition_with_alignment_exp450_antigravity(
            values,
            CONDITION_E_ANTIGRAVITY,
            backend,
            align_rows_override=align_rows_exp472_antigravity,
            alignment_observer=None if alignment_trace is None else alignment_trace.observe,
            alignment_plan_callback=None if alignment_trace is None else alignment_trace.begin_scoring,
        )
    except Exception:
        if alignment_trace is not None:
            alignment_trace.poison()
        raise
    if alignment_trace is not None:
        alignment_trace.mark_score_completed()
    return result


def compose_final_score_exp472_antigravity(
    values: RawDataset,
    result: Exp450DatasetConditionResult,
    backend: Exp472BackendDecisionAntigravity,
    *,
    alignment_trace: Exp472AlignmentTraceAntigravity | None = None,
) -> Exp472ScoreShardAntigravity:
    """Apply the frozen hard-plus-stable policy to one label-free E result."""
    del backend
    if alignment_trace is None:
        raise ValueError("Exp472 composition requires one sealed alignment trace")
    diagnostic = alignment_trace.sealed_diagnostic()
    if diagnostic["dataset"] != values.dataset:
        raise ValueError("Exp472 alignment trace dataset differs")
    raw = {values.dataset: values}
    sources = {values.dataset: result.sources}
    routes = {values.dataset: result.route}
    hard = {values.dataset: result.route.exp253.hard}
    low = score_low_frequency(raw, hard)
    stability = classify_stability(low, raw)
    surface = build_hard_evidence_surface(sources, routes, raw)
    policies = build_policies(surface, low, stability, raw, routes)
    policy = next(
        item for item in policies.policies if item.name == "hard_plus_stable"
    )
    scores = np.asarray(policy.scores[values.dataset], dtype=np.float64)
    return Exp472ScoreShardAntigravity(
        dataset=values.dataset,
        test_identity=test_identity_exp472_antigravity(values),
        scores=scores,
        alerts=policy.alerts[values.dataset],
        upsampled_rows=int(diagnostic["upsampled_rows"]),
        diagnostic=diagnostic,
    )


__all__ = [
    "compose_final_score_exp472_antigravity",
    "score_dataset_condition_e_exp472_antigravity",
]
