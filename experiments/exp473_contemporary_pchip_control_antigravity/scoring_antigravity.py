"""Label-free Exp473 same-environment PCHIP control scoring boundary."""

from __future__ import annotations

from collections.abc import Sequence

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

from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import Exp450DatasetConditionResult
from experiments.exp450_resampling_image_ablation_antigravity.scoring_antigravity import (
    score_dataset_condition_with_alignment_exp450_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.transforms_antigravity import (
    align_rows_exp450_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import (
    test_identity_exp472_antigravity,
)

from .config_antigravity import CONDITION_D_PRIME_EXP473_ANTIGRAVITY
from .contracts_antigravity import Exp473ScoreShardAntigravity



def align_rows_d_prime_exp473_antigravity(
    rows: Sequence[np.ndarray], target_length: int
) -> np.ndarray:
    """Adapt Exp450's frozen PCHIP aligner to its isolated override boundary."""
    return align_rows_exp450_antigravity(
        rows, target_length, mode="polyphase_pchip"
    )


def score_dataset_condition_d_prime_exp473_antigravity(
    values: RawDataset,
    backend: Exp472BackendDecisionAntigravity,
) -> Exp450DatasetConditionResult:
    """Recompute condition D with the current process's canonical PCHIP path."""
    return score_dataset_condition_with_alignment_exp450_antigravity(
        values,
        CONDITION_D_PRIME_EXP473_ANTIGRAVITY.as_exp450_condition_antigravity(),
        backend,
        align_rows_override=align_rows_d_prime_exp473_antigravity,
    )


def compose_final_score_exp473_antigravity(
    values: RawDataset, result: Exp450DatasetConditionResult
) -> Exp473ScoreShardAntigravity:
    """Apply the frozen hard-plus-stable policy to one current D result."""
    if result.dataset != values.dataset or result.condition_id != "D":
        raise ValueError("Exp473 condition-D result identity differs")
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
    return Exp473ScoreShardAntigravity(
        dataset=values.dataset,
        test_identity=test_identity_exp472_antigravity(values),
        scores=np.asarray(policy.scores[values.dataset], dtype=np.float64),
        alerts=policy.alerts[values.dataset],
    )


__all__ = [
    "CONDITION_D_PRIME_EXP473_ANTIGRAVITY",
    "align_rows_d_prime_exp473_antigravity",
    "compose_final_score_exp473_antigravity",
    "score_dataset_condition_d_prime_exp473_antigravity",
]
