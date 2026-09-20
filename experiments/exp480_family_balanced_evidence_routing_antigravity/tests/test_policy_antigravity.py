"""Task 3 tests for family-balanced candidate-wise routing."""

from __future__ import annotations

from experiments.exp480_family_balanced_evidence_routing_antigravity.config_antigravity import (
    FEATURE_KEYS_ANTIGRAVITY,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.policy_antigravity import (
    FamilyBalancedCandidatePolicyAntigravity,
)


def _row_antigravity(
    dataset: str,
    offset: float,
    *,
    multi_gain: tuple[float, int] = (0.0, 0),
    hydra_gain: tuple[float, int] = (0.0, 0),
    hybrid_gain: tuple[float, int] = (0.0, 0),
) -> EvidenceRowAntigravity:
    baseline_auc = 0.70
    baseline_zf = 2
    return EvidenceRowAntigravity(
        dataset=dataset,
        source_family=dataset.split("_normal_", 1)[0],
        proposal="minirocket_local_gap",
        features=(offset, 10.0, 0.5, 0.0, 3.0, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", baseline_auc, baseline_zf),
            CandidateMetricAntigravity(
                "multirocket_local_gap", baseline_auc + multi_gain[0], baseline_zf + multi_gain[1]
            ),
            CandidateMetricAntigravity(
                "hydra_local_gap", baseline_auc + hydra_gain[0], baseline_zf + hydra_gain[1]
            ),
            CandidateMetricAntigravity(
                "hydra_multirocket_local_gap",
                baseline_auc + hybrid_gain[0],
                baseline_zf + hybrid_gain[1],
            ),
        ),
    )


def _query_antigravity() -> dict[str, float]:
    return dict(
        zip(
            FEATURE_KEYS_ANTIGRAVITY,
            (0.0, 10.0, 0.5, 0.0, 3.0, 1.0, 0.2),
            strict=True,
        )
    )


def test_each_family_contributes_only_one_nearest_row_antigravity() -> None:
    """Repeating one source family must never create extra KNN votes."""
    rows = tuple(
        _row_antigravity(f"A_normal_{index}", index / 100, hydra_gain=(0.1, 1))
        for index in range(8)
    ) + (
        _row_antigravity("B_normal_1", 0.2, hydra_gain=(0.1, 1)),
        _row_antigravity("C_normal_1", 0.3, hydra_gain=(0.1, 1)),
    )
    decision = FamilyBalancedCandidatePolicyAntigravity(
        k=3, min_support=3, positive_rate_threshold=1.0
    ).fit(rows).route(_query_antigravity())

    assert decision.method == "hydra_local_gap"
    assert decision.neighbour_families == ("A", "B", "C")
    assert decision.support == 3


def test_uncertain_evidence_returns_exact_mini_antigravity() -> None:
    """One losing family among two must trigger the conservative fallback."""
    rows = (
        _row_antigravity("A_normal_1", 0.0, hydra_gain=(0.1, 1)),
        _row_antigravity("B_normal_1", 0.1, hydra_gain=(-0.1, -1)),
    )
    decision = FamilyBalancedCandidatePolicyAntigravity(
        k=2, min_support=2, positive_rate_threshold=0.90
    ).fit(rows).route(_query_antigravity())

    assert decision.method == "minirocket_local_gap"
    assert decision.accepted is False
    assert decision.reason == "candidate_gates_failed"


def test_candidate_is_evaluated_even_when_old_proposal_is_mini_antigravity() -> None:
    """Candidate-wise evidence must not inherit the old proposal restriction."""
    rows = tuple(
        _row_antigravity(f"F{index}_normal_1", index / 10, multi_gain=(0.05, 2))
        for index in range(4)
    )
    decision = FamilyBalancedCandidatePolicyAntigravity(
        k=4, min_support=4, positive_rate_threshold=1.0
    ).fit(rows).route(_query_antigravity())

    assert decision.method == "multirocket_local_gap"
    assert decision.accepted is True
    assert decision.mean_zf_gain == 2.0


def test_equal_candidates_follow_the_frozen_method_order_antigravity() -> None:
    """Changing input ordering must not change an exact performance tie."""
    rows = tuple(
        _row_antigravity(
            f"F{index}_normal_1", index / 10,
            multi_gain=(0.05, 1), hydra_gain=(0.05, 1), hybrid_gain=(0.05, 1),
        )
        for index in range(4)
    )
    policy = FamilyBalancedCandidatePolicyAntigravity(
        k=4, min_support=4, positive_rate_threshold=1.0
    ).fit(reversed(rows))

    first = policy.route(_query_antigravity())
    second = policy.route(_query_antigravity())
    assert first == second
    assert first.method == "multirocket_local_gap"


def test_nonfinite_query_value_uses_the_training_median_antigravity() -> None:
    """A missing numeric feature must not silently change the nearest families."""
    rows = (
        _row_antigravity("C_normal_1", 0.3, hydra_gain=(-0.1, -1)),
        _row_antigravity("D_normal_1", 0.4, hydra_gain=(-0.1, -1)),
        _row_antigravity("A_normal_1", 0.0, hydra_gain=(0.1, 1)),
        _row_antigravity("B_normal_1", 0.1, hydra_gain=(0.1, 1)),
    )
    policy = FamilyBalancedCandidatePolicyAntigravity(
        k=2, min_support=2, positive_rate_threshold=1.0
    ).fit(rows)
    finite = policy.route(_query_antigravity())
    missing = _query_antigravity()
    missing["skewness"] = float("nan")

    assert policy.route(missing) == finite
    assert finite.neighbour_families == ("A", "B")
