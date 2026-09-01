"""Task 2 tests for the Exp481 strict residual-correction policy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.config_antigravity import (
    FEATURE_KEYS_ANTIGRAVITY,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.contracts_antigravity import (
    InnerCandidateGateAntigravity,
    ReferenceStatisticsAntigravity,
    ResidualDecisionAntigravity,
    StrictCandidateAssessmentAntigravity,
)
from experiments.exp481_strict_evidence_residual_correction_antigravity.policy_antigravity import (
    StrictResidualPolicyAntigravity,
    nearest_distinct_family_indices_reference_antigravity,
    strict_statistics_reference_antigravity,
)


def _row_antigravity(
    dataset: str,
    offset: float,
    *,
    proposal: str = "minirocket_local_gap",
    multi_gain: tuple[float, int] = (0.0, 0),
    hydra_gain: tuple[float, int] = (0.0, 0),
    hybrid_gain: tuple[float, int] = (0.0, 0),
) -> EvidenceRowAntigravity:
    baseline_auc = 0.70
    baseline_zf = 2
    return EvidenceRowAntigravity(
        dataset=dataset,
        source_family=dataset.split("_normal_", 1)[0],
        proposal=proposal,
        features=(offset, 10.0, 0.5, 0.0, 3.0, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", baseline_auc, baseline_zf),
            CandidateMetricAntigravity(
                "multirocket_local_gap",
                baseline_auc + multi_gain[0],
                baseline_zf + multi_gain[1],
            ),
            CandidateMetricAntigravity(
                "hydra_local_gap",
                baseline_auc + hydra_gain[0],
                baseline_zf + hydra_gain[1],
            ),
            CandidateMetricAntigravity(
                "hydra_multirocket_local_gap",
                baseline_auc + hybrid_gain[0],
                baseline_zf + hybrid_gain[1],
            ),
        ),
    )


def query_features_antigravity() -> dict[str, float]:
    return dict(
        zip(
            FEATURE_KEYS_ANTIGRAVITY,
            (0.0, 10.0, 0.5, 0.0, 3.0, 1.0, 0.2),
            strict=True,
        )
    )


def _assessment_antigravity(
    *,
    method: str = "multirocket_local_gap",
    passed: bool = True,
    support: int = 2,
    zf_wins: int = 2,
    zf_losses: int = 0,
    zf_ties: int = 0,
    min_auc_gain: float = 0.1,
    q20_auc_gain: float = 0.1,
    neighbour_families: tuple[str, ...] = ("A", "B"),
) -> StrictCandidateAssessmentAntigravity:
    return StrictCandidateAssessmentAntigravity(
        method=method,
        passed=passed,
        support=support,
        zf_wins=zf_wins,
        zf_losses=zf_losses,
        zf_ties=zf_ties,
        min_auc_gain=min_auc_gain,
        q20_auc_gain=q20_auc_gain,
        neighbour_families=neighbour_families,
    )


def test_tied_zf_is_not_a_strict_win_antigravity() -> None:
    result = strict_statistics_reference_antigravity(
        zf_gains=(1, 0, -1), auc_gains=(0.3, 0.2, 0.1), q=0.20
    )
    assert (result.zf_wins, result.zf_ties, result.zf_losses) == (1, 1, 1)
    assert result.passed is False
    assert result.min_auc_gain == 0.1
    assert result.q20_auc_gain == float(
        np.quantile(np.asarray((0.1, 0.2, 0.3), dtype=np.float64), 0.20)
    )


def test_equal_zf_wins_and_losses_fail_antigravity() -> None:
    result = strict_statistics_reference_antigravity(
        zf_gains=(1, -1), auc_gains=(0.4, 0.2), q=0.20
    )
    assert result.zf_wins == result.zf_losses == 1
    assert result.passed is False


def test_zero_min_auc_may_pass_but_zero_q20_fails_antigravity() -> None:
    passing = strict_statistics_reference_antigravity(
        zf_gains=(1, 1, -1), auc_gains=(0.0, 0.2, 0.4), q=0.20
    )
    assert passing.min_auc_gain == 0.0
    assert passing.q20_auc_gain > 0.0
    assert passing.passed is True

    failing = strict_statistics_reference_antigravity(
        zf_gains=(1, 1, -1), auc_gains=(0.0, 0.0, 0.0), q=0.20
    )
    assert failing.min_auc_gain == 0.0
    assert failing.q20_auc_gain == 0.0
    assert failing.passed is False


def test_nearest_distinct_family_keeps_one_stable_row_antigravity() -> None:
    distances = np.asarray([0.50, 0.10, 0.10, 0.20], dtype=np.float64)
    families = ("B", "A", "A", "C")
    chosen = nearest_distinct_family_indices_reference_antigravity(
        distances, families, k=2
    )
    assert chosen.tolist() == [1, 3]
    assert tuple(families[int(index)] for index in chosen) == ("A", "C")

    chosen_three = nearest_distinct_family_indices_reference_antigravity(
        distances, families, k=3
    )
    assert chosen_three.tolist() == [1, 3, 0]


def test_non_mini_row_knn_is_preserved_antigravity() -> None:
    rows = tuple(
        _row_antigravity(f"F{index}_normal_1", index / 10, multi_gain=(0.1, 1))
        for index in range(3)
    )
    policy = StrictResidualPolicyAntigravity(k=3).fit(rows)
    decision = policy.route(
        query_features_antigravity(), row_knn_method="hydra_local_gap"
    )
    assert decision.method == "hydra_local_gap"
    assert decision.changed is False
    assert decision.reason == "row_knn_alternative_preserved"
    assert decision.selected_candidate is None
    assert decision.assessments == ()


def test_minirocket_is_the_only_residual_eligible_method_antigravity() -> None:
    rows = tuple(
        _row_antigravity(f"F{index}_normal_1", index / 10, multi_gain=(0.1, 1))
        for index in range(3)
    )
    policy = StrictResidualPolicyAntigravity(k=3).fit(rows)
    for method in (
        "multirocket_local_gap",
        "hydra_local_gap",
        "hydra_multirocket_local_gap",
    ):
        decision = policy.route(
            query_features_antigravity(), row_knn_method=method
        )
        assert decision.method == method
        assert decision.changed is False
        assert decision.reason == "row_knn_alternative_preserved"
        assert decision.assessments == ()

    mini = policy.route(
        query_features_antigravity(), row_knn_method="minirocket_local_gap"
    )
    assert len(mini.assessments) == 3
    assert mini.reason != "row_knn_alternative_preserved"


def test_candidate_gains_ignore_row_proposal_antigravity() -> None:
    rows = tuple(
        _row_antigravity(
            f"F{index}_normal_1",
            index / 10,
            proposal="hydra_local_gap",
            multi_gain=(0.10, 2),
            hydra_gain=(0.20, 3),
        )
        for index in range(3)
    )
    decision = (
        StrictResidualPolicyAntigravity(k=3)
        .fit(rows)
        .route(query_features_antigravity(), row_knn_method="minirocket_local_gap")
    )
    by_method = {item.method: item for item in decision.assessments}
    multi = by_method["multirocket_local_gap"]
    assert multi.zf_wins == 3
    assert multi.zf_losses == 0
    assert multi.min_auc_gain == pytest.approx(0.10)
    assert multi.passed is True


def test_only_enabled_methods_may_be_selected_antigravity() -> None:
    rows = tuple(
        _row_antigravity(
            f"F{index}_normal_1",
            index / 10,
            multi_gain=(0.20, 2),
            hydra_gain=(0.10, 1),
        )
        for index in range(3)
    )
    policy = StrictResidualPolicyAntigravity(k=3).fit(rows)
    hydra_only = policy.route(
        query_features_antigravity(),
        row_knn_method="minirocket_local_gap",
        enabled_methods=("hydra_local_gap",),
    )
    assert hydra_only.method == "hydra_local_gap"
    assert hydra_only.changed is True
    assert hydra_only.reason == "strict_candidate_accept"
    assert len(hydra_only.assessments) == 3

    empty = policy.route(
        query_features_antigravity(),
        row_knn_method="minirocket_local_gap",
        enabled_methods=(),
    )
    assert empty.method == "minirocket_local_gap"
    assert empty.changed is False
    assert empty.reason == "inner_gate_disabled"
    assert empty.selected_candidate is None
    assert len(empty.assessments) == 3


def test_equal_candidate_rank_returns_row_knn_antigravity() -> None:
    rows = tuple(
        _row_antigravity(
            f"F{index}_normal_1",
            index / 10,
            multi_gain=(0.10, 1),
            hydra_gain=(0.10, 1),
        )
        for index in range(3)
    )
    decision = (
        StrictResidualPolicyAntigravity(k=3)
        .fit(rows)
        .route(query_features_antigravity(), row_knn_method="minirocket_local_gap")
    )
    assert decision.method == "minirocket_local_gap"
    assert decision.changed is False
    assert decision.reason == "candidate_rank_tie"
    assert decision.selected_candidate is None
    passed = tuple(item for item in decision.assessments if item.passed)
    assert {item.method for item in passed} == {
        "multirocket_local_gap",
        "hydra_local_gap",
    }


def test_nonfinite_query_value_uses_training_median_antigravity() -> None:
    rows = tuple(
        _row_antigravity(
            f"F{index}_normal_1",
            index / 10,
            multi_gain=(0.10, 1) if index == 0 else (0.0, 0),
        )
        for index in range(3)
    )
    policy = StrictResidualPolicyAntigravity(k=3).fit(rows)
    finite = policy.route(
        query_features_antigravity(), row_knn_method="minirocket_local_gap"
    )
    missing = query_features_antigravity()
    missing["skewness"] = float("nan")
    assert policy.route(missing, row_knn_method="minirocket_local_gap") == finite


def test_missing_extra_keys_and_unknown_methods_are_rejected_antigravity() -> None:
    rows = tuple(
        _row_antigravity(f"F{index}_normal_1", index / 10) for index in range(3)
    )
    policy = StrictResidualPolicyAntigravity(k=3).fit(rows)
    query = query_features_antigravity()
    missing = dict(query)
    missing.pop("width")
    extra = dict(query)
    extra["dataset"] = 1.0
    with pytest.raises(ValueError):
        policy.route(missing, row_knn_method="minirocket_local_gap")
    with pytest.raises(ValueError):
        policy.route(extra, row_knn_method="minirocket_local_gap")
    with pytest.raises(ValueError):
        policy.route(query, row_knn_method="not_a_method")
    with pytest.raises(ValueError):
        policy.route(
            query,
            row_knn_method="minirocket_local_gap",
            enabled_methods=("not_a_method",),
        )
    with pytest.raises(RuntimeError):
        StrictResidualPolicyAntigravity(k=3).route(
            query, row_knn_method="minirocket_local_gap"
        )


def test_contracts_reject_minirocket_candidates_and_inconsistent_gate_counts_antigravity() -> None:
    with pytest.raises(ValueError):
        _assessment_antigravity(method="minirocket_local_gap")
    with pytest.raises(ValueError):
        InnerCandidateGateAntigravity(
            method="minirocket_local_gap",
            enabled=True,
            changed_families=1,
            winning_families=1,
            losing_families=0,
            tied_families=0,
            min_auc_gain=0.1,
            q20_auc_gain=0.1,
        )
    with pytest.raises(ValueError):
        ResidualDecisionAntigravity(
            method="minirocket_local_gap",
            changed=False,
            reason="strict_local_gate_failed",
            selected_candidate="minirocket_local_gap",
            assessments=(),
        )
    with pytest.raises(ValueError):
        InnerCandidateGateAntigravity(
            method="hydra_local_gap",
            enabled=True,
            changed_families=3,
            winning_families=1,
            losing_families=0,
            tied_families=0,
            min_auc_gain=0.1,
            q20_auc_gain=0.1,
        )
    gate = InnerCandidateGateAntigravity(
        method="hydra_local_gap",
        enabled=True,
        changed_families=3,
        winning_families=1,
        losing_families=1,
        tied_families=1,
        min_auc_gain=0.1,
        q20_auc_gain=0.1,
    )
    assert gate.changed_families == (
        gate.winning_families + gate.losing_families + gate.tied_families
    )
    preserved = ResidualDecisionAntigravity(
        method="minirocket_local_gap",
        changed=False,
        reason="strict_local_gate_failed",
        selected_candidate=None,
        assessments=(),
    )
    assert preserved.method == "minirocket_local_gap"
    assert preserved.selected_candidate is None
    accepted = ResidualDecisionAntigravity(
        method="multirocket_local_gap",
        changed=True,
        reason="strict_candidate_accept",
        selected_candidate="multirocket_local_gap",
        assessments=(_assessment_antigravity(),),
    )
    assert accepted.selected_candidate == "multirocket_local_gap"


def test_contracts_reject_nonfinite_duplicates_and_inconsistent_counts_antigravity() -> None:
    with pytest.raises(ValueError):
        _assessment_antigravity(min_auc_gain=float("nan"))
    with pytest.raises(ValueError):
        _assessment_antigravity(q20_auc_gain=float("inf"))
    with pytest.raises(ValueError):
        _assessment_antigravity(neighbour_families=("A", "A"), support=2)
    with pytest.raises(ValueError):
        _assessment_antigravity(support=3, zf_wins=2, zf_losses=0, zf_ties=0)
    with pytest.raises(ValueError):
        ReferenceStatisticsAntigravity(
            passed=True,
            zf_wins=1,
            zf_losses=0,
            zf_ties=0,
            min_auc_gain=float("nan"),
            q20_auc_gain=0.1,
        )
    with pytest.raises(ValueError):
        InnerCandidateGateAntigravity(
            method="hydra_local_gap",
            enabled=True,
            changed_families=1,
            winning_families=1,
            losing_families=0,
            tied_families=0,
            min_auc_gain=float("inf"),
            q20_auc_gain=0.1,
        )
    with pytest.raises(ValueError):
        ResidualDecisionAntigravity(
            method="not_a_method",
            changed=False,
            reason="row_knn_alternative_preserved",
            selected_candidate=None,
            assessments=(),
        )
    with pytest.raises(ValueError):
        _assessment_antigravity(passed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        InnerCandidateGateAntigravity(
            method="hydra_local_gap",
            enabled=True,
            changed_families=-1,
            winning_families=0,
            losing_families=0,
            tied_families=0,
            min_auc_gain=0.0,
            q20_auc_gain=0.1,
        )


def test_route_feature_keys_have_no_leakage_tokens_antigravity() -> None:
    forbidden = ("dataset", "family", "name", "score", "auc", "zf", "label")
    for key in FEATURE_KEYS_ANTIGRAVITY:
        lowered = key.lower()
        for token in forbidden:
            assert token not in lowered


def test_unique_passing_candidate_is_accepted_antigravity() -> None:
    rows = tuple(
        _row_antigravity(f"F{index}_normal_1", index / 10, multi_gain=(0.10, 1))
        for index in range(3)
    )
    decision = (
        StrictResidualPolicyAntigravity(k=3)
        .fit(rows)
        .route(query_features_antigravity(), row_knn_method="minirocket_local_gap")
    )
    assert decision.method == "multirocket_local_gap"
    assert decision.changed is True
    assert decision.reason == "strict_candidate_accept"
    assert decision.selected_candidate == "multirocket_local_gap"


def test_no_passing_candidate_keeps_minirocket_antigravity() -> None:
    rows = tuple(
        _row_antigravity(
            f"F{index}_normal_1",
            index / 10,
            multi_gain=(0.10, 1) if index == 0 else (-0.10, -1),
        )
        for index in range(2)
    )
    decision = (
        StrictResidualPolicyAntigravity(k=2)
        .fit(rows)
        .route(query_features_antigravity(), row_knn_method="minirocket_local_gap")
    )
    assert decision.method == "minirocket_local_gap"
    assert decision.changed is False
    assert decision.reason == "strict_local_gate_failed"
    assert decision.selected_candidate is None
    assert math.isfinite(decision.assessments[0].min_auc_gain)
