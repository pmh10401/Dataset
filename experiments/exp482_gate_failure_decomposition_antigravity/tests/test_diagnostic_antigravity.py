"""Focused tests for the Exp482 gate-failure census."""

import pytest

from experiments.exp481_strict_evidence_residual_correction_antigravity.contracts_antigravity import (
    StrictCandidateAssessmentAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp482_gate_failure_decomposition_antigravity.diagnostic_antigravity import (
    aggregate_findings_antigravity,
    build_result_antigravity,
    collect_train_census_antigravity,
    diagnose_assessments_antigravity,
    render_dashboard_antigravity,
    render_report_antigravity,
    summarize_census_antigravity,
)


def _assessment_antigravity(
    method: str,
    *,
    wins: int,
    losses: int,
    ties: int,
    minimum: float,
    q20: float,
) -> StrictCandidateAssessmentAntigravity:
    return StrictCandidateAssessmentAntigravity(
        method=method,
        passed=False,
        support=wins + losses + ties,
        zf_wins=wins,
        zf_losses=losses,
        zf_ties=ties,
        min_auc_gain=minimum,
        q20_auc_gain=q20,
        neighbour_families=tuple(f"F{index}" for index in range(wins + losses + ties)),
    )


def test_diagnosis_separates_each_strict_failure_clause_antigravity() -> None:
    records = diagnose_assessments_antigravity(
        (
            _assessment_antigravity(
                "multirocket_local_gap",
                wins=2,
                losses=2,
                ties=1,
                minimum=0.10,
                q20=0.05,
            ),
            _assessment_antigravity(
                "hydra_local_gap",
                wins=3,
                losses=1,
                ties=1,
                minimum=-0.02,
                q20=0.01,
            ),
            _assessment_antigravity(
                "hydra_multirocket_local_gap",
                wins=3,
                losses=1,
                ties=1,
                minimum=0.0,
                q20=0.0,
            ),
        ),
        outer_family="OUT",
        inner_family="IN",
        dataset="IN_normal_1",
    )

    by_method = {record["method"]: record for record in records}
    assert by_method["multirocket_local_gap"]["failure_clauses"] == (
        "wins_not_greater",
    )
    assert by_method["multirocket_local_gap"]["zf_margin"] == 0
    assert by_method["hydra_local_gap"]["failure_clauses"] == (
        "negative_min_auc",
    )
    assert by_method["hydra_local_gap"]["min_auc_margin"] == -0.02
    assert by_method["hydra_multirocket_local_gap"]["failure_clauses"] == (
        "nonpositive_q20_auc",
    )
    assert by_method["hydra_multirocket_local_gap"]["q20_auc_margin"] == 0.0
    assert {record["rank_status"] for record in records} == {"not_reached"}
    assert not any(record["accepted"] for record in records)


def _flat_row_antigravity(family: str) -> EvidenceRowAntigravity:
    return EvidenceRowAntigravity(
        dataset=f"{family}_normal_1",
        source_family=family,
        proposal="minirocket_local_gap",
        features=(10.0, 20.0, 0.4, 0.0, 3.0, 1.0, 0.2),
        scores=tuple(
            CandidateMetricAntigravity(method, 0.8, 2)
            for method in (
                "minirocket_local_gap",
                "multirocket_local_gap",
                "hydra_local_gap",
                "hydra_multirocket_local_gap",
            )
        ),
    )


def test_train_census_records_every_inner_candidate_without_changing_route_antigravity() -> None:
    result = collect_train_census_antigravity(
        tuple(_flat_row_antigravity(family) for family in ("A", "B", "C", "D"))
    )

    assert result["routing"] == {
        "inner_rows": 12,
        "strict_local_gate_failed": 12,
        "strict_candidate_accept": 0,
        "row_knn_alternative_preserved": 0,
    }
    assert len(result["records"]) == 36
    assert {tuple(record["failure_clauses"]) for record in result["records"]} == {
        ("wins_not_greater", "nonpositive_q20_auc")
    }


def test_summary_separates_overlapping_and_single_clause_failures_antigravity() -> None:
    records = (
        {
            "outer_family": "A",
            "method": "multirocket_local_gap",
            "failure_clauses": ("wins_not_greater",),
            "zf_margin": 0,
            "min_auc_margin": 0.1,
            "q20_auc_margin": 0.1,
            "rank_status": "not_reached",
        },
        {
            "outer_family": "A",
            "method": "multirocket_local_gap",
            "failure_clauses": (
                "wins_not_greater",
                "nonpositive_q20_auc",
            ),
            "zf_margin": 1,
            "min_auc_margin": 0.0,
            "q20_auc_margin": 0.0,
            "rank_status": "not_reached",
        },
        {
            "outer_family": "A",
            "method": "multirocket_local_gap",
            "failure_clauses": (),
            "zf_margin": 2,
            "min_auc_margin": 0.2,
            "q20_auc_margin": 0.2,
            "rank_status": "unique_best",
        },
    )
    summary = summarize_census_antigravity(
        {
            "routing": {
                "inner_rows": 1,
                "strict_local_gate_failed": 0,
                "strict_candidate_accept": 1,
                "row_knn_alternative_preserved": 0,
            },
            "records": records,
        }
    )

    failures = summary["clause_failures"][0]
    assert failures["candidate_count"] == 3
    assert failures["wins_not_greater_count"] == 2
    assert failures["nonpositive_q20_auc_count"] == 1
    assert failures["negative_min_auc_count"] == 0
    assert summary["failure_combinations"] == [
        {
            "outer_family": "A",
            "method": "multirocket_local_gap",
            "combination": "pass",
            "count": 1,
        },
        {
            "outer_family": "A",
            "method": "multirocket_local_gap",
            "combination": "wins_not_greater",
            "count": 1,
        },
        {
            "outer_family": "A",
            "method": "multirocket_local_gap",
            "combination": "wins_not_greater+nonpositive_q20_auc",
            "count": 1,
        },
    ]
    assert summary["single_clause_failures"] == [
        {
            "outer_family": "A",
            "method": "multirocket_local_gap",
            "clause": "wins_not_greater",
            "count": 1,
            "margin_min": 0.0,
            "margin_p20": 0.0,
            "margin_median": 0.0,
            "margin_p80": 0.0,
            "margin_max": 0.0,
        }
    ]

    result = {
        "state": "diagnostic_complete",
        "dataset_count": 4,
        "family_count": 4,
        "candidate_record_count": 3,
        "exp481_identity": {"evaluation_equal": True},
        "tables": summary,
    }
    dashboard = render_dashboard_antigravity(result)
    report = render_report_antigravity(result)
    assert dashboard.count('type="checkbox" checked disabled') == 4
    assert "Exp481 결정 동일: <strong>예</strong>" in dashboard
    assert "다중 조건 실패: <strong>1</strong>" in dashboard
    assert "K=31은 유지" in dashboard
    assert report.count("<table") == 5
    assert "wins_not_greater" in report
    assert "한 조건만 낮추기 전에" in report
    assert "F(m) &lt; 20" in report
    assert "A_oracle(m) ≥ 20" in report
    assert aggregate_findings_antigravity(summary) == {
        "candidate_count": 3,
        "passed_count": 1,
        "single_clause_failure_count": 1,
        "multiple_clause_failure_count": 1,
        "wins_not_greater_count": 2,
        "negative_min_auc_count": 0,
        "nonpositive_q20_auc_count": 1,
        "candidate_rank_tie_count": 0,
    }


def test_result_rejects_any_exp481_decision_change_antigravity() -> None:
    with pytest.raises(RuntimeError, match="Exp481 evaluation changed"):
        build_result_antigravity(
            {"routing": {}, "records": ()},
            stored_evaluation={"decisions": ["MiniRocket"]},
            recomputed_evaluation={"decisions": ["Hydra"]},
            dataset_count=1,
            family_count=1,
            exp481_result_sha256="a" * 64,
        )
