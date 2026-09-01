"""Focused tests for the Exp483 TRAIN opportunity and oracle audit."""

import json

import pytest

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    EvidenceRowAntigravity,
)
from experiments.exp483_train_opportunity_oracle_antigravity.diagnostic_antigravity import (
    build_result_antigravity,
    collect_opportunity_census_antigravity,
    decide_next_step_antigravity,
    oracle_neighbour_assessment_antigravity,
    opportunity_records_antigravity,
    render_dashboard_antigravity,
    render_report_antigravity,
    summarize_opportunities_antigravity,
    summarize_oracle_records_antigravity,
)


def test_row_opportunity_keeps_zero_auc_separate_from_strict_positive_antigravity() -> None:
    row = EvidenceRowAntigravity(
        dataset="F_normal_1",
        source_family="F",
        proposal="minirocket_local_gap",
        features=(10.0, 20.0, 0.4, 0.0, 3.0, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", 0.80, 2),
            CandidateMetricAntigravity("multirocket_local_gap", 0.80, 3),
            CandidateMetricAntigravity("hydra_local_gap", 0.79, 4),
            CandidateMetricAntigravity("hydra_multirocket_local_gap", 0.81, 2),
        ),
    )

    records = {
        record["method"]: record
        for record in opportunity_records_antigravity(
            row, inner_family="F"
        )
    }

    assert records["multirocket_local_gap"] == {
        "inner_family": "F",
        "dataset": "F_normal_1",
        "method": "multirocket_local_gap",
        "zf_positive": True,
        "auc_nonnegative": True,
        "auc_positive": False,
        "opportunity": True,
        "strict_positive": False,
    }
    assert records["hydra_local_gap"]["zf_positive"] is True
    assert records["hydra_local_gap"]["auc_nonnegative"] is False
    assert records["hydra_local_gap"]["opportunity"] is False
    assert records["hydra_multirocket_local_gap"]["auc_positive"] is True
    assert records["hydra_multirocket_local_gap"]["zf_positive"] is False
    assert records["hydra_multirocket_local_gap"]["opportunity"] is False


def test_opportunity_summary_counts_rows_datasets_and_families_separately_antigravity() -> None:
    records = (
        {
            "dataset": "F1_normal_1",
            "inner_family": "F1",
            "method": "multirocket_local_gap",
            "zf_positive": True,
            "auc_nonnegative": True,
            "auc_positive": False,
            "opportunity": True,
            "strict_positive": False,
        },
        {
            "dataset": "F1_normal_2",
            "inner_family": "F1",
            "method": "multirocket_local_gap",
            "zf_positive": False,
            "auc_nonnegative": True,
            "auc_positive": True,
            "opportunity": False,
            "strict_positive": False,
        },
        {
            "dataset": "F2_normal_1",
            "inner_family": "F2",
            "method": "multirocket_local_gap",
            "zf_positive": True,
            "auc_nonnegative": True,
            "auc_positive": True,
            "opportunity": True,
            "strict_positive": True,
        },
    )

    assert summarize_opportunities_antigravity(records) == [
        {
            "method": "multirocket_local_gap",
            "evaluated_rows": 3,
            "zf_positive_count": 2,
            "auc_nonnegative_count": 3,
            "auc_positive_count": 2,
            "opportunity_count": 2,
            "strict_positive_count": 1,
            "opportunity_dataset_count": 2,
            "opportunity_family_count": 2,
            "opportunity_rate": 2 / 3,
            "strict_positive_rate": 1 / 3,
        }
    ]


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


def test_census_audits_only_nested_rows_routed_to_minirocket_antigravity() -> None:
    result = collect_opportunity_census_antigravity(
        tuple(_flat_row_antigravity(family) for family in ("A", "B", "C", "D"))
    )

    assert result["routing"] == {
        "inner_rows": 12,
        "minirocket_inner_rows": 12,
        "row_knn_alternative_preserved": 0,
    }
    assert len(result["opportunity_summary"]) == 3
    assert {
        (row["evaluated_rows"], row["opportunity_count"], row["opportunity_family_count"])
        for row in result["opportunity_summary"]
    } == {(12, 0, 0)}
    assert result["oracle_summary"] == []
    assert result["opportunity_neighbour_counts"] == []


def _multirocket_gain_row_antigravity(
    family: str, suffix: int, *, auc_gain: float, zf_gain: int
) -> EvidenceRowAntigravity:
    return EvidenceRowAntigravity(
        dataset=f"{family}_normal_{suffix}",
        source_family=family,
        proposal="minirocket_local_gap",
        features=(10.0, 20.0, 0.4, 0.0, 3.0, 1.0, 0.2),
        scores=(
            CandidateMetricAntigravity("minirocket_local_gap", 0.5, 2),
            CandidateMetricAntigravity(
                "multirocket_local_gap", 0.5 + auc_gain, 2 + zf_gain
            ),
            CandidateMetricAntigravity("hydra_local_gap", 0.5, 2),
            CandidateMetricAntigravity("hydra_multirocket_local_gap", 0.5, 2),
        ),
    )


def test_oracle_uses_best_opportunity_from_each_distinct_family_antigravity() -> None:
    rows = (
        _multirocket_gain_row_antigravity("A", 1, auc_gain=0.01, zf_gain=1),
        _multirocket_gain_row_antigravity("A", 2, auc_gain=0.20, zf_gain=2),
        _multirocket_gain_row_antigravity("B", 1, auc_gain=0.02, zf_gain=1),
        _multirocket_gain_row_antigravity("C", 1, auc_gain=-0.01, zf_gain=2),
        _multirocket_gain_row_antigravity("D", 1, auc_gain=0.0, zf_gain=1),
    )

    result = oracle_neighbour_assessment_antigravity(
        rows, method="multirocket_local_gap", k=3
    )

    assert result["opportunity_family_count"] == 3
    assert result["support"] == 3
    assert result["support_ge_k"] is True
    assert result["passed"] is True
    assert result["zf_wins"] == 3
    assert result["zf_losses"] == 0
    assert result["neighbour_families"] == ("A", "B", "D")
    assert result["neighbour_datasets"] == (
        "A_normal_2",
        "B_normal_1",
        "D_normal_1",
    )
    assert result["q20_auc_gain"] > 0.0


def test_oracle_rejects_support_below_k_antigravity() -> None:
    result = oracle_neighbour_assessment_antigravity(
        (
            _multirocket_gain_row_antigravity(
                "A", 1, auc_gain=0.10, zf_gain=1
            ),
            _multirocket_gain_row_antigravity(
                "B", 1, auc_gain=0.20, zf_gain=1
            ),
        ),
        method="multirocket_local_gap",
        k=3,
    )

    assert result["support"] == 2
    assert result["support_ge_k"] is False
    assert result["passed"] is False
    assert result["reason"] == "insufficient_distinct_families"


def test_oracle_summary_keeps_nested_counts_and_distinct_family_accepts_antigravity() -> None:
    records = (
        {
            "method": "hydra_local_gap",
            "dataset": "F1_normal_1",
            "inner_family": "F1",
            "opportunity": True,
            "opportunity_neighbour_count": 31,
            "oracle_support_ge_k": True,
            "oracle_accepted": True,
        },
        {
            "method": "hydra_local_gap",
            "dataset": "F1_normal_1",
            "inner_family": "F1",
            "opportunity": True,
            "opportunity_neighbour_count": 31,
            "oracle_support_ge_k": True,
            "oracle_accepted": True,
        },
        {
            "method": "hydra_local_gap",
            "dataset": "F2_normal_1",
            "inner_family": "F2",
            "opportunity": True,
            "opportunity_neighbour_count": 30,
            "oracle_support_ge_k": False,
            "oracle_accepted": False,
        },
        {
            "method": "hydra_local_gap",
            "dataset": "F3_normal_1",
            "inner_family": "F3",
            "opportunity": False,
            "opportunity_neighbour_count": 31,
            "oracle_support_ge_k": True,
            "oracle_accepted": False,
        },
    )

    assert summarize_oracle_records_antigravity(records) == {
        "oracle_summary": [
            {
                "method": "hydra_local_gap",
                "opportunity_count": 3,
                "opportunity_family_count": 2,
                "opportunity_neighbour_ge_k_count": 2,
                "opportunity_neighbour_ge_k_rate": 2 / 3,
                "oracle_accept_count": 2,
                "oracle_accept_dataset_count": 1,
                "oracle_accept_family_count": 1,
                "oracle_accept_rate": 2 / 3,
            }
        ],
        "opportunity_neighbour_counts": [
            {
                "method": "hydra_local_gap",
                "opportunity_neighbour_count": 30,
                "target_opportunity_rows": 1,
            },
            {
                "method": "hydra_local_gap",
                "opportunity_neighbour_count": 31,
                "target_opportunity_rows": 2,
            },
        ],
    }


def test_next_step_uses_preregistered_family_threshold_antigravity() -> None:
    cases = (
        (
            [{"method": "m", "opportunity_family_count": 19}],
            [{"method": "m", "oracle_accept_family_count": 19}],
            "abandon_current_residual_candidates",
        ),
        (
            [{"method": "m", "opportunity_family_count": 20}],
            [{"method": "m", "oracle_accept_family_count": 19}],
            "redesign_candidates_gate_or_k",
        ),
        (
            [{"method": "m", "opportunity_family_count": 20}],
            [{"method": "m", "oracle_accept_family_count": 20}],
            "replace_descriptors_and_distance",
        ),
    )

    assert [
        decide_next_step_antigravity(opportunity, oracle, min_families=20)[
            "decision"
        ]
        for opportunity, oracle, _expected in cases
    ] == [expected for _opportunity, _oracle, expected in cases]


def test_result_rejects_exp482_routing_mismatch_antigravity() -> None:
    census = {
        "routing": {
            "inner_rows": 12,
            "minirocket_inner_rows": 11,
            "row_knn_alternative_preserved": 1,
        },
        "opportunity_summary": [],
        "oracle_summary": [],
        "opportunity_neighbour_counts": [],
    }
    exp482_result = {
        "tables": {
            "routing_composition": [
                {"reason": "inner_rows", "count": 12},
                {"reason": "strict_local_gate_failed", "count": 12},
                {"reason": "strict_candidate_accept", "count": 0},
                {"reason": "row_knn_alternative_preserved", "count": 0},
            ]
        }
    }

    with pytest.raises(RuntimeError, match="Exp482 routing changed"):
        build_result_antigravity(
            census,
            exp482_result=exp482_result,
            dataset_count=4,
            family_count=4,
            exp481_result_sha256="a" * 64,
            exp482_result_sha256="b" * 64,
        )


def test_result_html_explains_descriptor_replacement_decision_antigravity() -> None:
    census = {
        "routing": {
            "inner_rows": 12,
            "minirocket_inner_rows": 12,
            "row_knn_alternative_preserved": 0,
        },
        "opportunity_summary": [
            {
                "method": "hydra_local_gap",
                "evaluated_rows": 12,
                "zf_positive_count": 6,
                "auc_nonnegative_count": 10,
                "auc_positive_count": 5,
                "opportunity_count": 5,
                "strict_positive_count": 5,
                "opportunity_dataset_count": 20,
                "opportunity_family_count": 20,
                "opportunity_rate": 5 / 12,
                "strict_positive_rate": 5 / 12,
            }
        ],
        "oracle_summary": [
            {
                "method": "hydra_local_gap",
                "opportunity_count": 5,
                "opportunity_family_count": 20,
                "opportunity_neighbour_ge_k_count": 5,
                "opportunity_neighbour_ge_k_rate": 1.0,
                "oracle_accept_count": 5,
                "oracle_accept_dataset_count": 20,
                "oracle_accept_family_count": 20,
                "oracle_accept_rate": 1.0,
            }
        ],
        "opportunity_neighbour_counts": [
            {
                "method": "hydra_local_gap",
                "opportunity_neighbour_count": 31,
                "target_opportunity_rows": 5,
            }
        ],
    }
    exp482_result = {
        "tables": {
            "routing_composition": [
                {"reason": "inner_rows", "count": 12},
                {"reason": "strict_local_gate_failed", "count": 12},
                {"reason": "strict_candidate_accept", "count": 0},
                {"reason": "row_knn_alternative_preserved", "count": 0},
            ]
        }
    }

    result = build_result_antigravity(
        census,
        exp482_result=exp482_result,
        dataset_count=4,
        family_count=4,
        exp481_result_sha256="a" * 64,
        exp482_result_sha256="b" * 64,
    )
    dashboard = render_dashboard_antigravity(result)
    report = render_report_antigravity(result)
    canonical_result = json.loads(json.dumps(result, sort_keys=True))

    assert result["decision"]["decision"] == "replace_descriptors_and_distance"
    assert result["acceleration_assessment"]["numba_parity_result"] == "not_applicable"
    assert result["acceleration_assessment"]["numba_benchmark_seconds"] is None
    assert result["acceleration_assessment"]["mlx_parity_result"] == "not_applicable"
    assert result["acceleration_assessment"]["mlx_benchmark_seconds"] is None
    assert dashboard.count('type="checkbox" checked disabled') == 4
    assert "7개 특징과 거리 계산을 개선" in dashboard
    assert "Hydra: 개선 기회 <strong>20개 family</strong>" in dashboard
    assert report.count("<table") == 4
    assert "TRAIN 전용" in report
    assert "실제 라우터에는 사용하지 않습니다" in report
    assert "왜 특징과 거리가 문제인가" in report
    assert report == render_report_antigravity(canonical_result)
