from __future__ import annotations

from experiments.exp475_joint_tail_fallback_routing_antigravity.run_antigravity import (
    build_gates_antigravity,
    render_dashboard_antigravity,
)
from experiments.exp475_joint_tail_fallback_routing_antigravity import run_antigravity as runner_antigravity


def _result_antigravity() -> dict[str, object]:
    return {
        "dataset_count": 4,
        "family_count": 2,
        "baseline": {
            "average_auc": 0.90,
            "average_zf_tp": 1.0,
            "severe_auc_loss_over_0_05": 2,
        },
        "policy": {
            "average_auc": 0.91,
            "average_zf_tp": 1.0,
            "severe_auc_loss_over_0_05": 1,
            "accepted_non_baseline": 1,
            "fallback_count": 3,
        },
        "leakage_checks": {
            "family_overlap_count": 0,
            "dataset_name_feature_count": 0,
            "held_out_score_feature_count": 0,
        },
        "decisions": [
            {
                "dataset": "Alpha_normal_1",
                "method": "hydra_local_gap",
                "accepted": True,
                "reason": "high_confidence_joint_tail_accept",
                "auc": 0.91,
                "zf": 1,
                "baseline_auc": 0.90,
                "baseline_zf": 1,
                "oracle_auc": 0.92,
            }
        ],
    }


def test_gates_require_no_auc_or_zf_degradation_antigravity() -> None:
    result = _result_antigravity()

    gates = build_gates_antigravity(result)

    assert gates["overall_pass"] is True
    result["policy"]["average_zf_tp"] = 0.99  # type: ignore[index]
    assert build_gates_antigravity(result)["overall_pass"] is False


def test_dashboard_checks_completed_work_but_not_official_promotion_antigravity() -> None:
    result = _result_antigravity()
    result["gates"] = build_gates_antigravity(result)
    result["generated_at_utc"] = "2026-08-28T00:00:00+00:00"
    result["selection_changes_from_exp474"] = 0
    result["zf_tail_rejection_count"] = 0
    prior = {"average_auc": 0.905, "average_zf_tp": 0.99, "accepted_non_baseline": 2}

    html = render_dashboard_antigravity(result, prior)

    assert "Exp474" in html and "Exp475" in html
    assert html.count('type="checkbox"') == 7
    assert html.count("checked disabled") == 6
    assert "선택을 바꾸지 못했습니다" in html
    assert "공식 파이프라인 반영 (별도 승인 전 미수행)" in html


def test_selection_change_count_compares_method_and_acceptance_antigravity() -> None:
    prior = [
        {"dataset": "A_normal_1", "method": "hydra_local_gap", "accepted": True},
        {"dataset": "B_normal_1", "method": "minirocket_local_gap", "accepted": False},
    ]
    current = [
        {"dataset": "A_normal_1", "method": "minirocket_local_gap", "accepted": False},
        {"dataset": "B_normal_1", "method": "minirocket_local_gap", "accepted": False},
    ]

    assert runner_antigravity.count_selection_changes_antigravity(current, prior) == 1
