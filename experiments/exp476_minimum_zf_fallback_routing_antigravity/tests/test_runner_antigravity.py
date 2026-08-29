from __future__ import annotations

from experiments.exp476_minimum_zf_fallback_routing_antigravity.run_antigravity import (
    build_gates_antigravity,
    render_dashboard_antigravity,
)


def test_gates_require_auc_and_zf_not_worse_than_minirocket_antigravity() -> None:
    result = {
        "baseline": {"average_auc": 0.90, "average_zf_tp": 1.0},
        "policy": {
            "average_auc": 0.91,
            "average_zf_tp": 1.01,
            "severe_auc_loss_over_0_05": 1,
        },
        "leakage_checks": {"family_overlap_count": 0},
    }
    assert build_gates_antigravity(result)["overall_pass"] is True


def test_dashboard_checks_completed_steps_and_leaves_official_promotion_unchecked_antigravity() -> None:
    result = {
        "dataset_count": 10,
        "family_count": 2,
        "baseline": {"average_auc": 0.90, "average_zf_tp": 1.0, "severe_auc_loss_over_0_05": 2},
        "policy": {
            "average_auc": 0.91,
            "average_zf_tp": 1.01,
            "severe_auc_loss_over_0_05": 1,
            "accepted_non_baseline": 2,
            "fallback_count": 8,
        },
        "leakage_checks": {"family_overlap_count": 0},
        "minimum_zf_rejection_count": 3,
        "decisions": [],
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
    }
    result["gates"] = build_gates_antigravity(result)
    prior = {"average_auc": 0.90, "average_zf_tp": 0.99, "accepted_non_baseline": 4}

    html = render_dashboard_antigravity(result, prior)

    assert "ZF 손실 사례가 하나라도" in html
    assert html.count('type="checkbox"') == 7
    assert html.count("checked disabled") == 6
    assert "공식 파이프라인 반영 (별도 승인 전 미수행)" in html
