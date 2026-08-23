"""Small Korean dashboard snapshot derived from Exp465 evidence."""

from __future__ import annotations

from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_qualification_antigravity
from .qualification_antigravity import Exp465QualificationResultAntigravity, recompute_qualification_antigravity


def export_dashboard_antigravity(
    qualification: Exp465QualificationResultAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    """Return a label-free, score-derived dashboard payload in simple Korean."""
    replayed = recompute_qualification_antigravity(qualification)
    expected = evaluate_qualification_antigravity(replayed)
    if decision != expected:
        raise ValueError("dashboard decision differs from replayed evidence")
    return {
        "후보": "C8",
        "통과_기록": decision.primary_passed_count,
        "필요_기록": decision.primary_required_count,
        "다음_재현_가능": decision.eligible_for_routing_research,
        "실패_이유": list(decision.failure_reasons),
        "정상_confirmation": {str(seed): replayed.seed_normal_confirmations[seed] for seed in replayed.seed_normal_confirmations},
        "방법_요약": [
            {
                "방법": summary.method,
                "통과": summary.passed_count,
                "실패": summary.failed_count,
                "최저_AUC": list(summary.min_per_type_auc),
                "최저_평균_AUC": summary.min_mean_auc,
                "정상_confirmation": summary.normal_confirmation_count,
                "점수_유한": summary.all_scores_finite,
                "통과_여부": summary.passed,
            }
            for summary in replayed.summaries
        ],
        "보수성_비교": {
            "gradual_not_below_c0": sum(row.c8_gradual_auc >= row.c0_gradual_auc for row in replayed.rows),
            "period_not_below_c0": sum(row.c8_period_auc >= row.c0_period_auc for row in replayed.rows),
        },
        "TEST_라벨_조회": 0,
    }


__all__ = ("export_dashboard_antigravity",)
