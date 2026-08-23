"""Small label-free Korean dashboard snapshot for Exp466 Task 5."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp466QualificationAntigravity
from .evaluation_antigravity import QualificationDecisionAntigravity


def _decision_payload_antigravity(
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    return {
        "status": decision.status,
        "primary_candidate": decision.primary_candidate,
        "eligible_for_exp467_design": decision.eligible_for_exp467_design,
        "primary_passed_count": decision.primary_passed_count,
        "primary_required_count": decision.primary_required_count,
        "failure_reasons": list(decision.failure_reasons),
        "routing_applied": decision.routing_applied,
        "champion_modified": decision.champion_modified,
        "pending_modified": decision.pending_modified,
        "promotion_applied": decision.promotion_applied,
        "test_labels_opened": decision.test_labels_opened,
        "label_query_count": decision.label_query_count,
    }


def _expected_decision_payload_antigravity(
    qualification: Exp466QualificationAntigravity,
) -> dict[str, object]:
    passed_count = sum(row.passed for row in qualification.fold_results)
    reasons: list[str] = []
    if passed_count != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
        reasons.append("c9_not_10_of_10")
    for row in qualification.fold_results:
        for reason in row.failure_reasons:
            if reason not in reasons:
                reasons.append(reason)
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if qualification.normal_confirmations[seed] != 0:
            reason = f"seed_{seed}_normal_confirmations_not_zero"
            if reason not in reasons:
                reasons.append(reason)
    eligible = not reasons
    return {
        "status": (
            "qualification_passed_awaiting_exp467_design"
            if eligible
            else "qualification_failed_blocked_before_fn_validation"
        ),
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "eligible_for_exp467_design": eligible,
        "primary_passed_count": passed_count,
        "primary_required_count": FROZEN_CONFIG_ANTIGRAVITY.required_fold_results,
        "failure_reasons": reasons,
        "routing_applied": False,
        "champion_modified": False,
        "pending_modified": False,
        "promotion_applied": False,
        "test_labels_opened": False,
        "label_query_count": 0,
    }


def _validate_inputs_antigravity(
    qualification: Exp466QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> None:
    if type(qualification) is not Exp466QualificationAntigravity:
        raise ValueError("dashboard requires exact Exp466 qualification")
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("dashboard requires exact Exp466 decision")
    if _decision_payload_antigravity(decision) != _expected_decision_payload_antigravity(
        qualification,
    ):
        raise ValueError("dashboard decision differs from qualification")


def export_dashboard_antigravity(
    qualification: Exp466QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    """Return deterministic Korean text without claiming a Task 5 real result."""

    _validate_inputs_antigravity(qualification, decision)
    improved = tuple(
        f"{row.seed}:{row.fold_index}"
        for row in qualification.fold_results
        if row.c9_period_auc > row.c0_period_auc
    )
    failed = tuple(
        f"{row.seed}:{row.fold_index}"
        for row in qualification.fold_results
        if not row.passed
    )
    normal_confirmations = sum(qualification.normal_confirmations.values())
    failed_sentence = (
        f"실패한 검증 묶음은 {', '.join(failed)}입니다."
        if failed
        else "실패한 검증 묶음은 없습니다."
    )
    html = (
        '<!doctype html>\n<html lang="ko">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Exp466 Task 5 C9 확인 요약</title>\n"
        "</head>\n<body>\n"
        "<main>\n"
        "<h1>Exp466 Task 5 C9 확인 요약</h1>\n"
        f"<p>C9의 주기 AUC가 C0보다 좋아진 검증 묶음은 {len(improved)}개입니다.</p>\n"
        f"<p>{failed_sentence}</p>\n"
        f"<p>정상 오탐 확인은 {normal_confirmations}개입니다.</p>\n"
        "<p>Task 5에는 실제 비공개 또는 공식 결과가 없습니다.</p>\n"
        "<p>이 합성 값은 성능 결과가 아닙니다.</p>\n"
        "</main>\n"
        "</body>\n</html>\n"
    )
    return {
        "후보": "C9",
        "주기_AUC_개선_fold_수": len(improved),
        "주기_AUC_개선_fold": list(improved),
        "실패_fold": list(failed),
        "정상_confirmation_수": normal_confirmations,
        "Task5_실제_private_official_결과": False,
        "html_antigravity": html,
    }


__all__ = ("export_dashboard_antigravity",)
