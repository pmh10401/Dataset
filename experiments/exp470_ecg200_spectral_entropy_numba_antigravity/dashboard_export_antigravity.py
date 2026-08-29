"""Portable Korean HTML for an already inspected synthetic-only publication."""

from __future__ import annotations

import html

from .replay_antigravity import InspectedResultAntigravity


def _text_antigravity(value: object) -> str:
    return html.escape(str(value), quote=True)


def export_dashboard_antigravity(inspected: object) -> str:
    if type(inspected) is not InspectedResultAntigravity:
        raise ValueError("dashboard export requires an inspected result")
    q = inspected.qualification
    d = inspected.decision
    fixed = ", ".join(f"{seed}: {count}" for seed, count in q.period_fixed.items())
    broken = ", ".join(f"{seed}: {count}" for seed, count in q.period_broken.items())
    normal = ", ".join(f"{seed}: {count}" for seed, count in q.normal_confirmations.items())
    protected = ", ".join(f"{name}: {count}" for name, count in q.protected_broken.items())
    c0_auc = {
        change: sum(row.c0_per_type_auc[change] for row in q.fold_results) / len(q.fold_results)
        for change in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
    }
    c11_auc = {
        change: sum(row.c11_per_type_auc[change] for row in q.fold_results) / len(q.fold_results)
        for change in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
    }
    fold_pairs = "".join(
        "<li>seed " + _text_antigravity(row.seed) + ", fold " + _text_antigravity(row.fold_index)
        + ": period fixed " + _text_antigravity(row.pair_repairs["period_change"].fixed)
        + ", broken " + _text_antigravity(row.pair_repairs["period_change"].broken) + "</li>"
        for row in q.fold_results
    )
    return "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><title>Exp470 결과 replay</title></head><body><main>" + \
        "<h1>Exp470 결과 replay 확인</h1><p>저장된 합성 기록을 다시 계산한 화면입니다. 실제 private 또는 official 실행 결과는 아닙니다.</p>" + \
        "<label><input type=\"checkbox\" name=\"replay\" checked disabled> 결과 replay 검증</label>" + \
        "<label><input type=\"checkbox\" name=\"private\" disabled> private 실행</label>" + \
        "<label><input type=\"checkbox\" name=\"official\" disabled> official 실행</label>" + \
        "<label><input type=\"checkbox\" name=\"test\" disabled> TEST</label>" + \
        "<label><input type=\"checkbox\" name=\"fn\" disabled> FN</label>" + \
        "<label><input type=\"checkbox\" name=\"routing\" disabled> routing</label>" + \
        "<section><h2>현재 판정</h2><p>통과: " + _text_antigravity(d.passed) + ", private 준비: " + _text_antigravity(d.eligible_for_private_reproducibility) + "</p>" + \
        "<p>상태: awaiting_private_checkpoint</p><p>Numba: 사용함, MLX: 평가했지만 선택하지 않음</p></section>" + \
        "<section><h2>fold와 seed 합계</h2><p>period fixed: " + _text_antigravity(fixed) + "</p><p>period broken: " + _text_antigravity(broken) + "</p><p>정상 오확인: " + _text_antigravity(normal) + "</p><p>보호 대상 broken: " + _text_antigravity(protected) + "</p>" + \
        "<p>fold 수: " + _text_antigravity(len(inspected.records)) + "</p><ul>" + fold_pairs + "</ul>" + \
        "<p>C0 AUC: " + _text_antigravity(c0_auc) + "</p><p>C11 AUC: " + _text_antigravity(c11_auc) + "</p></section>" + \
        "</main></body></html>\n"


__all__ = ("export_dashboard_antigravity",)
