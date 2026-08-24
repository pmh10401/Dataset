"""Portable Korean HTML for an already inspected Exp471 synthetic publication."""

from __future__ import annotations

import html

from .contracts_antigravity import Exp471QualificationAntigravity
from .replay_antigravity import InspectedResultAntigravity


_CHANGE_TYPES_ANTIGRAVITY = ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")


def _text_antigravity(value: object) -> str:
    return html.escape(str(value), quote=True)


def _mean_auc_antigravity(
    key: str,
    qualification: Exp471QualificationAntigravity,
) -> dict[str, float]:
    return {
        change: sum(getattr(row, key)[change] for row in qualification.fold_results) / len(qualification.fold_results)
        for change in _CHANGE_TYPES_ANTIGRAVITY
    }


def export_dashboard_antigravity(inspected: object) -> str:
    if type(inspected) is not InspectedResultAntigravity:
        raise ValueError("dashboard export requires an inspected result")
    decision = inspected.decision
    qualification = inspected.qualification
    first_record = next(iter(inspected.records.values()))
    fit_shift_text = "[" + ", ".join(str(int(value)) for value in first_record.fit_shifts) + "]"
    peak_absent_total = int(sum(int((~record.fit_peak_presence).sum()) for record in inspected.records.values()))
    fixed = ", ".join(f"{seed}: {count}" for seed, count in qualification.period_fixed.items())
    broken = ", ".join(f"{seed}: {count}" for seed, count in qualification.period_broken.items())
    normal = ", ".join(f"{seed}: {count}" for seed, count in qualification.normal_confirmations.items())
    protected = ", ".join(f"{name}: {count}" for name, count in qualification.protected_broken.items())
    c0_auc = _mean_auc_antigravity("c0_per_type_auc", qualification)
    c12_auc = _mean_auc_antigravity("c12_per_type_auc", qualification)
    fold_pairs = "".join(
        "<li>seed " + _text_antigravity(row.seed) + ", fold " + _text_antigravity(row.fold_index)
        + ": period fixed " + _text_antigravity(row.pair_repairs["period_change"].fixed)
        + ", broken " + _text_antigravity(row.pair_repairs["period_change"].broken)
        + ", peak/shift evidence replayed</li>"
        for row in qualification.fold_results
    )
    return (
        '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
        "<title>Exp471 결과 replay</title></head><body><main>"
        "<h1>Exp471 ACF peak-lag shift 결과 확인</h1>"
        "<p>저장된 합성 입력과 peak·shift·raw 증거를 다시 계산한 화면입니다. 실제 private 또는 official 실행 결과는 아닙니다.</p>"
        '<label><input type="checkbox" name="replay" checked disabled> 결과 replay 검증</label>'
        '<label><input type="checkbox" name="private" disabled> private 실행</label>'
        '<label><input type="checkbox" name="official" disabled> official 실행</label>'
        '<label><input type="checkbox" name="test" disabled> TEST</label>'
        '<label><input type="checkbox" name="fn" disabled> FN</label>'
        '<label><input type="checkbox" name="routing" disabled> routing</label>'
        "<section><h2>현재 판정</h2><p>통과: " + _text_antigravity(decision.passed)
        + ", private 재현 준비: " + _text_antigravity(decision.eligible_for_private_reproducibility) + "</p>"
        "<p>상태: awaiting_private_checkpoint</p>"
        "<p>Numba: 사용함, MLX: 평가했지만 선택하지 않음</p></section>"
        "<section><h2>C0/E12/C12와 pair repair</h2>"
        "<p>period fixed: " + _text_antigravity(fixed) + "</p>"
        "<p>fit shifts: " + _text_antigravity(fit_shift_text) + "</p>"
        "<p>peak 부재 감지 수: " + _text_antigravity(peak_absent_total) + "</p>"
        "<p>period broken: " + _text_antigravity(broken) + "</p>"
        "<p>보호 대상 broken: " + _text_antigravity(protected) + "</p>"
        "<p>정상 confirmation: " + _text_antigravity(normal) + "</p>"
        "<p>fold 수: " + _text_antigravity(len(inspected.records)) + "</p><ul>" + fold_pairs + "</ul>"
        "<p>C0 AUC: " + _text_antigravity(c0_auc) + "</p>"
        "<p>C12 AUC: " + _text_antigravity(c12_auc) + "</p></section>"
        "</main></body></html>\n"
    )


__all__ = ("export_dashboard_antigravity",)
