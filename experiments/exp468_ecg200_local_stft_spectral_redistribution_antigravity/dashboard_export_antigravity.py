"""Deterministic, self-contained Korean HTML dashboard for Exp468 Task 5."""

from __future__ import annotations

from html import escape
from math import isfinite

from .replay_antigravity import InspectedResultAntigravity


def _number_antigravity(value: object) -> str:
    if type(value) not in (int, float) or not isfinite(float(value)):
        raise ValueError("dashboard values must be finite numbers")
    return f"{float(value):.12g}"


def _text_antigravity(value: object) -> str:
    return escape(str(value), quote=True)


def export_dashboard_antigravity(inspected: InspectedResultAntigravity) -> str:
    """Return the same HTML bytes for the same inspected publication."""

    if type(inspected) is not InspectedResultAntigravity:
        raise ValueError("dashboard export requires an inspected Exp468 result")
    snapshot = inspected.dashboard_snapshot
    fold_rows = snapshot["folds"]
    improved_folds = snapshot["improved_folds"]
    protected = snapshot["protected_min_deltas"]
    normal_rows = snapshot["normal_confirmations_by_seed"]
    decision = inspected.decision

    fold_html = "".join(
        "<tr>"
        f"<td>seed {_text_antigravity(row['seed'])}</td>"
        f"<td>fold {_text_antigravity(row['fold_index'])}</td>"
        f"<td>{_number_antigravity(row['c0_period_auc'])}</td>"
        f"<td>{_number_antigravity(row['c10_period_auc'])}</td>"
        f"<td>{_number_antigravity(row['delta'])}</td>"
        f"<td>{_text_antigravity(row['period_confirmations'])}</td>"
        f"<td>{_text_antigravity(', '.join(row['failure_reasons']) or '없음')}</td>"
        "</tr>"
        for row in fold_rows
    )
    improved_text = ", ".join(
        f"seed {row['seed']} / fold {row['fold_index']}" for row in improved_folds
    ) or "없음"
    protected_html = "".join(
        f"<li>{_text_antigravity(change_type)}: 최소 C10−C0 = "
        f"{_number_antigravity(delta)}</li>"
        for change_type, delta in protected.items()
    )
    normal_html = "".join(
        f"<li>seed {_text_antigravity(row['seed'])}: "
        f"정상 확인 {_text_antigravity(row['count'])}개 "
        f"({_text_antigravity('정상 확인 오류') if row['count'] else '오류 없음'})</li>"
        for row in normal_rows
    )
    status_text = (
        "qualification PASS — private 재현성 확인 대기"
        if decision.eligible_for_private_reproducibility
        else "qualification FAIL — private 실행 전 차단"
    )
    status_class = "status warn" if not decision.eligible_for_private_reproducibility else "status"
    return (
        "<!doctype html>\n"
        "<html lang=\"ko\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Exp468 Task 5 대시보드</title>"
        "<style>body{max-width:1180px;margin:2rem auto;padding:0 1rem;"
        "font:16px/1.6 system-ui;color:#172033}table{border-collapse:collapse;width:100%;"
        "margin:1rem 0}th,td{border:1px solid #aab3bf;padding:.45rem;text-align:left;"
        "vertical-align:top}code{background:#eef2f7;padding:.1rem .25rem}.status{"
        "padding:1rem;border-left:6px solid #2457a6;background:#f5f7fa}"
        ".warn{border-left-color:#b42318;background:#fff4f2}</style></head><body>"
        "<main><h1>Exp468 결과·재현성 대시보드</h1>"
        f"<section class=\"{status_class}\">"
        f"<strong>{_text_antigravity(status_text)}</strong>"
        "<p>이 결과는 C0, S10, C10의 TRAIN-only qualification 계약을 보여줍니다. "
        "private 실행 1·2는 <strong>아직 실행하지 않음</strong>입니다. "
        "qualification PASS여도 <strong>private 재현성 확인 대기</strong>입니다.</p></section>"
        "<h2>period 변화</h2>"
        "<p>각 fold에서 C0보다 C10이 얼마나 좋아졌는지 확인합니다.</p>"
        "<table><thead><tr><th>seed</th><th>fold</th><th>C0 period AUC</th>"
        "<th>C10 period AUC</th><th>차이(C10−C0)</th><th>period 확인 수</th>"
        "<th>실패 이유</th></tr></thead><tbody>"
        f"{fold_html}</tbody></table>"
        f"<p><strong>C0보다 개선한 fold</strong>: {_text_antigravity(snapshot['improved_fold_count'])}개"
        f" — {_text_antigravity(improved_text)}</p>"
        "<h2>보호해야 하는 세 변화</h2><ul>"
        f"{protected_html}</ul>"
        "<h2>정상 입력 확인</h2><p>정상 입력은 잘못 확인되면 안 됩니다. 이 표의 설명은 <strong>정상 확인 오류</strong>입니다.</p><ul>"
        f"{normal_html}</ul>"
        "<h2>private 재현성</h2><ul>"
        "<li>실행 1: <strong>아직 실행하지 않음</strong></li>"
        "<li>실행 2: <strong>아직 실행하지 않음</strong></li></ul>"
        "<h2>Task 5 진행 상태</h2>"
        "<p><label><input type=\"checkbox\" checked disabled> 일곱 leaf publication 생성</label></p>"
        "<p><label><input type=\"checkbox\" checked disabled> raw 입력 독립 replay 확인</label></p>"
        "<p><label><input type=\"checkbox\" checked disabled> 결정적 HTML dashboard 생성</label></p>"
        "<p><label><input type=\"checkbox\" disabled> Task 6 private 실행</label></p>"
        "</main></body></html>\n"
    )


__all__ = ("export_dashboard_antigravity",)
