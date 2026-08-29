"""Run Exp476 and publish JSON evidence plus an easy HTML dashboard."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path

from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    attach_features_antigravity,
    load_checkpoint_records_antigravity,
)
from experiments.exp474_conservative_fallback_routing_antigravity.run_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
    FEATURE_CACHE_ANTIGRAVITY,
    _write_json_antigravity,
)
from experiments.exp475_joint_tail_fallback_routing_antigravity.run_antigravity import (
    build_gates_antigravity,
    count_selection_changes_antigravity,
)

from .evaluation_antigravity import evaluate_minimum_zf_family_blocked_antigravity


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
EXP475_RESULT_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "experiments/exp475_joint_tail_fallback_routing_antigravity/results_antigravity.json"
)
EXPECTED_EXP475_SHA256_ANTIGRAVITY = "6d10981f56af82cd057ad3e0121a4d16568a85c7f00bf06d774c70357f9dd63f"
RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP476_DASHBOARD_antigravity.html"
POLICY_KWARGS_ANTIGRAVITY = {
    "k": 31,
    "min_support": 20,
    "positive_rate_threshold": 0.90,
    "lower_quantile": 0.20,
    "min_auc_gain": 0.0,
    "min_zf_gain": 0.0,
    "min_lower_zf_gain": 0.0,
}


def render_dashboard_antigravity(
    result: dict[str, object], prior_policy: dict[str, object]
) -> str:
    baseline = result["baseline"]
    policy = result["policy"]
    gates = result["gates"]
    decisions = result["decisions"]
    assert isinstance(baseline, dict) and isinstance(policy, dict) and isinstance(gates, dict)
    assert isinstance(decisions, list)
    passed = bool(gates["overall_pass"])
    methods = Counter(
        str(row["method"]) for row in decisions if isinstance(row, dict) and row.get("accepted")
    )
    reasons = Counter(
        str(row["reason"]) for row in decisions if isinstance(row, dict) and not row.get("accepted")
    )
    worst = sorted(
        (row for row in decisions if isinstance(row, dict)),
        key=lambda row: float(row["oracle_auc"]) - float(row["auc"]),
        reverse=True,
    )[:12]
    gate_rows = "".join(
        f"<tr><td>{escape(str(name))}</td><td>{escape(str(data['actual']))}</td><td>{escape(str(data['required']))}</td>"
        f"<td class={'pass' if data['pass'] else 'hold'}>{'통과' if data['pass'] else '미달'}</td></tr>"
        for name, data in gates.items()
        if name != "overall_pass" and isinstance(data, dict)
    )
    method_rows = "".join(
        f"<tr><td><code>{escape(method)}</code></td><td>{count}</td></tr>"
        for method, count in sorted(methods.items())
    ) or "<tr><td colspan='2'>승인된 비기준선 선택 없음</td></tr>"
    reason_rows = "".join(
        f"<tr><td>{escape(reason)}</td><td>{count}</td></tr>" for reason, count in reasons.most_common()
    )
    worst_rows = "".join(
        f"<tr><td>{escape(str(row['dataset']))}</td><td><code>{escape(str(row['method']))}</code></td>"
        f"<td>{float(row['auc']):.4f}</td><td>{float(row['oracle_auc'])-float(row['auc']):.4f}</td>"
        f"<td>{escape(str(row['reason']))}</td></tr>"
        for row in worst
    )
    checked = "checked" if passed else ""
    status = "검증 통과" if passed else "검증 보류"
    status_class = "pass" if passed else "hold"
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp476 ZF 최솟값 보호 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.58;color:#182230}}h1,h2{{line-height:1.25}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:12px;padding:16px;background:#f8fafc}}.big{{font-size:1.6rem;font-weight:750}}.pass{{color:#08783e;font-weight:700}}.hold{{color:#b54708;font-weight:700}}.muted{{color:#596579}}.status{{border-left:5px solid;padding:12px 16px;background:#f8fafc}}table{{border-collapse:collapse;width:100%;margin:12px 0 26px}}th,td{{border:1px solid #d8dee8;padding:8px 10px;text-align:left}}th{{background:#eef2f7}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}
</style></head><body><h1>Exp476 ZF 최솟값 보호</h1><p class="status {status_class}"><strong>{status}</strong> — 가까운 과거 사례에 ZF 손실 사례가 하나라도 있으면 MiniRocket으로 돌아갑니다.</p>
<div class="cards"><div class="card"><div class="muted">데이터셋</div><div class="big">{result['dataset_count']}</div></div><div class="card"><div class="muted">분리한 가족</div><div class="big">{result['family_count']}</div></div><div class="card"><div class="muted">평균 AUC</div><div class="big">{float(policy['average_auc']):.4f}</div></div><div class="card"><div class="muted">평균 ZF-TP</div><div class="big">{float(policy['average_zf_tp']):.3f}</div></div><div class="card"><div class="muted">최솟값 거절</div><div class="big">{result['minimum_zf_rejection_count']}</div></div></div>
<h2>1. 쉬운 설명</h2><p>Exp475의 하위 20% 조건은 기존 90% 성공률 조건과 겹쳤습니다. Exp476은 더 직접적으로, 가까운 31개 과거 사례 중 ZF 손실 사례가 하나라도 있으면 다른 방법을 믿지 않고 MiniRocket을 선택합니다.</p>
<h2>2. 결과 비교</h2><table><tr><th>방법</th><th>평균 AUC</th><th>평균 ZF-TP</th><th>큰 AUC 손실</th><th>비기준선 승인</th></tr><tr><td>항상 MiniRocket</td><td>{float(baseline['average_auc']):.6f}</td><td>{float(baseline['average_zf_tp']):.6f}</td><td>{baseline['severe_auc_loss_over_0_05']}</td><td>0</td></tr><tr><td>Exp475</td><td>{float(prior_policy['average_auc']):.6f}</td><td>{float(prior_policy['average_zf_tp']):.6f}</td><td>{prior_policy.get('severe_auc_loss_over_0_05','-')}</td><td>{prior_policy['accepted_non_baseline']}</td></tr><tr><td><strong>Exp476</strong></td><td><strong>{float(policy['average_auc']):.6f}</strong></td><td><strong>{float(policy['average_zf_tp']):.6f}</strong></td><td><strong>{policy['severe_auc_loss_over_0_05']}</strong></td><td><strong>{policy['accepted_non_baseline']}</strong></td></tr></table>
<h2>3. 통과 기준</h2><table><tr><th>기준</th><th>실제</th><th>요구</th><th>판정</th></tr>{gate_rows}</table>
<h2>4. 선택 내역</h2><p>최솟값 보호로 <strong>{result['minimum_zf_rejection_count']}개</strong>를 추가 거절했습니다. 비기준선 승인 {policy['accepted_non_baseline']}개, MiniRocket 사용 {policy['fallback_count']}개입니다.</p><table><tr><th>승인 방법</th><th>개수</th></tr>{method_rows}</table><table><tr><th>MiniRocket 사용 이유</th><th>개수</th></tr>{reason_rows}</table>
<h2>5. 가장 큰 잔여 손실</h2><table><tr><th>데이터셋</th><th>선택</th><th>AUC</th><th>최적 후보와 차이</th><th>이유</th></tr>{worst_rows}</table>
<h2>6. 진행 체크</h2><label><input type="checkbox" checked disabled> Exp476 단일 변경 설계 승인</label><label><input type="checkbox" checked disabled> 코드맵과 Exp475 해시 확인</label><label><input type="checkbox" checked disabled> ZF 최솟값 보호 테스트</label><label><input type="checkbox" checked disabled> 기존 962개 특징 재사용</label><label><input type="checkbox" checked disabled> 120개 가족 분리 검증</label><label><input type="checkbox" {checked} disabled> 사전 통과 기준 충족</label><label><input type="checkbox" disabled> 공식 파이프라인 반영 (별도 승인 전 미수행)</label>
<h2>7. 재현 정보</h2><ul><li>Exp475 결과 SHA-256: <code>{EXPECTED_EXP475_SHA256_ANTIGRAVITY}</code></li><li>새 특징·점수 계산 없음</li><li>MLX: 새 연산 단계가 없어 평가 대상 아님</li><li>생성 시각: {escape(str(result['generated_at_utc']))}</li></ul></body></html>"""


def run_antigravity() -> dict[str, object]:
    prior_bytes = EXP475_RESULT_ANTIGRAVITY.read_bytes()
    prior_sha = sha256(prior_bytes).hexdigest()
    if prior_sha != EXPECTED_EXP475_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp475 result changed; refusing to run Exp476")
    prior = json.loads(prior_bytes)
    features = json.loads(FEATURE_CACHE_ANTIGRAVITY.read_text(encoding="utf-8"))
    records = load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    result = evaluate_minimum_zf_family_blocked_antigravity(
        attach_features_antigravity(records, features), policy_kwargs=POLICY_KWARGS_ANTIGRAVITY
    )
    current_decisions = result["decisions"]
    prior_decisions = prior["decisions"]
    assert isinstance(current_decisions, list) and isinstance(prior_decisions, list)
    result["selection_changes_from_exp475"] = count_selection_changes_antigravity(
        current_decisions, prior_decisions
    )
    result["gates"] = build_gates_antigravity(result)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["frozen_exp475_result_sha256"] = prior_sha
    result["exp475_policy_summary"] = prior["policy"]
    result["acceleration_assessment"] = {
        "selected_backend": "reuse_frozen_inputs",
        "mlx_eligible": False,
        "reason": "Exp476 introduces no new feature or score-source computation.",
    }
    _write_json_antigravity(RESULT_PATH_ANTIGRAVITY, result)
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        render_dashboard_antigravity(result, prior["policy"]), encoding="utf-8"
    )
    print(json.dumps({"baseline":result["baseline"],"exp475":prior["policy"],"exp476":result["policy"],"minimum_zf_rejections":result["minimum_zf_rejection_count"],"selection_changes":result["selection_changes_from_exp475"],"gates":result["gates"]},ensure_ascii=False,indent=2))
    return result


if __name__ == "__main__":
    run_antigravity()
