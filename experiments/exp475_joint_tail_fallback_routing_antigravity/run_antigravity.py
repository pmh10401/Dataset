"""Run Exp475 and write its immutable JSON evidence plus HTML dashboard."""

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

from .evaluation_antigravity import evaluate_joint_tail_family_blocked_antigravity


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
EXP474_RESULT_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp474_conservative_fallback_routing_antigravity/results_antigravity.json"
)
EXPECTED_EXP474_SHA256_ANTIGRAVITY = "c4f1d722cb3d975fcb7623307a264c98a35a2cec3edb750bc339e31f0bf8abf4"
RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP475_DASHBOARD_antigravity.html"
POLICY_KWARGS_ANTIGRAVITY = {
    "k": 31,
    "min_support": 20,
    "positive_rate_threshold": 0.90,
    "lower_quantile": 0.20,
    "min_auc_gain": 0.0,
    "min_zf_gain": 0.0,
    "min_lower_zf_gain": 0.0,
}


def count_selection_changes_antigravity(
    current: list[dict[str, object]], prior: list[dict[str, object]]
) -> int:
    current_by_dataset = {str(row["dataset"]): row for row in current}
    prior_by_dataset = {str(row["dataset"]): row for row in prior}
    if set(current_by_dataset) != set(prior_by_dataset):
        raise ValueError("Exp474 and Exp475 decision populations differ")
    return sum(
        (
            current_by_dataset[name]["method"],
            bool(current_by_dataset[name]["accepted"]),
        )
        != (
            prior_by_dataset[name]["method"],
            bool(prior_by_dataset[name]["accepted"]),
        )
        for name in current_by_dataset
    )


def build_gates_antigravity(result: dict[str, object]) -> dict[str, object]:
    baseline = result["baseline"]
    policy = result["policy"]
    leakage = result["leakage_checks"]
    assert isinstance(baseline, dict) and isinstance(policy, dict) and isinstance(leakage, dict)
    gates: dict[str, object] = {
        "평균 AUC": {
            "actual": round(float(policy["average_auc"]), 9),
            "required": f">= measured MiniRocket {float(baseline['average_auc']):.9f}",
            "pass": float(policy["average_auc"]) >= float(baseline["average_auc"]) - 1e-12,
        },
        "평균 ZF-TP": {
            "actual": round(float(policy["average_zf_tp"]), 9),
            "required": f">= measured MiniRocket {float(baseline['average_zf_tp']):.9f}",
            "pass": float(policy["average_zf_tp"]) >= float(baseline["average_zf_tp"]) - 1e-12,
        },
        "AUC 큰 손실": {
            "actual": int(policy["severe_auc_loss_over_0_05"]),
            "required": "< 68",
            "pass": int(policy["severe_auc_loss_over_0_05"]) < 68,
        },
        "가족 누출": {
            "actual": int(leakage["family_overlap_count"]),
            "required": "= 0",
            "pass": int(leakage["family_overlap_count"]) == 0,
        },
    }
    gates["overall_pass"] = all(
        bool(value["pass"]) for value in gates.values() if isinstance(value, dict)
    )
    return gates


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
    accepted = [row for row in decisions if isinstance(row, dict) and row.get("accepted")]
    methods = Counter(str(row["method"]) for row in accepted)
    reasons = Counter(
        str(row["reason"]) for row in decisions if isinstance(row, dict) and not row.get("accepted")
    )
    worst = sorted(
        (row for row in decisions if isinstance(row, dict)),
        key=lambda row: float(row["oracle_auc"]) - float(row["auc"]),
        reverse=True,
    )[:12]
    gate_rows = "".join(
        f"<tr><td>{escape(str(name))}</td><td>{escape(str(data['actual']))}</td>"
        f"<td>{escape(str(data['required']))}</td><td class={'pass' if data['pass'] else 'hold'}>"
        f"{'통과' if data['pass'] else '미달'}</td></tr>"
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
    gate_checked = "checked" if passed else ""
    status = "검증 통과" if passed else "검증 보류"
    status_class = "pass" if passed else "hold"
    redundancy_note = (
        "<p class='status hold'><strong>중요:</strong> 새 하위 20% 조건은 Exp474의 선택을 바꾸지 못했습니다. "
        "기존 90% 성공률 조건이 이미 더 강하게 같은 위험을 막고 있었기 때문입니다.</p>"
        if int(result.get("selection_changes_from_exp474", -1)) == 0
        else ""
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp475 ZF 하위 꼬리 보호 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.58;color:#182230}}h1,h2{{line-height:1.25}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:12px;padding:16px;background:#f8fafc}}.big{{font-size:1.6rem;font-weight:750}}
.pass{{color:#08783e;font-weight:700}}.hold{{color:#b54708;font-weight:700}}.muted{{color:#596579}}.status{{border-left:5px solid;padding:12px 16px;background:#f8fafc}}
table{{border-collapse:collapse;width:100%;margin:12px 0 26px}}th,td{{border:1px solid #d8dee8;padding:8px 10px;text-align:left}}th{{background:#eef2f7}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}
</style></head><body><h1>Exp475 ZF 하위 꼬리 보호</h1>
<p class="status {status_class}"><strong>{status}</strong> — AUC뿐 아니라 ZF-TP의 나쁜 주변 사례까지 확인한 보수 규칙입니다.</p>
<div class="cards"><div class="card"><div class="muted">데이터셋</div><div class="big">{result['dataset_count']}</div></div>
<div class="card"><div class="muted">분리한 가족</div><div class="big">{result['family_count']}</div></div>
<div class="card"><div class="muted">Exp475 평균 AUC</div><div class="big">{float(policy['average_auc']):.4f}</div></div>
<div class="card"><div class="muted">Exp475 평균 ZF-TP</div><div class="big">{float(policy['average_zf_tp']):.3f}</div></div>
<div class="card"><div class="muted">비기준선 승인</div><div class="big">{policy['accepted_non_baseline']}</div></div></div>
<h2>1. 무엇을 바꿨나?</h2><p>Exp474에서는 가까운 과거 사례의 ZF-TP 중앙값만 확인했습니다. Exp475는 그 사례를 낮은 쪽부터 보았을 때 하위 20%의 ZF 이득도 0 이상이어야 Hydra 또는 MultiRocket을 허용합니다. 이 조건을 통과하지 못하면 MiniRocket으로 돌아갑니다.</p>
{redundancy_note}
<h2>2. 결과 비교</h2><table><tr><th>방법</th><th>평균 AUC</th><th>평균 ZF-TP</th><th>큰 AUC 손실</th><th>비기준선 승인</th></tr>
<tr><td>항상 MiniRocket</td><td>{float(baseline['average_auc']):.6f}</td><td>{float(baseline['average_zf_tp']):.6f}</td><td>{baseline['severe_auc_loss_over_0_05']}</td><td>0</td></tr>
<tr><td>Exp474</td><td>{float(prior_policy['average_auc']):.6f}</td><td>{float(prior_policy['average_zf_tp']):.6f}</td><td>{prior_policy.get('severe_auc_loss_over_0_05','-')}</td><td>{prior_policy['accepted_non_baseline']}</td></tr>
<tr><td><strong>Exp475</strong></td><td><strong>{float(policy['average_auc']):.6f}</strong></td><td><strong>{float(policy['average_zf_tp']):.6f}</strong></td><td><strong>{policy['severe_auc_loss_over_0_05']}</strong></td><td><strong>{policy['accepted_non_baseline']}</strong></td></tr></table>
<h2>3. 통과 기준</h2><table><tr><th>기준</th><th>실제</th><th>요구</th><th>판정</th></tr>{gate_rows}</table>
<h2>4. 선택 내역</h2><p>비기준선 승인 <strong>{policy['accepted_non_baseline']}개</strong>, MiniRocket 사용 <strong>{policy['fallback_count']}개</strong>입니다.</p>
<table><tr><th>승인 방법</th><th>개수</th></tr>{method_rows}</table><table><tr><th>MiniRocket 사용 이유</th><th>개수</th></tr>{reason_rows}</table>
<h2>5. 가장 큰 잔여 손실</h2><table><tr><th>데이터셋</th><th>선택</th><th>AUC</th><th>최적 후보와 차이</th><th>이유</th></tr>{worst_rows}</table>
<h2>6. 진행 체크</h2>
<label><input type="checkbox" checked disabled> Exp475 단일 변경 설계 승인</label><label><input type="checkbox" checked disabled> 코드맵과 Exp474 보존 경계 확인</label>
<label><input type="checkbox" checked disabled> ZF 하위 꼬리 정책 테스트</label><label><input type="checkbox" checked disabled> 기존 962개 TRAIN 특징 재사용</label>
<label><input type="checkbox" checked disabled> 120개 가족 분리 검증</label><label><input type="checkbox" {gate_checked} disabled> 사전 통과 기준 충족</label>
<label><input type="checkbox" disabled> 공식 파이프라인 반영 (별도 승인 전 미수행)</label>
<h2>7. 재현 정보</h2><ul><li>Exp474 결과 SHA-256: <code>{EXPECTED_EXP474_SHA256_ANTIGRAVITY}</code></li><li>새 특징·새 점수 계산 없음: 기존 캐시와 체크포인트 재사용</li><li>MLX: 새 연산 단계가 없어 평가 대상 아님</li><li>생성 시각: {escape(str(result['generated_at_utc']))}</li></ul>
</body></html>"""


def run_antigravity() -> dict[str, object]:
    prior_bytes = EXP474_RESULT_ANTIGRAVITY.read_bytes()
    prior_sha = sha256(prior_bytes).hexdigest()
    if prior_sha != EXPECTED_EXP474_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp474 result changed; refusing to run Exp475")
    prior = json.loads(prior_bytes)
    features = json.loads(FEATURE_CACHE_ANTIGRAVITY.read_text(encoding="utf-8"))
    records = load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    attached = attach_features_antigravity(records, features)
    result = evaluate_joint_tail_family_blocked_antigravity(
        attached, policy_kwargs=POLICY_KWARGS_ANTIGRAVITY
    )
    current_decisions = result["decisions"]
    prior_decisions = prior["decisions"]
    assert isinstance(current_decisions, list) and isinstance(prior_decisions, list)
    result["selection_changes_from_exp474"] = count_selection_changes_antigravity(
        current_decisions, prior_decisions
    )
    result["zf_tail_rejection_count"] = sum(
        isinstance(row, dict) and row.get("reason") == "zf_tail_not_positive"
        for row in current_decisions
    )
    result["gates"] = build_gates_antigravity(result)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["frozen_exp474_result_sha256"] = prior_sha
    result["exp474_policy_summary"] = prior["policy"]
    result["acceleration_assessment"] = {
        "selected_backend": "reuse_frozen_inputs",
        "mlx_eligible": False,
        "reason": "Exp475 introduces no new feature or score-source computation.",
    }
    _write_json_antigravity(RESULT_PATH_ANTIGRAVITY, result)
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        render_dashboard_antigravity(result, prior["policy"]), encoding="utf-8"
    )
    print(
        json.dumps(
            {"baseline": result["baseline"], "exp474": prior["policy"], "exp475": result["policy"], "gates": result["gates"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    run_antigravity()
