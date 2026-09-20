"""Run Exp474 qualification and write its JSON evidence and HTML dashboard."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .evaluation_antigravity import (
    attach_features_antigravity,
    evaluate_family_blocked_antigravity,
    load_checkpoint_records_antigravity,
)
from .policy_antigravity import FEATURE_KEYS_ANTIGRAVITY
from .ragged_features_antigravity import extract_ragged_routing_features_antigravity


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
CHECKPOINT_DIR_ANTIGRAVITY = ROOT_ANTIGRAVITY / "experiments/exp_routing_label_free_antigravity/checkpoints"
DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
FEATURE_CACHE_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "train_features_antigravity.json"
RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP474_DASHBOARD_antigravity.html"

POLICY_KWARGS_ANTIGRAVITY = {
    "k": 31,
    "min_support": 20,
    "positive_rate_threshold": 0.90,
    "lower_quantile": 0.20,
    "min_auc_gain": 0.0,
    "min_zf_gain": 0.0,
}

_TRAIN_ROWS_SQL_ANTIGRAVITY = (
    "SELECT i.values_blob FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id "
    "WHERE d.name=? AND i.label='0' AND i.split='TRAIN' ORDER BY i.instance_index"
)


def _json_safe_antigravity(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe_antigravity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_antigravity(item) for item in value]
    return value


def _write_json_antigravity(path: Path, payload: object) -> None:
    temporary = path.with_name(path.stem + "_temporary_antigravity" + path.suffix)
    temporary.write_text(
        json.dumps(_json_safe_antigravity(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_or_build_features_antigravity(records: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    if FEATURE_CACHE_ANTIGRAVITY.exists():
        cached = json.loads(FEATURE_CACHE_ANTIGRAVITY.read_text(encoding="utf-8"))
    else:
        cached = {}
    wanted = [str(record["dataset"]) for record in records]
    missing = [dataset for dataset in wanted if dataset not in cached]
    if not missing:
        return {dataset: cached[dataset] for dataset in wanted}

    connection = sqlite3.connect(f"file:{DATABASE_ANTIGRAVITY}?mode=ro", uri=True)
    try:
        for index, dataset in enumerate(missing, start=1):
            blobs = connection.execute(_TRAIN_ROWS_SQL_ANTIGRAVITY, (dataset,)).fetchall()
            rows = [np.frombuffer(blob[0], dtype="<f4").astype(np.float64) for blob in blobs]
            raw = extract_ragged_routing_features_antigravity(rows)
            features = {key: float(raw[key]) for key in FEATURE_KEYS_ANTIGRAVITY}
            if not all(math.isfinite(value) for value in features.values()):
                raise ValueError(f"non-finite TRAIN feature for {dataset}")
            cached[dataset] = features
            if index % 25 == 0 or index == len(missing):
                _write_json_antigravity(FEATURE_CACHE_ANTIGRAVITY, cached)
                print(f"features {len(cached)}/{len(wanted)}")
    finally:
        connection.close()
    return {dataset: cached[dataset] for dataset in wanted}


def _format_float_antigravity(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def render_dashboard_antigravity(result: dict[str, object]) -> str:
    baseline = result["baseline"]
    policy = result["policy"]
    gates = result["gates"]
    assert isinstance(baseline, dict) and isinstance(policy, dict) and isinstance(gates, dict)
    decisions = result["decisions"]
    assert isinstance(decisions, list)
    passed = bool(gates["overall_pass"])
    status_class = "pass" if passed else "hold"
    status_text = "검증 통과" if passed else "검증 보류"
    accepted = [row for row in decisions if isinstance(row, dict) and row.get("accepted")]
    method_counts = Counter(str(row["method"]) for row in accepted)
    reasons = Counter(str(row["reason"]) for row in decisions if isinstance(row, dict) and not row.get("accepted"))
    worst = sorted(
        (row for row in decisions if isinstance(row, dict)),
        key=lambda row: float(row["oracle_auc"]) - float(row["auc"]),
        reverse=True,
    )[:15]

    gate_rows = "".join(
        f"<tr><td>{escape(str(name))}</td><td>{escape(str(data['actual']))}</td>"
        f"<td>{escape(str(data['required']))}</td><td class={'pass' if data['pass'] else 'hold'}>"
        f"{'통과' if data['pass'] else '미달'}</td></tr>"
        for name, data in gates.items()
        if name != "overall_pass" and isinstance(data, dict)
    )
    method_rows = "".join(
        f"<tr><td><code>{escape(method)}</code></td><td>{count}</td></tr>"
        for method, count in sorted(method_counts.items())
    ) or "<tr><td colspan='2'>승인된 비기준선 선택 없음</td></tr>"
    reason_rows = "".join(
        f"<tr><td>{escape(reason)}</td><td>{count}</td></tr>" for reason, count in reasons.most_common()
    )
    worst_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['dataset']))}</td><td><code>{escape(str(row['method']))}</code></td>"
        f"<td>{float(row['auc']):.4f}</td><td>{float(row['oracle_auc']) - float(row['auc']):.4f}</td>"
        f"<td>{escape(str(row['reason']))}</td></tr>"
        for row in worst
    )
    promotion_checkbox = "checked" if passed else ""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp474 보수적 라우팅 대시보드</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.58;color:#182230}}
h1,h2{{line-height:1.25}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.card{{border:1px solid #d8dee8;border-radius:12px;padding:16px;background:#f8fafc}} .big{{font-size:1.65rem;font-weight:750}}
.pass{{color:#08783e;font-weight:700}} .hold{{color:#b54708;font-weight:700}} .muted{{color:#596579}}
table{{border-collapse:collapse;width:100%;margin:12px 0 26px}}th,td{{border:1px solid #d8dee8;padding:8px 10px;text-align:left}}th{{background:#eef2f7}}
code{{background:#eef2f7;padding:2px 5px;border-radius:4px}} .status{{border-left:5px solid;padding:12px 16px;background:#f8fafc}}
label{{display:block;margin:8px 0}} input{{margin-right:8px}}
</style></head><body>
<h1>Exp474 보수적 라우팅</h1>
<p class="status {status_class}"><strong>{status_text}</strong> — 확신이 낮으면 MiniRocket으로 돌아가는 규칙을 가족 단위로 분리해 검증했습니다.</p>
<div class="cards">
<div class="card"><div class="muted">검증 데이터셋</div><div class="big">{result['dataset_count']}</div></div>
<div class="card"><div class="muted">완전히 분리한 가족</div><div class="big">{result['family_count']}</div></div>
<div class="card"><div class="muted">Exp474 평균 AUC</div><div class="big">{_format_float_antigravity(policy['average_auc'],4)}</div></div>
<div class="card"><div class="muted">Exp474 평균 ZF-TP</div><div class="big">{_format_float_antigravity(policy['average_zf_tp'],2)}</div></div>
<div class="card"><div class="muted">비기준선 승인</div><div class="big">{policy['accepted_non_baseline']}</div></div>
</div>

<h2>1. 쉬운 설명</h2>
<p>기본 선택은 MiniRocket입니다. 기존 라우터가 Hydra나 MultiRocket을 추천하더라도, TRAIN 데이터의 모양이 과거의 성공 사례와 충분히 비슷하고 그 주변 사례 대부분이 AUC와 ZF-TP를 함께 지켰을 때만 추천을 받아들입니다. 조금이라도 불확실하면 MiniRocket을 사용합니다.</p>
<p>데이터셋 이름은 가족을 나누는 시험용으로만 사용했습니다. 실제 선택 계산에는 이름이나 TEST 결과가 들어가지 않았습니다.</p>

<h2>2. 기준선과 비교</h2>
<table><tr><th>방법</th><th>평균 AUC</th><th>평균 ZF-TP</th><th>AUC 큰 손실 &gt; 0.05</th></tr>
<tr><td>항상 MiniRocket</td><td>{_format_float_antigravity(baseline['average_auc'])}</td><td>{_format_float_antigravity(baseline['average_zf_tp'],3)}</td><td>{baseline['severe_auc_loss_over_0_05']}</td></tr>
<tr><td><strong>Exp474 보수적 라우터</strong></td><td><strong>{_format_float_antigravity(policy['average_auc'])}</strong></td><td><strong>{_format_float_antigravity(policy['average_zf_tp'],3)}</strong></td><td><strong>{policy['severe_auc_loss_over_0_05']}</strong></td></tr>
<tr><td>이전 라우터</td><td>0.976637</td><td>1.729</td><td>68</td></tr></table>

<h2>3. 통과 기준</h2><table><tr><th>기준</th><th>실제</th><th>요구</th><th>판정</th></tr>{gate_rows}</table>

<h2>4. 선택 결과</h2>
<p>비기준선 추천을 받아들인 데이터셋은 <strong>{policy['accepted_non_baseline']}개</strong>, MiniRocket을 사용한 데이터셋은 <strong>{policy['fallback_count']}개</strong>입니다.</p>
<table><tr><th>승인된 방법</th><th>개수</th></tr>{method_rows}</table>
<table><tr><th>MiniRocket으로 돌아간 이유</th><th>개수</th></tr>{reason_rows}</table>

<h2>5. 가장 큰 잔여 손실</h2><table><tr><th>데이터셋</th><th>최종 선택</th><th>AUC</th><th>최적 후보와 차이</th><th>판단 이유</th></tr>{worst_rows}</table>

<h2>6. 진행 체크</h2>
<label><input type="checkbox" checked disabled> 설계 승인</label>
<label><input type="checkbox" checked disabled> 코드맵과 영향 범위 확인</label>
<label><input type="checkbox" checked disabled> 정책 단위 테스트</label>
<label><input type="checkbox" checked disabled> {result['dataset_count']}개 TRAIN 특징 생성</label>
<label><input type="checkbox" checked disabled> 가족 분리 검증</label>
<label><input type="checkbox" {promotion_checkbox} disabled> 사전 통과 기준 충족</label>
<label><input type="checkbox" disabled> 공식 파이프라인 반영 (별도 승인 전 미수행)</label>

<h2>7. 재현 정보</h2>
<ul><li>정책: 가까운 사례 31개, 최소 지지 20개, 동시 비열화 방지 성공률 90% 이상, AUC 하위 20% 이득 0 이상</li>
<li>가속 판단: 기존 NumPy 특징 정의를 정확히 보존할 수 없어 MLX 부적격, CPU 사용</li>
<li>생성 시각: {escape(str(result['generated_at_utc']))}</li>
<li>결과 파일: <code>experiments/exp474_conservative_fallback_routing_antigravity/results_antigravity.json</code></li></ul>
</body></html>"""


def run_antigravity() -> dict[str, object]:
    records = load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    print(f"valid checkpoint records {len(records)}")
    features = load_or_build_features_antigravity(records)
    attached = attach_features_antigravity(records, features)
    result = evaluate_family_blocked_antigravity(attached, policy_kwargs=POLICY_KWARGS_ANTIGRAVITY)
    baseline = result["baseline"]
    policy = result["policy"]
    leakage = result["leakage_checks"]
    assert isinstance(baseline, dict) and isinstance(policy, dict) and isinstance(leakage, dict)
    gates = {
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
    gates["overall_pass"] = all(bool(value["pass"]) for value in gates.values())
    result["gates"] = gates
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["acceleration_assessment"] = {
        "selected_backend": "cpu_numpy",
        "mlx_eligible": False,
        "reason": "Frozen NumPy FFT, skewness, and kurtosis feature definitions cannot be reproduced exactly in MLX without changing the extractor.",
    }
    _write_json_antigravity(RESULT_PATH_ANTIGRAVITY, result)
    DASHBOARD_PATH_ANTIGRAVITY.write_text(render_dashboard_antigravity(result), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "policy": policy, "gates": gates}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run_antigravity()
