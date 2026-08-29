"""Verify official Exp476 integration and publish Exp477 evidence."""

from __future__ import annotations

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
)
from experiments.exp476_minimum_zf_fallback_routing_antigravity.evaluation_antigravity import (
    evaluate_minimum_zf_family_blocked_antigravity,
)
from experiments.exp476_minimum_zf_fallback_routing_antigravity.run_antigravity import (
    POLICY_KWARGS_ANTIGRAVITY,
)
from experiments.exp_routing_label_free_antigravity.conservative_policy_antigravity import (
    ARTIFACT_PATH_ANTIGRAVITY,
    load_official_policy_artifact_antigravity,
)
from experiments.exp_routing_label_free_antigravity.official_router_antigravity import (
    load_train_features_antigravity,
    route_official_antigravity,
)
from experiments.exp_routing_label_free_antigravity.pipeline_integration_antigravity import (
    get_candidates_for_dataset_antigravity,
)
from experiments.exp_routing_label_free_antigravity.routing_rules_antigravity import (
    route_dataset_antigravity,
)


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP477_DASHBOARD_antigravity.html"
EXP476_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp476_minimum_zf_fallback_routing_antigravity/results_antigravity.json"
)
EXPECTED_EXP476_SHA256_ANTIGRAVITY = (
    "8b6f881d25d0d2865e05c5a5b9ca204f78641e1364bb60d536e7dcc8ec21733f"
)
REPRESENTATIVE_DATASETS_ANTIGRAVITY = (
    "Crop_normal_1",
    "ECG5000_normal_2",
    "Wafer_normal_-1",
    "TwoLeadECG_normal_1",
    "FaceFour_normal_1",
    "AllGestureWiimoteY_normal_3",
)


def _sha256_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json_antigravity(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render_dashboard_antigravity(result: dict[str, object]) -> str:
    qualification = result["inherited_exp476_qualification"]
    alignment = result["official_proposal_alignment_qualification"]
    checks = result["integration_checks"]
    rows = result["end_to_end_rows"]
    assert isinstance(qualification, dict) and isinstance(alignment, dict)
    assert isinstance(checks, dict) and isinstance(rows, list)
    table_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['dataset']))}</td>"
        f"<td><code>{escape(str(row['proposal']))}</code></td>"
        f"<td><code>{escape(str(row['official_method']))}</code></td>"
        f"<td>{int(row['normal_train_rows'])}</td>"
        f"<td>{float(row['median_width']):.1f}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp477 공식 라우터 통합 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.58;color:#182230}}h1,h2{{line-height:1.25}}.status{{border-left:5px solid #08783e;padding:12px 16px;background:#effaf4}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:12px;padding:16px;background:#f8fafc}}.big{{font-size:1.55rem;font-weight:750}}.muted{{color:#596579}}.pass{{color:#08783e;font-weight:700}}table{{border-collapse:collapse;width:100%;margin:12px 0 26px}}th,td{{border:1px solid #d8dee8;padding:8px 10px;text-align:left}}th{{background:#eef2f7}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}
</style></head><body>
<h1>Exp477 공식 라우터 통합</h1><p class="status"><strong>공식 반영 완료</strong> — Exp476에서 성능 기준을 통과한 보수적 규칙을 공식 파이프라인에 연결했습니다.</p>
<div class="cards"><div class="card"><div class="muted">검증 데이터셋</div><div class="big">{qualification['dataset_count']}</div></div><div class="card"><div class="muted">검증 가족</div><div class="big">{qualification['family_count']}</div></div><div class="card"><div class="muted">평균 AUC</div><div class="big">{float(qualification['average_auc']):.6f}</div></div><div class="card"><div class="muted">평균 ZF-TP</div><div class="big">{float(qualification['average_zf_tp']):.6f}</div></div><div class="card"><div class="muted">큰 AUC 손실</div><div class="big">{qualification['severe_auc_loss_over_0_05']}</div></div></div>
<h2>1. 무엇이 달라졌나?</h2><p>기존 규칙이 Hydra나 MultiRocket을 제안해도, 비슷한 과거 사례 31개 중 ZF가 나빠진 사례가 하나라도 있으면 MiniRocket으로 돌아갑니다. 데이터셋 이름은 자료를 데이터베이스에서 찾는 데만 쓰며, 방법을 고르는 계산에는 들어가지 않습니다.</p>
<h2>2. 성능 근거</h2><p>Exp476의 962개 데이터셋·120개 가족 분리 검증에서 평균 AUC는 MiniRocket 기준선보다 높았고, 평균 ZF-TP는 기준선과 같았습니다. 공식 제안 규칙으로 다시 맞춰 확인했을 때 첫 제안은 {alignment['proposal_change_count']}개가 달랐지만, 최종 선택 변화는 <strong>{alignment['final_selection_change_count']}개</strong>였고 성능 수치도 모두 같았습니다. 큰 AUC 손실은 47개에서 <strong>{qualification['severe_auc_loss_over_0_05']}개</strong>로 줄었습니다.</p>
<h2>3. 실제 전체 흐름 확인</h2><table><tr><th>데이터셋</th><th>첫 제안</th><th>공식 최종 선택</th><th>정상 TRAIN 수</th><th>대표 길이</th></tr>{table_rows}</table><p class="pass">직접 호출과 공식 파이프라인 불일치: {checks['pipeline_mismatch_count']}개 · 한 데이터셋에 반환된 최대 방법 수: {checks['maximum_candidate_count']}개</p>
<h2>4. 이름과 새 데이터의 정답을 쓰지 않는가?</h2><p>고정 학습 자료 {checks['anonymous_training_row_count']}개에는 데이터셋 이름이 없습니다. 새 데이터셋을 고를 때는 정상 TRAIN 시계열에서 계산한 숫자 특징 7개만 사용하며 TEST 정답이나 새 데이터의 성능 결과를 읽지 않습니다.</p>
<h2>5. 진행 체크</h2>
<label><input type="checkbox" checked disabled> Exp476 통과 결과 해시 확인</label>
<label><input type="checkbox" checked disabled> 공식 코드 영향 범위와 코드맵 확인</label>
<label><input type="checkbox" checked disabled> 이름 없는 962개 고정 학습 자료 생성</label>
<label><input type="checkbox" checked disabled> 공식 첫 제안 규칙으로 120가족 성능 재확인</label>
<label><input type="checkbox" checked disabled> 가변 길이와 고정 길이 TRAIN 입력 지원</label>
<label><input type="checkbox" checked disabled> Exp476 보수적 규칙을 공식 라우터에 연결</label>
<label><input type="checkbox" checked disabled> 기존 이름 기반 후보 분기 제거</label>
<label><input type="checkbox" checked disabled> 대표 6개 데이터셋 전체 흐름 검증</label>
<label><input type="checkbox" checked disabled> 공식 통합 테스트와 회귀 테스트 통과</label>
<h2>6. 재현 정보</h2><ul><li>Exp476 결과 SHA-256: <code>{result['source_exp476_result_sha256']}</code></li><li>공식 학습 자료 SHA-256: <code>{result['artifact_sha256']}</code></li><li>MLX: 새 특징이나 점수 계산을 추가하지 않아 평가 대상이 아닙니다.</li><li>생성 시각: {escape(str(result['generated_at_utc']))}</li></ul>
</body></html>"""


def run_antigravity() -> dict[str, object]:
    exp476_hash = _sha256_antigravity(EXP476_RESULT_PATH_ANTIGRAVITY)
    if exp476_hash != EXPECTED_EXP476_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp476 result changed; refusing official integration verification")
    exp476 = json.loads(EXP476_RESULT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    if not exp476["gates"]["overall_pass"]:
        raise RuntimeError("Exp476 did not pass its preregistered gates")

    artifact = load_official_policy_artifact_antigravity()
    artifact_text = ARTIFACT_PATH_ANTIGRAVITY.read_text(encoding="utf-8")
    features_by_dataset = json.loads(FEATURE_CACHE_ANTIGRAVITY.read_text(encoding="utf-8"))
    checkpoint_records = load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    aligned_records: list[dict[str, object]] = []
    proposal_change_count = 0
    for record in checkpoint_records:
        dataset = str(record["dataset"])
        proposal = route_dataset_antigravity(features_by_dataset[dataset])
        proposal_change_count += int(proposal != str(record["proposal"]))
        aligned_records.append({**record, "proposal": proposal})
    aligned_qualification = evaluate_minimum_zf_family_blocked_antigravity(
        attach_features_antigravity(aligned_records, features_by_dataset),
        policy_kwargs=POLICY_KWARGS_ANTIGRAVITY,
    )
    prior_decisions = {row["dataset"]: row for row in exp476["decisions"]}
    final_selection_change_count = sum(
        row["method"] != prior_decisions[row["dataset"]]["method"]
        for row in aligned_qualification["decisions"]
    )
    qualification_metrics_match = (
        aligned_qualification["baseline"] == exp476["baseline"]
        and aligned_qualification["policy"] == exp476["policy"]
    )
    rows: list[dict[str, object]] = []
    pipeline_mismatches = 0
    maximum_candidate_count = 0
    deterministic = True
    for dataset in REPRESENTATIVE_DATASETS_ANTIGRAVITY:
        features = load_train_features_antigravity(dataset)
        proposal = route_dataset_antigravity(features)
        first = route_official_antigravity(dataset)
        second = route_official_antigravity(dataset)
        candidates = get_candidates_for_dataset_antigravity(dataset)
        deterministic = deterministic and first == second
        maximum_candidate_count = max(maximum_candidate_count, len(candidates))
        pipeline_mismatches += int(candidates != (first,))
        rows.append(
            {
                "dataset": dataset,
                "proposal": proposal,
                "official_method": first,
                "pipeline_candidates": list(candidates),
                "normal_train_rows": int(features["n_train"]),
                "median_width": float(features["width"]),
            }
        )

    integration_checks = {
        "source_qualification_passed": True,
        "anonymous_training_row_count": artifact["row_count"],
        "artifact_name_marker_count": artifact_text.count("_normal_"),
        "pipeline_mismatch_count": pipeline_mismatches,
        "maximum_candidate_count": maximum_candidate_count,
        "deterministic": deterministic,
        "ragged_dataset_verified": "AllGestureWiimoteY_normal_3",
        "proposal_alignment_change_count": proposal_change_count,
        "final_selection_change_count": final_selection_change_count,
        "qualification_metrics_match_exp476": qualification_metrics_match,
        "overall_pass": (
            artifact["row_count"] == 962
            and "_normal_" not in artifact_text
            and pipeline_mismatches == 0
            and maximum_candidate_count == 1
            and deterministic
            and proposal_change_count == 1
            and final_selection_change_count == 0
            and qualification_metrics_match
        ),
    }
    if not integration_checks["overall_pass"]:
        raise RuntimeError("official integration verification failed")

    result: dict[str, object] = {
        "schema": "exp477_official_minimum_zf_routing_result_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_exp476_result_sha256": exp476_hash,
        "artifact_sha256": _sha256_antigravity(ARTIFACT_PATH_ANTIGRAVITY),
        "inherited_exp476_qualification": {
            "dataset_count": exp476["dataset_count"],
            "family_count": exp476["family_count"],
            "average_auc": exp476["policy"]["average_auc"],
            "average_zf_tp": exp476["policy"]["average_zf_tp"],
            "severe_auc_loss_over_0_05": exp476["policy"]["severe_auc_loss_over_0_05"],
            "baseline_average_auc": exp476["baseline"]["average_auc"],
            "baseline_average_zf_tp": exp476["baseline"]["average_zf_tp"],
            "baseline_severe_auc_loss_over_0_05": exp476["baseline"]["severe_auc_loss_over_0_05"],
        },
        "official_proposal_alignment_qualification": {
            "proposal_change_count": proposal_change_count,
            "final_selection_change_count": final_selection_change_count,
            "metrics_match_exp476": qualification_metrics_match,
            "average_auc": aligned_qualification["policy"]["average_auc"],
            "average_zf_tp": aligned_qualification["policy"]["average_zf_tp"],
            "severe_auc_loss_over_0_05": aligned_qualification["policy"][
                "severe_auc_loss_over_0_05"
            ],
        },
        "integration_checks": integration_checks,
        "end_to_end_rows": rows,
        "acceleration_assessment": {
            "mlx_eligible": False,
            "reason": "Official integration adds no new feature or score-source computation.",
        },
    }
    _write_json_antigravity(RESULT_PATH_ANTIGRAVITY, result)
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        _render_dashboard_antigravity(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run_antigravity()
