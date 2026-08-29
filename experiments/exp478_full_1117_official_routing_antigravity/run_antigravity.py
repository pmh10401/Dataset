"""Run the official router across all 1117 eligible mapped datasets."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    load_checkpoint_records_antigravity,
)
from experiments.exp474_conservative_fallback_routing_antigravity.policy_antigravity import (
    BASELINE_METHOD_ANTIGRAVITY,
)
from experiments.exp474_conservative_fallback_routing_antigravity.run_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
)
from experiments.exp_routing_label_free_antigravity.conservative_policy_antigravity import (
    route_features_official_antigravity,
)
from experiments.exp_routing_label_free_antigravity.official_router_antigravity import (
    load_train_features_antigravity,
    route_official_antigravity,
)
from experiments.exp_routing_label_free_antigravity.pipeline_integration_antigravity import (
    _ALL_METHODS_ANTIGRAVITY,
    get_candidates_for_dataset_antigravity,
)
from experiments.exp_routing_label_free_antigravity.routing_rules_antigravity import (
    route_dataset_antigravity,
)


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
EXP477_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp477_official_minimum_zf_routing_antigravity/results_antigravity.json"
)
CHECKPOINT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "checkpoint_antigravity.json"
RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP478_DASHBOARD_antigravity.html"
EXPECTED_EXP477_SHA256_ANTIGRAVITY = (
    "e794cc506d7579fc63e2fe96ccfebc794a330975f6a4375889419263bc877949"
)
PROGRESS_INTERVAL_ANTIGRAVITY = 25
OFFICIAL_SOURCE_PATHS_ANTIGRAVITY = {
    "conservative_policy": ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/conservative_policy_antigravity.py",
    "official_policy_artifact": ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/official_exp476_policy_artifact_antigravity.json",
    "official_router": ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/official_router_antigravity.py",
    "pipeline_integration": ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/pipeline_integration_antigravity.py",
    "ragged_features": ROOT_ANTIGRAVITY
    / "experiments/exp474_conservative_fallback_routing_antigravity/ragged_features_antigravity.py",
    "routing_rules": ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/routing_rules_antigravity.py",
}

_POPULATION_SQL_ANTIGRAVITY = (
    "SELECT d.name, COUNT(i.id) FROM datasets AS d "
    "LEFT JOIN instances AS i ON i.dataset_id=d.id "
    "AND i.split='TRAIN' AND i.label='0' "
    "GROUP BY d.id, d.name ORDER BY d.name"
)


def _utc_now_antigravity() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json_antigravity(path: Path, payload: object) -> None:
    temporary = path.with_name(path.stem + "_temporary_antigravity" + path.suffix)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _official_source_hashes_antigravity() -> dict[str, str]:
    return {
        name: _sha256_antigravity(path)
        for name, path in OFFICIAL_SOURCE_PATHS_ANTIGRAVITY.items()
    }


def _resume_rows_antigravity(
    checkpoint: Mapping[str, object],
    *,
    population: Sequence[str],
    source_result_hash: str,
    source_hashes: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Reuse rows only when every result-affecting source is identical."""
    if checkpoint.get("source_exp477_result_sha256") != source_result_hash:
        return {}
    if checkpoint.get("source_hashes") != dict(source_hashes):
        return {}
    population_set = set(population)
    return {
        str(row["dataset"]): row
        for row in checkpoint.get("rows", [])  # type: ignore[union-attr]
        if isinstance(row, dict) and row.get("dataset") in population_set
    }


def select_population_antigravity(
    connection: sqlite3.Connection,
) -> tuple[list[str], list[dict[str, object]]]:
    """Return the exact mapped/routable population and explicit exclusions."""
    included: list[str] = []
    exclusions: list[dict[str, object]] = []
    for raw_name, raw_count in connection.execute(_POPULATION_SQL_ANTIGRAVITY):
        name = str(raw_name)
        count = int(raw_count)
        if "_normal_" not in name:
            exclusions.append(
                {
                    "dataset": name,
                    "reason": "normal_class_not_mapped",
                    "normal_train_rows": count,
                }
            )
        elif count < 2:
            exclusions.append(
                {
                    "dataset": name,
                    "reason": "fewer_than_two_normal_train_rows",
                    "normal_train_rows": count,
                }
            )
        else:
            included.append(name)
    return included, exclusions


def _audit_nonfinite_normal_train_antigravity(
    connection: sqlite3.Connection,
    population: Sequence[str],
) -> dict[str, object]:
    """Count rows excluded by the approved no-interpolation finite-row policy."""
    population_set = set(population)
    by_dataset: dict[str, dict[str, int]] = {}
    rows = connection.execute(
        "SELECT d.name, i.values_blob FROM instances AS i "
        "JOIN datasets AS d ON d.id=i.dataset_id "
        "WHERE d.name LIKE '%_normal_%' AND i.split='TRAIN' AND i.label='0' "
        "ORDER BY d.name, i.instance_index"
    )
    for raw_name, blob in rows:
        name = str(raw_name)
        if name not in population_set:
            continue
        nonfinite_values = int(np.sum(~np.isfinite(np.frombuffer(blob, dtype="<f4"))))
        if nonfinite_values:
            counts = by_dataset.setdefault(name, {"excluded_rows": 0, "nonfinite_values": 0})
            counts["excluded_rows"] += 1
            counts["nonfinite_values"] += nonfinite_values
    details = [
        {"dataset": name, **by_dataset[name]} for name in sorted(by_dataset)
    ]
    return {
        "affected_datasets": len(details),
        "excluded_rows": sum(int(row["excluded_rows"]) for row in details),
        "nonfinite_values": sum(int(row["nonfinite_values"]) for row in details),
        "families": sorted({str(row["dataset"]).rsplit("_normal_", 1)[0] for row in details}),
        "strategy": "exclude_nonfinite_rows_without_interpolation",
        "details": details,
    }


def evaluate_dataset_antigravity(dataset: str) -> dict[str, object]:
    """Exercise feature, direct, and pipeline routes for one real dataset."""
    started = perf_counter()
    try:
        features = load_train_features_antigravity(dataset)
        first_feature_method = route_features_official_antigravity(features)
        second_feature_method = route_features_official_antigravity(features)
        direct_method = route_official_antigravity(dataset)
        candidates = get_candidates_for_dataset_antigravity(dataset)
        pipeline_method = candidates[0] if len(candidates) == 1 else None
        valid = (
            first_feature_method in _ALL_METHODS_ANTIGRAVITY
            and direct_method in _ALL_METHODS_ANTIGRAVITY
            and pipeline_method in _ALL_METHODS_ANTIGRAVITY
        )
        return {
            "dataset": dataset,
            "status": "ok" if valid else "invalid_method",
            "proposal": route_dataset_antigravity(features),
            "feature_method": first_feature_method,
            "direct_method": direct_method,
            "pipeline_method": pipeline_method,
            "candidate_count": len(candidates),
            "deterministic": first_feature_method == second_feature_method,
            "pipeline_match": candidates == (direct_method,),
            "normal_train_rows": int(features["n_train"]),
            "median_width": float(features["width"]),
            "elapsed_seconds": perf_counter() - started,
        }
    except Exception as error:  # preserve the full census instead of stopping early
        return {
            "dataset": dataset,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": perf_counter() - started,
        }


def _summarise_routing_antigravity(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    success = [row for row in rows if row.get("status") == "ok"]
    failures = [row for row in rows if row.get("status") != "ok"]
    return {
        "completed": len(rows),
        "success": len(success),
        "failures": len(failures),
        "feature_direct_mismatches": sum(
            row.get("feature_method") != row.get("direct_method") for row in success
        ),
        "pipeline_mismatches": sum(not bool(row.get("pipeline_match")) for row in success),
        "nondeterministic": sum(not bool(row.get("deterministic")) for row in success),
        "non_singleton": sum(int(row.get("candidate_count", 0)) != 1 for row in success),
        "method_counts": dict(
            sorted(Counter(str(row["direct_method"]) for row in success).items())
        ),
        "elapsed_seconds": float(sum(float(row["elapsed_seconds"]) for row in rows)),
        "failure_rows": failures,
    }


def _overall_pass_antigravity(
    routing: Mapping[str, object],
    score_replay: Mapping[str, object],
    nonfinite_handling: Mapping[str, object],
) -> bool:
    """Apply the frozen Exp478 completion contract in one place."""
    return (
        int(routing.get("success", 0)) == 1117
        and int(routing.get("failures", 1)) == 0
        and int(routing.get("feature_direct_mismatches", 1)) == 0
        and int(routing.get("pipeline_mismatches", 1)) == 0
        and int(routing.get("nondeterministic", 1)) == 0
        and int(routing.get("non_singleton", 1)) == 0
        and int(score_replay.get("scored", 0)) == 962
        and int(score_replay.get("routed_without_scores", 0)) == 155
        and int(nonfinite_handling.get("affected_datasets", 0)) == 19
        and int(nonfinite_handling.get("excluded_rows", 0)) == 164
        and int(nonfinite_handling.get("nonfinite_values", 0)) == 983
    )


def _summarise_score_replay_antigravity(
    route_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    score_records = {
        str(row["dataset"]): row
        for row in load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    }
    replay_rows: list[dict[str, object]] = []
    for route in route_rows:
        dataset = str(route["dataset"])
        if route.get("status") != "ok" or dataset not in score_records:
            continue
        method = str(route["direct_method"])
        scores = score_records[dataset]["scores"]
        assert isinstance(scores, Mapping)
        selected = scores[method]
        baseline = scores[BASELINE_METHOD_ANTIGRAVITY]
        assert isinstance(selected, Mapping) and isinstance(baseline, Mapping)
        oracle_auc = max(
            float(value["auc"]) for value in scores.values() if isinstance(value, Mapping)
        )
        replay_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "auc": float(selected["auc"]),
                "zf": int(selected["zf"]),
                "baseline_auc": float(baseline["auc"]),
                "baseline_zf": int(baseline["zf"]),
                "oracle_auc": oracle_auc,
            }
        )
    if not replay_rows:
        raise RuntimeError("no frozen score rows matched the routed population")
    count = len(replay_rows)
    losses = [float(row["oracle_auc"]) - float(row["auc"]) for row in replay_rows]
    exp477 = json.loads(EXP477_RESULT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    qualification = exp477["inherited_exp476_qualification"]
    return {
        "scored": count,
        "routed_without_scores": len(route_rows) - count,
        "average_auc": sum(float(row["auc"]) for row in replay_rows) / count,
        "average_zf_tp": sum(float(row["zf"]) for row in replay_rows) / count,
        "average_oracle_auc_loss": sum(losses) / count,
        "severe_auc_loss_over_0_05": sum(loss > 0.05 for loss in losses),
        "very_severe_auc_loss_over_0_10": sum(loss > 0.10 for loss in losses),
        "baseline_average_auc": sum(float(row["baseline_auc"]) for row in replay_rows) / count,
        "baseline_average_zf_tp": sum(float(row["baseline_zf"]) for row in replay_rows) / count,
        "qualification_average_auc": qualification["average_auc"],
        "qualification_average_zf_tp": qualification["average_zf_tp"],
        "qualification_severe_auc_loss_over_0_05": qualification[
            "severe_auc_loss_over_0_05"
        ],
        "interpretation": (
            "Operational replay uses the fully fitted official policy. "
            "Family-blocked qualification remains the generalisation estimate."
        ),
    }


def render_dashboard_antigravity(result: Mapping[str, object]) -> str:
    """Render live or completed Exp478 evidence in easy Korean."""
    state = str(result.get("state", "running"))
    population = result["population"]
    routing = result["routing"]
    score_replay = result.get("score_replay", {})
    nonfinite_handling = result.get("nonfinite_row_handling", {})
    exclusions = result.get("exclusions", [])
    assert isinstance(population, Mapping) and isinstance(routing, Mapping)
    assert isinstance(score_replay, Mapping) and isinstance(nonfinite_handling, Mapping)
    assert isinstance(exclusions, Sequence)
    complete = state == "complete"
    routing_pass = (
        complete
        and int(routing.get("completed", 0)) == 1117
        and int(routing.get("failures", 1)) == 0
    )
    deterministic_pass = routing_pass and int(routing.get("nondeterministic", 1)) == 0
    feature_direct_pass = (
        routing_pass and int(routing.get("feature_direct_mismatches", 1)) == 0
    )
    pipeline_pass = (
        routing_pass
        and int(routing.get("pipeline_mismatches", 1)) == 0
        and int(routing.get("non_singleton", 1)) == 0
    )
    score_pass = (
        complete
        and int(score_replay.get("scored", 0)) == 962
        and int(score_replay.get("routed_without_scores", 0)) == 155
    )
    nonfinite_pass = (
        int(nonfinite_handling.get("affected_datasets", 0)) == 19
        and int(nonfinite_handling.get("excluded_rows", 0)) == 164
        and int(nonfinite_handling.get("nonfinite_values", 0)) == 983
    )
    source_pass = (
        str(result.get("source_exp477_result_sha256", ""))
        == EXPECTED_EXP477_SHA256_ANTIGRAVITY
    )
    overall_pass = complete and source_pass and _overall_pass_antigravity(
        routing, score_replay, nonfinite_handling
    )
    checked = lambda value: "checked" if value else ""
    progress = 100.0 * int(routing.get("completed", 0)) / 1117
    method_rows = "".join(
        f"<tr><td><code>{escape(str(method))}</code></td><td>{count}</td></tr>"
        for method, count in sorted(dict(routing.get("method_counts", {})).items())
    ) or "<tr><td colspan='2'>아직 결과 없음</td></tr>"
    failure_rows = "".join(
        f"<tr><td>{escape(str(row.get('dataset')))}</td><td>{escape(str(row.get('error_type', row.get('status'))))}</td><td>{escape(str(row.get('error', '')))}</td></tr>"
        for row in list(routing.get("failure_rows", []))[:30]
        if isinstance(row, Mapping)
    ) or "<tr><td colspan='3'>없음</td></tr>"
    exclusion_rows = "".join(
        f"<tr><td>{escape(str(row.get('dataset')))}</td><td>{escape(str(row.get('reason')))}</td></tr>"
        for row in exclusions
        if isinstance(row, Mapping)
    )
    status_text = "전수검사 통과" if overall_pass else ("실행 중" if not complete else "확인 필요")
    status_class = "pass" if overall_pass else "hold"
    auc_text = f"{float(score_replay['average_auc']):.6f}" if score_pass else "-"
    zf_text = f"{float(score_replay['average_zf_tp']):.6f}" if score_pass else "-"
    severe_text = str(score_replay["severe_auc_loss_over_0_05"]) if score_pass else "-"
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp478 공식 1117개 전수검사</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;line-height:1.58;color:#182230}}h1,h2{{line-height:1.25}}.status{{border-left:5px solid;padding:12px 16px;background:#f8fafc}}.pass{{color:#08783e;font-weight:700}}.hold{{color:#b54708;font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:12px;padding:16px;background:#f8fafc}}.big{{font-size:1.55rem;font-weight:750}}.muted{{color:#596579}}.bar{{height:22px;background:#e5e9f0;border-radius:8px;overflow:hidden}}.fill{{height:100%;background:#08783e;width:{progress:.2f}%}}table{{border-collapse:collapse;width:100%;margin:12px 0 26px}}th,td{{border:1px solid #d8dee8;padding:8px 10px;text-align:left}}th{{background:#eef2f7}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}
</style></head><body><h1>Exp478 공식 1117개 전수검사</h1><p class="status {status_class}"><strong>{status_text}</strong> — 라우팅 가능 모집단과 성능 평가 모집단을 분리해 표시합니다.</p>
<div class="cards"><div class="card"><div class="muted">DB 등록</div><div class="big">{population['registered']}</div></div><div class="card"><div class="muted">전수 라우팅</div><div class="big">{routing.get('completed',0)} / 1117</div></div><div class="card"><div class="muted">성공</div><div class="big">{routing.get('success',0)}</div></div><div class="card"><div class="muted">성능 확인 가능</div><div class="big">{score_replay.get('scored',0)} / 962</div></div><div class="card"><div class="muted">성능 미산출</div><div class="big">{score_replay.get('routed_without_scores',0)} / 155</div></div></div>
<h2>1. 진행률</h2><div class="bar"><div class="fill"></div></div><p>{progress:.1f}% · 마지막 갱신: {escape(str(result.get('generated_at_utc','-')))}</p>
<h2>2. 범위와 NaN 처리</h2><p>DB에는 1119개가 있습니다. 정상 클래스 매핑이 없는 1개와 정상 TRAIN이 1개뿐인 1개를 제외하여 1117개를 실제 공식 라우터로 검사합니다. 이 중 기존 후보 성능이 있는 962개만 AUC와 ZF 성능을 계산합니다.</p><p>19개 데이터셋에서 NaN 또는 Inf가 있는 TRAIN 행 164개, 값 983개를 제외했습니다. 값을 채우거나 보간하지 않았으며, 남은 정상 행에는 기존 특징 공식을 그대로 적용했습니다.</p><table><tr><th>제외 데이터셋</th><th>이유</th></tr>{exclusion_rows}</table>
<h2>3. 전수 라우팅 결과</h2><p>오류 {routing.get('failures',0)}개 · 특징/직접 불일치 {routing.get('feature_direct_mismatches',0)}개 · 직접/파이프라인 불일치 {routing.get('pipeline_mismatches',0)}개 · 반복 결과 불일치 {routing.get('nondeterministic',0)}개 · 단일 선택 위반 {routing.get('non_singleton',0)}개</p><table><tr><th>최종 방법</th><th>개수</th></tr>{method_rows}</table><table><tr><th>실패 데이터셋</th><th>종류</th><th>내용</th></tr>{failure_rows}</table>
<h2>4. 962개 성능 재생</h2><div class="cards"><div class="card"><div class="muted">평균 AUC</div><div class="big">{auc_text}</div></div><div class="card"><div class="muted">평균 ZF-TP</div><div class="big">{zf_text}</div></div><div class="card"><div class="muted">큰 AUC 손실</div><div class="big">{severe_text}</div></div></div><p>이 수치는 고정된 후보 결과에 현재 공식 선택을 적용한 운영 재생입니다. 새 데이터 일반화 판단에는 120가족 분리 검증 수치(AUC {float(score_replay.get('qualification_average_auc',0.0)):.6f}, ZF {float(score_replay.get('qualification_average_zf_tp',0.0)):.6f})를 사용합니다.</p>
<h2>5. 진행 체크</h2><label><input type="checkbox" checked disabled> 1119개 DB 모집단 확인</label><label><input type="checkbox" checked disabled> 1117개 포함·2개 제외 규칙 확인</label><label><input type="checkbox" {checked(source_pass)} disabled> Exp477 결과 해시 확인</label><label><input type="checkbox" {checked(nonfinite_pass)} disabled> 19개 데이터셋의 오염된 164행·983값만 제외</label><label><input type="checkbox" {checked(routing_pass)} disabled> 1117개 공식 라우팅 완료</label><label><input type="checkbox" {checked(deterministic_pass)} disabled> 1117개 반복 결과 일치</label><label><input type="checkbox" {checked(feature_direct_pass and pipeline_pass)} disabled> 특징·직접 호출·파이프라인·단일 선택 일치</label><label><input type="checkbox" {checked(score_pass)} disabled> 962개 성능 재생과 155개 미산출 분리</label><label><input type="checkbox" {checked(overall_pass)} disabled> 결과 JSON·HTML 최종 검증</label>
<h2>6. 재현 정보</h2><ul><li>Exp477 결과 SHA-256: <code>{escape(str(result.get('source_exp477_result_sha256','-')))}</code></li><li>새 점수 계산 없음: 기존 962개 후보 결과만 읽기</li><li>MLX: 새 점수·행렬 연산 단계가 없어 평가 대상 아님</li></ul></body></html>"""


def _progress_payload_antigravity(
    *,
    population_count: int,
    exclusions: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    source_hash: str,
    nonfinite_handling: Mapping[str, object],
) -> dict[str, object]:
    return {
        "state": "running",
        "generated_at_utc": _utc_now_antigravity(),
        "source_exp477_result_sha256": source_hash,
        "population": {"registered": population_count + len(exclusions), "included": population_count, "excluded": len(exclusions)},
        "routing": _summarise_routing_antigravity(rows),
        "score_replay": {},
        "nonfinite_row_handling": dict(nonfinite_handling),
        "exclusions": list(exclusions),
    }


def run_antigravity() -> dict[str, object]:
    source_hash = _sha256_antigravity(EXP477_RESULT_PATH_ANTIGRAVITY)
    if source_hash != EXPECTED_EXP477_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp477 result hash changed; refusing Exp478 full census")
    connection = sqlite3.connect(f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True)
    try:
        population, exclusions = select_population_antigravity(connection)
        nonfinite_handling = _audit_nonfinite_normal_train_antigravity(
            connection, population
        )
    finally:
        connection.close()
    if len(population) != 1117 or len(exclusions) != 2:
        raise RuntimeError("database population no longer matches the Exp478 contract")

    source_hashes = _official_source_hashes_antigravity()
    rows_by_dataset: dict[str, dict[str, object]] = {}
    if CHECKPOINT_PATH_ANTIGRAVITY.exists():
        checkpoint = json.loads(CHECKPOINT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        rows_by_dataset = _resume_rows_antigravity(
            checkpoint,
            population=population,
            source_result_hash=source_hash,
            source_hashes=source_hashes,
        )

    initial = _progress_payload_antigravity(
        population_count=len(population),
        exclusions=exclusions,
        rows=list(rows_by_dataset.values()),
        source_hash=source_hash,
        nonfinite_handling=nonfinite_handling,
    )
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        render_dashboard_antigravity(initial), encoding="utf-8"
    )
    pending = [dataset for dataset in population if dataset not in rows_by_dataset]
    for index, dataset in enumerate(pending, start=1):
        rows_by_dataset[dataset] = evaluate_dataset_antigravity(dataset)
        if index % PROGRESS_INTERVAL_ANTIGRAVITY == 0 or index == len(pending):
            ordered_rows = [rows_by_dataset[name] for name in population if name in rows_by_dataset]
            checkpoint = {
                "schema": "exp478_full_1117_checkpoint_v1",
                "updated_at_utc": _utc_now_antigravity(),
                "source_exp477_result_sha256": source_hash,
                "source_hashes": source_hashes,
                "target_count": len(population),
                "completed_count": len(ordered_rows),
                "rows": ordered_rows,
            }
            _write_json_antigravity(CHECKPOINT_PATH_ANTIGRAVITY, checkpoint)
            progress = _progress_payload_antigravity(
                population_count=len(population),
                exclusions=exclusions,
                rows=ordered_rows,
                source_hash=source_hash,
                nonfinite_handling=nonfinite_handling,
            )
            DASHBOARD_PATH_ANTIGRAVITY.write_text(
                render_dashboard_antigravity(progress), encoding="utf-8"
            )
            print(f"routes {len(ordered_rows)}/{len(population)}")

    ordered_rows = [rows_by_dataset[name] for name in population]
    routing = _summarise_routing_antigravity(ordered_rows)
    score_replay = _summarise_score_replay_antigravity(ordered_rows)
    overall_pass = _overall_pass_antigravity(
        routing, score_replay, nonfinite_handling
    )
    result: dict[str, object] = {
        "schema": "exp478_full_1117_official_routing_result_v1",
        "state": "complete",
        "generated_at_utc": _utc_now_antigravity(),
        "source_exp477_result_sha256": source_hash,
        "source_hashes": source_hashes,
        "population": {"registered": 1119, "included": 1117, "excluded": 2},
        "routing": routing,
        "score_replay": score_replay,
        "nonfinite_row_handling": nonfinite_handling,
        "exclusions": exclusions,
        "rows": ordered_rows,
        "overall_pass": overall_pass,
        "acceleration_assessment": {
            "mlx_eligible": False,
            "reason": "Exp478 adds no score source or dense tensor stage.",
        },
    }
    _write_json_antigravity(RESULT_PATH_ANTIGRAVITY, result)
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        render_dashboard_antigravity(result), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "population": result["population"],
                "routing": routing,
                "score_replay": score_replay,
                "overall_pass": overall_pass,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not overall_pass:
        raise RuntimeError("Exp478 full census found official routing failures")
    return result


if __name__ == "__main__":
    run_antigravity()
