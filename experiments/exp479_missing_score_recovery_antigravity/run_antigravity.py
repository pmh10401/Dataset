"""Resumable smoke and official execution for Exp479."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Mapping, Sequence

from experiments.exp474_conservative_fallback_routing_antigravity.run_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
)
from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    load_checkpoint_records_antigravity,
)
from experiments.exp478_full_1117_official_routing_antigravity.run_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    RESULT_PATH_ANTIGRAVITY as EXP478_RESULT_PATH_ANTIGRAVITY,
)
from experiments.exp_routing_label_free_antigravity.official_router_antigravity import (
    route_official_antigravity,
)

from .config_antigravity import (
    DASHBOARD_PATH_ANTIGRAVITY,
    EXPECTED_EXISTING_SCORE_DATASETS_ANTIGRAVITY,
    EXPECTED_RECOVERY_DATASETS_ANTIGRAVITY,
    IMPLEMENTATION_REPORT_PATH_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
    OFFICIAL_CHECKPOINT_PATH_ANTIGRAVITY,
    OFFICIAL_RESULT_PATH_ANTIGRAVITY,
    SMOKE_CHECKPOINT_PATH_ANTIGRAVITY,
    SMOKE_RESULT_PATH_ANTIGRAVITY,
)
from .evaluation_antigravity import (
    evaluate_scored_dataset_antigravity,
    load_retained_test_labels_antigravity,
    score_prepared_methods_antigravity,
)
from .population_antigravity import (
    build_population_manifest_antigravity,
    select_recovery_population_antigravity,
)
from .preprocessing_antigravity import prepare_dataset_values_antigravity


def write_json_atomic_antigravity(path: Path, payload: object) -> None:
    """Write finite JSON and atomically replace only the requested artifact."""
    target = Path(path)
    temporary = target.with_name(
        target.stem + "_temporary_antigravity" + target.suffix
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _write_text_atomic_antigravity(path: Path, text: str) -> None:
    target = Path(path)
    temporary = target.with_name(target.stem + "_temporary_antigravity" + target.suffix)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def resume_rows_antigravity(
    checkpoint: Mapping[str, object],
    *,
    source_fingerprint: str,
    population_sha256: str,
    mode: str,
    population: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Reuse rows only when every result-affecting identity and mode matches."""
    if (
        checkpoint.get("schema") != "exp479_checkpoint_v1"
        or checkpoint.get("source_fingerprint") != source_fingerprint
        or checkpoint.get("population_sha256") != population_sha256
        or checkpoint.get("mode") != mode
    ):
        return {}
    allowed = set(population)
    rows = checkpoint.get("rows")
    if type(rows) is not list:
        return {}
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if type(row) is not dict or type(row.get("dataset")) is not str:
            continue
        dataset = str(row["dataset"])
        if dataset in allowed and dataset not in result:
            result[dataset] = row
    return result


def _source_fingerprint_antigravity(manifest: Mapping[str, object]) -> str:
    paths = (
        Path(__file__).with_name("config_antigravity.py"),
        Path(__file__).with_name("contracts_antigravity.py"),
        Path(__file__).with_name("population_antigravity.py"),
        Path(__file__).with_name("preprocessing_antigravity.py"),
        Path(__file__).with_name("evaluation_antigravity.py"),
        Path(__file__).with_name("acceleration_antigravity.py"),
    )
    digest = sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def summarise_performance_antigravity(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Calculate exact performance from per-dataset rows without rounded inputs."""
    if not rows:
        raise ValueError("at least one performance row is required")
    names: set[str] = set()
    aucs: list[float] = []
    zfs: list[int] = []
    baseline_aucs: list[float] = []
    baseline_zfs: list[int] = []
    losses: list[float] = []
    methods: Counter[str] = Counter()
    for row in rows:
        dataset = row.get("dataset")
        method = row.get("method")
        if type(dataset) is not str or not dataset or dataset in names:
            raise ValueError("performance rows require unique dataset names")
        if method not in METHODS_ANTIGRAVITY:
            raise ValueError(f"invalid official method for {dataset}")
        auc = float(row["auc"])
        zf = row["zf"]
        baseline_auc = float(row["baseline_auc"])
        baseline_zf = row["baseline_zf"]
        oracle_auc = float(row["oracle_auc"])
        if not all(math.isfinite(value) for value in (auc, baseline_auc, oracle_auc)):
            raise ValueError(f"non-finite performance for {dataset}")
        if type(zf) is not int or type(baseline_zf) is not int or zf < 0 or baseline_zf < 0:
            raise ValueError(f"invalid ZF-TP for {dataset}")
        names.add(dataset)
        methods[str(method)] += 1
        aucs.append(auc)
        zfs.append(zf)
        baseline_aucs.append(baseline_auc)
        baseline_zfs.append(baseline_zf)
        losses.append(oracle_auc - auc)
    count = len(rows)
    return {
        "dataset_count": count,
        "average_auc": sum(aucs) / count,
        "average_zf_tp": sum(zfs) / count,
        "baseline_average_auc": sum(baseline_aucs) / count,
        "baseline_average_zf_tp": sum(baseline_zfs) / count,
        "average_oracle_auc_loss": sum(losses) / count,
        "severe_auc_loss_over_0_05": sum(loss > 0.05 for loss in losses),
        "very_severe_auc_loss_over_0_10": sum(loss > 0.10 for loss in losses),
        "method_counts": dict(sorted(methods.items())),
    }


def _canonical_recovery_rows_antigravity(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    canonical: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        official = row.get("official")
        baseline = row.get("baseline")
        if not isinstance(official, Mapping) or not isinstance(baseline, Mapping):
            raise ValueError("successful recovery row is missing metrics")
        canonical.append(
            {
                "dataset": row["dataset"],
                "method": row["official_method"],
                "auc": official["auc"],
                "zf": official["zf"],
                "baseline_auc": baseline["auc"],
                "baseline_zf": baseline["zf"],
                "oracle_auc": row["oracle_auc"],
            }
        )
    return canonical


def _frozen_performance_rows_antigravity() -> list[dict[str, object]]:
    records = load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    if len(records) != EXPECTED_EXISTING_SCORE_DATASETS_ANTIGRAVITY:
        raise RuntimeError(f"expected 962 frozen score records, got {len(records)}")
    canonical: list[dict[str, object]] = []
    baseline_method = METHODS_ANTIGRAVITY[0]
    for record in records:
        dataset = str(record["dataset"])
        method = route_official_antigravity(dataset)
        scores = record["scores"]
        if not isinstance(scores, Mapping) or tuple(scores) != METHODS_ANTIGRAVITY:
            raise ValueError(f"frozen scores do not match four methods: {dataset}")
        selected = scores[method]
        baseline = scores[baseline_method]
        if not isinstance(selected, Mapping) or not isinstance(baseline, Mapping):
            raise ValueError(f"invalid frozen score row: {dataset}")
        oracle_auc = max(
            float(value["auc"])
            for value in scores.values()
            if isinstance(value, Mapping)
        )
        canonical.append(
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
    return canonical


def render_official_dashboard_antigravity(result: Mapping[str, object]) -> str:
    """Render live Task 6 progress and check completion only on an exact pass."""
    completed = int(result.get("completed", 0))
    success = int(result.get("success", 0))
    failures = int(result.get("failures", 0))
    passed = (
        result.get("state") == "complete"
        and completed == EXPECTED_RECOVERY_DATASETS_ANTIGRAVITY
        and success == EXPECTED_RECOVERY_DATASETS_ANTIGRAVITY
        and failures == 0
    )
    status = "공식 실행 완료" if passed else "공식 실행 중"
    status_class = "pass" if passed else "running"
    latest = escape(str(result.get("latest_dataset", "아직 없음")))
    recovery = result.get("recovery_summary")
    combined = result.get("combined_summary")

    def metric(summary: object, key: str, digits: int = 6) -> str:
        if not isinstance(summary, Mapping) or key not in summary:
            return "-"
        value = summary[key]
        if key == "severe_auc_loss_over_0_05":
            return str(int(value))
        return f"{float(value):.{digits}f}"

    task6_checked = " checked" if passed else ""
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp479 성능 복구 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1050px;margin:30px auto;padding:0 20px 50px;line-height:1.6;color:#182230}}
h1,h2{{line-height:1.25}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:10px;padding:14px}}.big{{font-size:1.55rem;font-weight:750}}
.pass{{border-left:5px solid #08783e;background:#f2fbf6;padding:12px 16px}}.running{{border-left:5px solid #2563eb;background:#eff6ff;padding:12px 16px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee8;padding:8px 10px;text-align:left}}th{{background:#eef2f7}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}code{{background:#eef2f7;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>Exp479 성능 미산출 155개 복구</h1>
<p class="{status_class}"><strong>{status}</strong> — 완료 {completed} / 155, 성공 {success}, 실패 {failures}</p>
<div class="cards"><div class="card"><div>공식 진행률</div><div class="big">{completed} / 155</div></div><div class="card"><div>성공</div><div class="big">{success}</div></div><div class="card"><div>실패</div><div class="big">{failures}</div></div><div class="card"><div>최근 데이터셋</div><div><code>{latest}</code></div></div><div class="card"><div>누적 실행 시간</div><div class="big">{float(result.get('elapsed_seconds', 0.0)) / 60:.1f}분</div></div></div>
<h2>성능 결과</h2><table><tr><th>범위</th><th>공식 평균 AUC</th><th>기준선 평균 AUC</th><th>공식 평균 ZF-TP</th><th>기준선 평균 ZF-TP</th><th>AUC 큰 손실</th></tr>
<tr><td>새로 복구한 155개</td><td>{metric(recovery,'average_auc')}</td><td>{metric(recovery,'baseline_average_auc')}</td><td>{metric(recovery,'average_zf_tp')}</td><td>{metric(recovery,'baseline_average_zf_tp')}</td><td>{metric(recovery,'severe_auc_loss_over_0_05')}</td></tr>
<tr><td>기존 962개 + 신규 155개</td><td>{metric(combined,'average_auc')}</td><td>{metric(combined,'baseline_average_auc')}</td><td>{metric(combined,'average_zf_tp')}</td><td>{metric(combined,'baseline_average_zf_tp')}</td><td>{metric(combined,'severe_auc_loss_over_0_05')}</td></tr></table>
<p>기존 962개는 다시 학습하지 않고 저장된 네 후보 점수에 현재 공식 라우터를 적용했습니다. 신규 155개만 Numba CPU로 계산합니다. 데이터셋 이름이나 TEST 라벨은 라우팅 입력에 쓰지 않습니다.</p>
<p><strong>가속 판단:</strong> 고정 Python 환경은 MLX 미설치 상태이므로 Numba CPU 유지를 선택했습니다.</p>
<p><strong>복구 대상 SHA256:</strong> <code>{escape(str(result.get('population_sha256', '-')))}</code></p>
<h2>진행 체크</h2>
<label><input type="checkbox" checked disabled> Exp479 설계 승인</label><label><input type="checkbox" checked disabled> 파생 데이터셋 정상 라벨 의미 확인</label><label><input type="checkbox" checked disabled> 구현 계획서 작성</label><label><input type="checkbox" checked disabled> 구현 계획 승인</label><label><input type="checkbox" checked disabled> Task 1 — 대상 155개와 실행 계약 고정</label><label><input type="checkbox" checked disabled> Task 2 — Numba 전처리와 155개 감사</label><label><input type="checkbox" checked disabled> Task 3 — 네 후보 성능 평가 코드 구현</label><label><input type="checkbox" checked disabled> Task 4 — MLX 적용 여부 판정: Numba CPU 유지</label><label><input type="checkbox" checked disabled> Task 5 — 실행기와 소규모 실데이터 검증</label><label><input type="checkbox"{task6_checked} disabled> Task 6 — 공식 155개 실행과 1,117개 통합</label>
</body></html>"""


def _render_implementation_report_antigravity(result: Mapping[str, object]) -> str:
    recovery = result["recovery_summary"]
    combined = result["combined_summary"]
    assert isinstance(recovery, Mapping) and isinstance(combined, Mapping)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp479 구현 결과</title><style>body{{font-family:system-ui,-apple-system,sans-serif;max-width:900px;margin:32px auto;padding:0 20px 50px;line-height:1.65;color:#182230}}h1,h2{{line-height:1.25}}.pass{{border-left:5px solid #08783e;background:#f2fbf6;padding:12px 16px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee8;padding:8px 10px;text-align:left}}th{{background:#eef2f7}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}</style></head><body>
<h1>Exp479 구현 결과</h1><p class="pass"><strong>공식 실행 완료:</strong> 빠졌던 155개 데이터셋을 모두 계산했고 실패는 0개입니다.</p>
<h2>쉽게 설명하면</h2><p>길이가 제각각이라 이전 시험에서 점수를 내지 못했던 155개를 같은 길이로 맞춘 뒤 네 방법을 공정하게 비교했습니다. 기존 962개는 다시 돌리지 않고 보관된 점수를 그대로 사용해 전체 1,117개 결과를 계산했습니다.</p>
<table><tr><th>범위</th><th>공식 평균 AUC</th><th>항상 MiniRocket AUC</th><th>공식 평균 ZF-TP</th><th>항상 MiniRocket ZF-TP</th></tr><tr><td>신규 155개</td><td>{float(recovery['average_auc']):.6f}</td><td>{float(recovery['baseline_average_auc']):.6f}</td><td>{float(recovery['average_zf_tp']):.6f}</td><td>{float(recovery['baseline_average_zf_tp']):.6f}</td></tr><tr><td>전체 1,117개</td><td>{float(combined['average_auc']):.6f}</td><td>{float(combined['baseline_average_auc']):.6f}</td><td>{float(combined['average_zf_tp']):.6f}</td><td>{float(combined['baseline_average_zf_tp']):.6f}</td></tr></table>
<h2>해석할 때 주의할 점</h2><p>이 값은 현재 공식 라우터를 전체 데이터에 적용한 운영 성능입니다. 완전히 새로운 데이터 가족에 대한 일반화 성능은 기존 가족 분리 검증 결과와 구분해서 봐야 합니다.</p>
<h2>완료 확인</h2><label><input type="checkbox" checked disabled> 신규 155개 모두 네 방법 계산</label><label><input type="checkbox" checked disabled> 실패 0개와 유한 성능값 확인</label><label><input type="checkbox" checked disabled> 기존 962개를 변경하지 않고 1,117개 통합</label><label><input type="checkbox" checked disabled> 결과 JSON과 HTML 대시보드 저장</label>
</body></html>"""


def evaluate_dataset_antigravity(
    connection: sqlite3.Connection,
    dataset: str,
) -> dict[str, object]:
    """Score without labels, then open aligned labels and calculate metrics."""
    started = perf_counter()
    prepared = prepare_dataset_values_antigravity(connection, dataset)
    scored = score_prepared_methods_antigravity(prepared)
    labels = load_retained_test_labels_antigravity(
        connection, dataset, prepared.test_indices
    )
    official_method = route_official_antigravity(dataset)
    evaluated = evaluate_scored_dataset_antigravity(
        dataset,
        {method: evidence.test_scores for method, evidence in scored.items()},
        labels,
        official_method,
    )
    return {
        "status": "ok",
        **evaluated,
        "target_length": prepared.target_length,
        "retained_train_rows": len(prepared.train_rows),
        "retained_test_rows": len(prepared.test_rows),
        "excluded_train_rows": prepared.excluded_train_rows,
        "excluded_test_rows": prepared.excluded_test_rows,
        "feature_dimensions": {
            method: evidence.feature_dimension for method, evidence in scored.items()
        },
        "seed": next(iter(scored.values())).seed,
        "elapsed_seconds": perf_counter() - started,
    }


def run_smoke_antigravity(
    limit: int,
    *,
    resume: bool,
) -> dict[str, object]:
    """Run the first one or two frozen recovery datasets with safe resume."""
    if type(limit) is not int or limit not in (1, 2):
        raise ValueError("smoke limit must be exactly 1 or 2")
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        population = select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
        manifest = build_population_manifest_antigravity(
            population,
            checkpoint_dir=CHECKPOINT_DIR_ANTIGRAVITY,
            exp478_result_path=EXP478_RESULT_PATH_ANTIGRAVITY,
            database_path=DATABASE_PATH_ANTIGRAVITY,
        )
        fingerprint = _source_fingerprint_antigravity(manifest)
        existing: dict[str, dict[str, object]] = {}
        if resume and SMOKE_CHECKPOINT_PATH_ANTIGRAVITY.exists():
            checkpoint = json.loads(
                SMOKE_CHECKPOINT_PATH_ANTIGRAVITY.read_text(encoding="utf-8")
            )
            existing = resume_rows_antigravity(
                checkpoint,
                source_fingerprint=fingerprint,
                population_sha256=str(manifest["recovery_dataset_sha256"]),
                mode="smoke",
                population=population.datasets,
            )
        target = population.datasets[:limit]
        for dataset in target:
            if dataset in existing:
                continue
            try:
                existing[dataset] = evaluate_dataset_antigravity(connection, dataset)
            except Exception as error:
                existing[dataset] = {
                    "dataset": dataset,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            checkpoint_payload = {
                "schema": "exp479_checkpoint_v1",
                "mode": "smoke",
                "source_fingerprint": fingerprint,
                "population_sha256": manifest["recovery_dataset_sha256"],
                "rows": [existing[name] for name in sorted(existing)],
            }
            write_json_atomic_antigravity(
                SMOKE_CHECKPOINT_PATH_ANTIGRAVITY, checkpoint_payload
            )
    finally:
        connection.close()
    rows = [existing[name] for name in target]
    result = {
        "schema": "exp479_smoke_results_v1",
        "mode": "smoke",
        "requested": limit,
        "completed": len(rows),
        "success": sum(row.get("status") == "ok" for row in rows),
        "failures": sum(row.get("status") != "ok" for row in rows),
        "source_fingerprint": fingerprint,
        "population_sha256": manifest["recovery_dataset_sha256"],
        "rows": rows,
    }
    write_json_atomic_antigravity(SMOKE_RESULT_PATH_ANTIGRAVITY, result)
    return result


def run_official_antigravity(*, resume: bool) -> dict[str, object]:
    """Run the approved 155-dataset recovery and combine it with frozen 962 scores."""
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    run_started = perf_counter()
    try:
        population = select_recovery_population_antigravity(
            connection, CHECKPOINT_DIR_ANTIGRAVITY
        )
        manifest = build_population_manifest_antigravity(
            population,
            checkpoint_dir=CHECKPOINT_DIR_ANTIGRAVITY,
            exp478_result_path=EXP478_RESULT_PATH_ANTIGRAVITY,
            database_path=DATABASE_PATH_ANTIGRAVITY,
        )
        fingerprint = _source_fingerprint_antigravity(manifest)
        existing: dict[str, dict[str, object]] = {}
        if resume and OFFICIAL_CHECKPOINT_PATH_ANTIGRAVITY.exists():
            checkpoint = json.loads(
                OFFICIAL_CHECKPOINT_PATH_ANTIGRAVITY.read_text(encoding="utf-8")
            )
            existing = resume_rows_antigravity(
                checkpoint,
                source_fingerprint=fingerprint,
                population_sha256=str(manifest["recovery_dataset_sha256"]),
                mode="official",
                population=population.datasets,
            )

        initial_rows = [existing[name] for name in population.datasets if name in existing]
        initial_ok = [row for row in initial_rows if row.get("status") == "ok"]
        initial_elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in initial_rows)
        _write_text_atomic_antigravity(
            DASHBOARD_PATH_ANTIGRAVITY,
            render_official_dashboard_antigravity(
                {
                    "state": "running",
                    "completed": len(initial_rows),
                    "success": len(initial_ok),
                    "failures": len(initial_rows) - len(initial_ok),
                    "latest_dataset": initial_rows[-1]["dataset"] if initial_rows else "아직 없음",
                    "elapsed_seconds": initial_elapsed,
                }
            ),
        )
        for index, dataset in enumerate(population.datasets, start=1):
            if dataset in existing and existing[dataset].get("status") == "ok":
                print(f"[{index}/155] {dataset} resumed ok", flush=True)
                continue
            try:
                existing[dataset] = evaluate_dataset_antigravity(connection, dataset)
            except Exception as error:
                existing[dataset] = {
                    "dataset": dataset,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": 0.0,
                }
            ordered_rows = [
                existing[name] for name in population.datasets if name in existing
            ]
            checkpoint_payload = {
                "schema": "exp479_checkpoint_v1",
                "mode": "official",
                "source_fingerprint": fingerprint,
                "population_sha256": manifest["recovery_dataset_sha256"],
                "rows": ordered_rows,
            }
            write_json_atomic_antigravity(
                OFFICIAL_CHECKPOINT_PATH_ANTIGRAVITY, checkpoint_payload
            )
            success = sum(row.get("status") == "ok" for row in ordered_rows)
            failures = len(ordered_rows) - success
            recovery_summary = None
            successful_rows = _canonical_recovery_rows_antigravity(ordered_rows)
            if successful_rows:
                recovery_summary = summarise_performance_antigravity(successful_rows)
            _write_text_atomic_antigravity(
                DASHBOARD_PATH_ANTIGRAVITY,
                render_official_dashboard_antigravity(
                    {
                        "state": "running",
                        "completed": len(ordered_rows),
                        "success": success,
                        "failures": failures,
                        "latest_dataset": dataset,
                        "elapsed_seconds": sum(
                            float(row.get("elapsed_seconds", 0.0)) for row in ordered_rows
                        ),
                        "recovery_summary": recovery_summary,
                    }
                ),
            )
            state = "ok" if existing[dataset].get("status") == "ok" else "error"
            print(
                f"[{index}/155] {dataset} {state} "
                f"{float(existing[dataset].get('elapsed_seconds', 0.0)):.2f}s",
                flush=True,
            )
    finally:
        connection.close()

    rows = [existing[name] for name in population.datasets if name in existing]
    success = sum(row.get("status") == "ok" for row in rows)
    failures = len(rows) - success
    if len(rows) != EXPECTED_RECOVERY_DATASETS_ANTIGRAVITY or success != len(rows):
        result = {
            "schema": "exp479_official_results_v1",
            "mode": "official",
            "state": "failed",
            "completed": len(rows),
            "success": success,
            "failures": failures,
            "source_fingerprint": fingerprint,
            "population_sha256": manifest["recovery_dataset_sha256"],
            "rows": rows,
        }
        write_json_atomic_antigravity(OFFICIAL_RESULT_PATH_ANTIGRAVITY, result)
        _write_text_atomic_antigravity(
            DASHBOARD_PATH_ANTIGRAVITY,
            render_official_dashboard_antigravity(result),
        )
        raise RuntimeError(
            f"official recovery incomplete: completed={len(rows)} success={success} failures={failures}"
        )

    recovery_rows = _canonical_recovery_rows_antigravity(rows)
    recovery_summary = summarise_performance_antigravity(recovery_rows)
    frozen_rows = _frozen_performance_rows_antigravity()
    frozen_summary = summarise_performance_antigravity(frozen_rows)
    combined_rows = frozen_rows + recovery_rows
    if len(combined_rows) != 1117 or len({str(row["dataset"]) for row in combined_rows}) != 1117:
        raise RuntimeError("combined performance population is not exactly 1,117 unique datasets")
    combined_summary = summarise_performance_antigravity(combined_rows)
    result = {
        "schema": "exp479_official_results_v1",
        "mode": "official",
        "state": "complete",
        "completed": len(rows),
        "success": success,
        "failures": failures,
        "existing_frozen_scores": len(frozen_rows),
        "combined_datasets": len(combined_rows),
        "source_fingerprint": fingerprint,
        "population_sha256": manifest["recovery_dataset_sha256"],
        "exp478_result_sha256": manifest["source_hashes"]["exp478_result"],
        "selected_backend": "numba_cpu",
        "elapsed_seconds": sum(float(row.get("elapsed_seconds", 0.0)) for row in rows),
        "wall_seconds_this_invocation": perf_counter() - run_started,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_summary": recovery_summary,
        "frozen_962_summary": frozen_summary,
        "combined_summary": combined_summary,
        "rows": rows,
    }
    write_json_atomic_antigravity(OFFICIAL_RESULT_PATH_ANTIGRAVITY, result)
    _write_text_atomic_antigravity(
        DASHBOARD_PATH_ANTIGRAVITY,
        render_official_dashboard_antigravity(result),
    )
    _write_text_atomic_antigravity(
        IMPLEMENTATION_REPORT_PATH_ANTIGRAVITY,
        _render_implementation_report_antigravity(result),
    )
    return result


__all__ = [
    "evaluate_dataset_antigravity",
    "render_official_dashboard_antigravity",
    "resume_rows_antigravity",
    "run_official_antigravity",
    "run_smoke_antigravity",
    "summarise_performance_antigravity",
    "write_json_atomic_antigravity",
]
