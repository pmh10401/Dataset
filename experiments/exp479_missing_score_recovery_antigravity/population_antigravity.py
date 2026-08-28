"""Select the exact Exp479 missing-score recovery population."""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import sqlite3
from typing import Mapping

from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    load_checkpoint_records_antigravity,
)
from experiments.exp478_full_1117_official_routing_antigravity.run_antigravity import (
    select_population_antigravity,
)

from .config_antigravity import (
    DATABASE_EXPECTED_SHA256_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
)
from .contracts_antigravity import PopulationAuditAntigravity


def _sha256_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_score_record_antigravity(record: Mapping[str, object]) -> None:
    dataset = record.get("dataset")
    scores = record.get("scores")
    if type(dataset) is not str or not dataset or type(scores) is not dict:
        raise ValueError("existing score record is malformed")
    if tuple(sorted(scores)) != tuple(sorted(METHODS_ANTIGRAVITY)):
        raise ValueError(f"existing score record is incomplete: {dataset}")
    for method in METHODS_ANTIGRAVITY:
        metrics = scores[method]
        if type(metrics) is not dict or set(metrics) != {"auc", "zf"}:
            raise ValueError(f"existing score metrics are malformed: {dataset}/{method}")
        auc = metrics["auc"]
        zf = metrics["zf"]
        if type(auc) is not float or not math.isfinite(auc):
            raise ValueError(f"existing AUC is invalid: {dataset}/{method}")
        if type(zf) is not int or zf < 0:
            raise ValueError(f"existing ZF-TP is invalid: {dataset}/{method}")


def select_recovery_population_antigravity(
    connection: sqlite3.Connection,
    checkpoint_dir: Path,
) -> PopulationAuditAntigravity:
    """Return the exact sorted 1117-minus-962 Exp479 recovery population."""
    if type(connection) is not sqlite3.Connection:
        raise ValueError("connection must be an exact sqlite3.Connection")
    directory = Path(checkpoint_dir)
    checkpoint_paths = tuple(sorted(directory.glob("chunk_*_antigravity.json")))
    included, exclusions = select_population_antigravity(connection)
    records = load_checkpoint_records_antigravity(directory)
    for record in records:
        _validate_score_record_antigravity(record)
    included_set = set(included)
    scored = {str(record["dataset"]) for record in records}
    if not scored <= included_set:
        raise ValueError("existing scores contain datasets outside the 1117 population")
    datasets = tuple(sorted(included_set - scored))
    excluded_datasets = tuple(str(row["dataset"]) for row in exclusions)
    return PopulationAuditAntigravity(
        datasets=datasets,
        excluded_datasets=excluded_datasets,
        included_count=len(included),
        existing_score_count=len(records),
        checkpoint_count=len(checkpoint_paths),
    )


def build_population_manifest_antigravity(
    audit: PopulationAuditAntigravity,
    *,
    checkpoint_dir: Path,
    exp478_result_path: Path,
    database_path: Path,
) -> dict[str, object]:
    """Pin the exact Task 1 inputs without reading TEST labels or rewriting sources."""
    if type(audit) is not PopulationAuditAntigravity:
        raise ValueError("audit must be an exact PopulationAuditAntigravity")
    checkpoint_paths = tuple(
        sorted(Path(checkpoint_dir).glob("chunk_*_antigravity.json"))
    )
    if len(checkpoint_paths) != audit.checkpoint_count:
        raise ValueError("score checkpoint count changed after population selection")
    result_path = Path(exp478_result_path)
    database = Path(database_path)
    database_stat = database.stat()
    recovery_bytes = ("\n".join(audit.datasets) + "\n").encode("utf-8")
    return {
        "schema": "exp479_population_manifest_v1",
        "counts": {
            "included": audit.included_count,
            "existing_scores": audit.existing_score_count,
            "recovery": len(audit.datasets),
            "checkpoint_files": audit.checkpoint_count,
        },
        "recovery_dataset_sha256": sha256(recovery_bytes).hexdigest(),
        "source_hashes": {
            "exp478_result": _sha256_antigravity(result_path),
            "score_checkpoints": {
                path.name: _sha256_antigravity(path) for path in checkpoint_paths
            },
        },
        "database": {
            "path": str(database),
            "size_bytes": int(database_stat.st_size),
            "inode": int(database_stat.st_ino),
            "expected_sha256": DATABASE_EXPECTED_SHA256_ANTIGRAVITY,
        },
    }


def render_task1_dashboard_antigravity(
    audit: PopulationAuditAntigravity,
    manifest: Mapping[str, object],
) -> str:
    """Render the truthful pre-scoring dashboard after Task 1 completes."""
    if type(audit) is not PopulationAuditAntigravity:
        raise ValueError("audit must be an exact PopulationAuditAntigravity")
    if type(manifest) is not dict or manifest.get("schema") != "exp479_population_manifest_v1":
        raise ValueError("population manifest is invalid")
    target_hash = manifest.get("recovery_dataset_sha256")
    if type(target_hash) is not str or len(target_hash) != 64:
        raise ValueError("recovery dataset hash is invalid")
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Exp479 성능 복구 대시보드</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 20px 50px; line-height: 1.62; color: #182230; }}
    h1,h2 {{ line-height: 1.25; }} .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
    .card {{ border:1px solid #d8dee8; border-radius:9px; padding:14px; }} .big {{ font-size:1.8rem; font-weight:750; }}
    .ready {{ border-left:5px solid #175cd3; background:#eff8ff; padding:12px 16px; }}
    .pending {{ border-left:5px solid #b54708; background:#fff8f0; padding:12px 16px; }}
    code {{ background:#eef2f7; padding:2px 5px; border-radius:4px; }} label {{ display:block; margin:8px 0; }} input {{ margin-right:8px; }}
  </style>
</head>
<body>
  <h1>Exp479 성능 미산출 155개 복구</h1>
  <p class=\"ready\"><strong>현재 상태:</strong> Task 1 완료. 기존 점수가 없는 정확한 155개 대상과 실행 근거를 고정했습니다.</p>
  <div class=\"cards\">
    <div class=\"card\"><div>전체 대상</div><div class=\"big\">{audit.included_count:,}</div></div>
    <div class=\"card\"><div>기존 점수 있음</div><div class=\"big\">{audit.existing_score_count:,}</div></div>
    <div class=\"card\"><div>이번 복구 대상</div><div class=\"big\">{len(audit.datasets):,}</div></div>
    <div class=\"card\"><div>기존 체크포인트</div><div class=\"big\">{audit.checkpoint_count}</div></div>
  </div>
  <h2>대상 고정 결과</h2>
  <p><code>1,117 - 962 = 155</code>이며 중복은 없습니다. 제외 데이터셋은 <code>CornellWhaleChallenge</code>와 <code>Phoneme_normal_39</code>입니다.</p>
  <p>대상 목록 SHA256: <code>{target_hash}</code></p>
  <p class=\"pending\"><strong>성능 상태:</strong> 아직 성능 수치를 계산하지 않았습니다. Task 2 전처리와 Task 3 평가가 끝난 뒤 실제 AUC와 ZF-TP를 표시합니다.</p>
  <h2>진행 체크</h2>
  <label><input type=\"checkbox\" checked disabled> Exp479 설계 승인</label>
  <label><input type=\"checkbox\" checked disabled> 파생 데이터셋 정상 라벨 의미 확인</label>
  <label><input type=\"checkbox\" checked disabled> 구현 계획서 작성</label>
  <label><input type=\"checkbox\" checked disabled> 구현 계획 승인</label>
  <label><input type=\"checkbox\" checked disabled> Task 1 — 대상 155개와 실행 계약 고정</label>
  <label><input type=\"checkbox\" disabled> Task 2 — Numba 전처리</label>
  <label><input type=\"checkbox\" disabled> Task 3 — 네 후보 성능 평가</label>
  <label><input type=\"checkbox\" disabled> Task 4 — MLX 적용 여부 판정</label>
  <label><input type=\"checkbox\" disabled> Task 5 — 실행기와 소규모 검증</label>
  <label><input type=\"checkbox\" disabled> Task 6 — 공식 155개 실행과 1,117개 통합</label>
</body>
</html>
"""


__all__ = [
    "build_population_manifest_antigravity",
    "render_task1_dashboard_antigravity",
    "select_recovery_population_antigravity",
]
