"""Numba-only ragged row preparation for Exp479."""

from __future__ import annotations

import sqlite3
from typing import Mapping, Sequence

from numba import njit
import numpy as np

from .contracts_antigravity import PopulationAuditAntigravity, PreparedDatasetAntigravity


_VALUE_SQL_ANTIGRAVITY = (
    "SELECT i.split, i.instance_index, i.values_blob FROM instances AS i "
    "JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND "
    "((i.split='TRAIN' AND i.label='0') OR i.split='TEST') "
    "ORDER BY CASE i.split WHEN 'TRAIN' THEN 0 ELSE 1 END, i.instance_index"
)


def lower_median_length_antigravity(lengths: Sequence[int]) -> int:
    """Return the lower of the two middle lengths for an even-size collection."""
    values = tuple(lengths)
    if not values:
        raise ValueError("lengths must be non-empty")
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("lengths must contain positive exact integers")
    ordered = tuple(sorted(values))
    return ordered[(len(ordered) - 1) // 2]


@njit(cache=True)
def resample_row_linear_antigravity(
    row: np.ndarray, target_length: int
) -> np.ndarray:
    """Linearly resample one finite row; equal lengths are copied exactly."""
    if row.ndim != 1 or row.size < 1 or target_length < 1:
        raise ValueError("a non-empty row and positive target_length are required")
    if row.size == target_length:
        return row.copy()
    output = np.empty(target_length, dtype=np.float64)
    if target_length == 1:
        output[0] = row[0]
        return output
    scale = (row.size - 1) / (target_length - 1)
    for index in range(target_length):
        position = index * scale
        left = int(position)
        right = min(left + 1, row.size - 1)
        weight = position - left
        output[index] = row[left] * (1.0 - weight) + row[right] * weight
    return output


def _decode_finite_row_antigravity(blob: object) -> np.ndarray | None:
    if type(blob) is not bytes or not blob or len(blob) % 4:
        raise ValueError("values_blob must contain little-endian float32 values")
    row = np.frombuffer(blob, dtype="<f4").astype(np.float64)
    return row if np.isfinite(row).all() else None


def _standardize_rows_antigravity(rows: list[np.ndarray], target_length: int) -> np.ndarray:
    prepared = np.empty((len(rows), target_length), dtype=np.float64)
    for index, row in enumerate(rows):
        prepared[index] = resample_row_linear_antigravity(row, target_length)
    means = prepared.mean(axis=1, keepdims=True)
    scales = prepared.std(axis=1, keepdims=True)
    if np.any(scales == 0.0) or not np.isfinite(scales).all():
        raise ValueError("finite non-constant rows are required")
    standardized = (prepared - means) / scales
    if not np.isfinite(standardized).all():
        raise ValueError("standardized rows must be finite")
    return standardized


def prepare_dataset_values_antigravity(
    connection: sqlite3.Connection,
    dataset: str,
) -> PreparedDatasetAntigravity:
    """Prepare TRAIN-normal and TEST values while returning no TEST labels."""
    if type(connection) is not sqlite3.Connection:
        raise ValueError("connection must be an exact sqlite3.Connection")
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset must be an exact non-empty string")
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    test_indices: list[int] = []
    raw_train = 0
    raw_test = 0
    excluded_train = 0
    excluded_test = 0
    for split, instance_index, blob in connection.execute(
        _VALUE_SQL_ANTIGRAVITY, (dataset,)
    ):
        if split not in ("TRAIN", "TEST") or type(instance_index) is not int:
            raise ValueError("source row identity is invalid")
        row = _decode_finite_row_antigravity(blob)
        if split == "TRAIN":
            raw_train += 1
            if row is None:
                excluded_train += 1
            else:
                train.append(row)
        else:
            raw_test += 1
            if row is None:
                excluded_test += 1
            else:
                test.append(row)
                test_indices.append(instance_index)
    if not train or not test:
        raise ValueError("finite TRAIN normal and TEST rows are required")
    target_length = lower_median_length_antigravity(tuple(row.size for row in train))
    return PreparedDatasetAntigravity(
        dataset=dataset,
        train_rows=_standardize_rows_antigravity(train, target_length),
        test_rows=_standardize_rows_antigravity(test, target_length),
        test_indices=tuple(test_indices),
        target_length=target_length,
        raw_train_rows=raw_train,
        excluded_train_rows=excluded_train,
        raw_test_rows=raw_test,
        excluded_test_rows=excluded_test,
    )


def audit_recovery_preprocessing_antigravity(
    connection: sqlite3.Connection,
    datasets: tuple[str, ...],
) -> dict[str, object]:
    """Run the value-only preparation over the exact ordered recovery population."""
    if type(datasets) is not tuple or datasets != tuple(sorted(set(datasets))):
        raise ValueError("datasets must be a sorted unique exact tuple")
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        try:
            prepared = prepare_dataset_values_antigravity(connection, dataset)
            rows.append(
                {
                    "dataset": dataset,
                    "status": "ok",
                    "target_length": prepared.target_length,
                    "raw_train_rows": prepared.raw_train_rows,
                    "retained_train_rows": len(prepared.train_rows),
                    "excluded_train_rows": prepared.excluded_train_rows,
                    "raw_test_rows": prepared.raw_test_rows,
                    "retained_test_rows": len(prepared.test_rows),
                    "excluded_test_rows": prepared.excluded_test_rows,
                }
            )
        except Exception as error:
            rows.append(
                {
                    "dataset": dataset,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    success = [row for row in rows if row["status"] == "ok"]
    return {
        "schema": "exp479_preprocessing_audit_v1",
        "summary": {
            "dataset_count": len(rows),
            "success": len(success),
            "failures": len(rows) - len(success),
            "excluded_train_rows": sum(
                int(row["excluded_train_rows"]) for row in success
            ),
            "excluded_test_rows": sum(
                int(row["excluded_test_rows"]) for row in success
            ),
        },
        "rows": rows,
    }


def render_task2_dashboard_antigravity(
    population: PopulationAuditAntigravity,
    manifest: Mapping[str, object],
    audit: Mapping[str, object],
) -> str:
    """Render the dashboard after all 155 value-only preparations pass."""
    if type(population) is not PopulationAuditAntigravity:
        raise ValueError("population must be an exact PopulationAuditAntigravity")
    if type(manifest) is not dict or manifest.get("schema") != "exp479_population_manifest_v1":
        raise ValueError("population manifest is invalid")
    if type(audit) is not dict or audit.get("schema") != "exp479_preprocessing_audit_v1":
        raise ValueError("preprocessing audit is invalid")
    summary = audit.get("summary")
    if type(summary) is not dict:
        raise ValueError("preprocessing summary is invalid")
    expected = {
        "dataset_count": 155,
        "success": 155,
        "failures": 0,
        "excluded_train_rows": 164,
        "excluded_test_rows": 235,
    }
    if summary != expected:
        raise ValueError("preprocessing summary differs from the verified Task 2 result")
    target_hash = manifest["recovery_dataset_sha256"]
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Exp479 성능 복구 대시보드</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1000px; margin: 32px auto; padding: 0 20px 50px; line-height: 1.62; color: #182230; }}
    h1,h2 {{ line-height:1.25; }} .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; }}
    .card {{ border:1px solid #d8dee8; border-radius:9px; padding:14px; }} .big {{ font-size:1.7rem; font-weight:750; }}
    .pass {{ border-left:5px solid #08783e; background:#f2fbf6; padding:12px 16px; }}
    .pending {{ border-left:5px solid #b54708; background:#fff8f0; padding:12px 16px; }}
    code {{ background:#eef2f7; padding:2px 5px; border-radius:4px; }} label {{ display:block; margin:8px 0; }} input {{ margin-right:8px; }}
  </style>
</head>
<body>
  <h1>Exp479 성능 미산출 155개 복구</h1>
  <p class=\"pass\"><strong>현재 상태:</strong> Task 2 완료. 155개 모두 Numba 전처리에 성공했고 실패는 없습니다.</p>
  <div class=\"cards\">
    <div class=\"card\"><div>복구 대상</div><div class=\"big\">155</div></div>
    <div class=\"card\"><div>전처리 결과</div><div class=\"big\">성공 155</div></div>
    <div class=\"card\"><div>실패</div><div class=\"big\">실패 0</div></div>
    <div class=\"card\"><div>TRAIN 비유한 행 제외</div><div class=\"big\">164</div></div>
    <div class=\"card\"><div>TEST 비유한 행 제외</div><div class=\"big\">235</div></div>
  </div>
  <h2>무엇을 확인했나</h2>
  <ul>
    <li>파생 데이터셋 내부의 TRAIN <code>label=0</code>만 사용했습니다.</li>
    <li>NaN 또는 Inf가 있는 행은 값을 채우지 않고 행 전체를 제외했습니다.</li>
    <li>남은 TRAIN 길이의 작은 쪽 중앙값으로 Numba 선형 재표본화했습니다.</li>
    <li>남은 TRAIN은 데이터셋마다 최소 8개, TEST는 최소 38개였습니다.</li>
    <li>목표 길이는 24~467이었고, 같은 길이 행은 보간 단계에서 정확히 복사됐습니다.</li>
  </ul>
  <p>대상 목록 SHA256: <code>{target_hash}</code></p>
  <p class=\"pending\"><strong>성능 상태:</strong> 아직 네 방법의 AUC와 ZF-TP는 계산하지 않았습니다. 다음 Task 3에서 실제 성능 평가를 구현합니다.</p>
  <h2>진행 체크</h2>
  <label><input type=\"checkbox\" checked disabled> Exp479 설계 승인</label>
  <label><input type=\"checkbox\" checked disabled> 파생 데이터셋 정상 라벨 의미 확인</label>
  <label><input type=\"checkbox\" checked disabled> 구현 계획서 작성</label>
  <label><input type=\"checkbox\" checked disabled> 구현 계획 승인</label>
  <label><input type=\"checkbox\" checked disabled> Task 1 — 대상 155개와 실행 계약 고정</label>
  <label><input type=\"checkbox\" checked disabled> Task 2 — Numba 전처리와 155개 감사</label>
  <label><input type=\"checkbox\" disabled> Task 3 — 네 후보 성능 평가</label>
  <label><input type=\"checkbox\" disabled> Task 4 — MLX 적용 여부 판정</label>
  <label><input type=\"checkbox\" disabled> Task 5 — 실행기와 소규모 검증</label>
  <label><input type=\"checkbox\" disabled> Task 6 — 공식 155개 실행과 1,117개 통합</label>
</body>
</html>
"""


__all__ = [
    "audit_recovery_preprocessing_antigravity",
    "lower_median_length_antigravity",
    "prepare_dataset_values_antigravity",
    "render_task2_dashboard_antigravity",
    "resample_row_linear_antigravity",
]
