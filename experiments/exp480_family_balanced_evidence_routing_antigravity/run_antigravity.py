"""Validated smoke and gated official execution for Exp480."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Mapping

from .acceleration_antigravity import assess_mlx_antigravity
from .config_antigravity import (
    DASHBOARD_PATH_ANTIGRAVITY,
    EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    MLX_ASSESSMENT_PATH_ANTIGRAVITY,
    OFFICIAL_RESULT_PATH_ANTIGRAVITY,
    RUN_MANIFEST_PATH_ANTIGRAVITY,
    SMOKE_RESULT_PATH_ANTIGRAVITY,
)
from .evidence_antigravity import load_evidence_database_antigravity
from .evaluation_antigravity import evaluate_out_of_family_antigravity


def _json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def write_json_atomic_antigravity(path: Path, payload: object) -> None:
    """Expose only a complete strict JSON target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + "_temporary_antigravity" + target.suffix)
    temporary.write_bytes(_json_bytes_antigravity(payload))
    temporary.replace(target)


def run_validate_antigravity() -> dict[str, object]:
    """Validate the exact derived evidence census without evaluating policies."""
    rows = load_evidence_database_antigravity(EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    with sqlite3.connect(
        f"file:{EVIDENCE_DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    ) as connection:
        metric_count = int(
            connection.execute("SELECT count(*) FROM candidate_metrics").fetchone()[0]
        )
    result = {
        "status": "ready",
        "datasets": len(rows),
        "families": len({row.source_family for row in rows}),
        "candidate_metrics": metric_count,
    }
    if result != {
        "status": "ready",
        "datasets": 1117,
        "families": 128,
        "candidate_metrics": 4468,
    }:
        raise RuntimeError("Exp480 evidence census is not ready")
    return result


def render_dashboard_antigravity(result: Mapping[str, object]) -> str:
    """Render truthful task state; smoke is never displayed as official evidence."""
    state = str(result.get("state", "task5_pending"))
    task5_complete = bool(result.get("task5_complete", False))
    official_complete = state == "official_complete"
    if state == "smoke_complete":
        status = "Task 5 smoke 완료 — 공식 성능 결과가 아닙니다."
    elif official_complete:
        status = "Task 6 공식 자격 검증 완료"
    else:
        status = "Task 5 구현 중"
    task5_checked = " checked" if task5_complete else ""
    task6_checked = " checked" if official_complete else ""
    fallback_reason = str(result.get("backend_fallback_reason", ""))
    if fallback_reason.startswith("mlx_unavailable:"):
        backend_note = "MLX 정확 연산 불가 → NumPy CPU 유지"
    elif fallback_reason:
        backend_note = f"MLX 기준 미충족 → NumPy CPU 유지 ({fallback_reason})"
    else:
        backend_note = "MLX 기준 통과" if result.get("selected_backend") == "mlx" else "가속 판정 대기"
    return f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Exp480 family 균형 라우팅 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1050px;margin:30px auto;padding:0 20px 60px;line-height:1.62;color:#182230}}
.status{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:9px;padding:14px}}.big{{font-size:1.7rem;font-weight:750}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}
</style></head><body><h1>Exp480 family 균형 경험 라우팅</h1>
<p class=\"status\"><strong>{escape(status)}</strong></p>
<div class=\"cards\"><div class=\"card\"><div>평가 family</div><div class=\"big\">{int(result.get('family_count', 0))}</div></div><div class=\"card\"><div>평가 데이터셋</div><div class=\"big\">{int(result.get('dataset_count', 0))}</div></div><div class=\"card\"><div>backend</div><div class=\"big\">{escape(str(result.get('selected_backend', '미판정')))}</div></div></div>
<p><strong>가속 판정:</strong> {escape(backend_note)}</p>
<h2>이름 사용 경계</h2><p>과거 family ID는 중복 투표 제거와 검증 분리에만 씁니다. 새 대상의 이름이나 family는 라우팅 입력이 아니며 TRAIN 데이터의 7개 숫자 특징만 사용합니다.</p>
<h2>진행 체크</h2>
<label><input type=\"checkbox\" checked disabled> 설계 및 구현 계획 승인</label>
<label><input type=\"checkbox\" checked disabled> Task 1 — 코드맵과 변경 경계 고정</label>
<label><input type=\"checkbox\" checked disabled> Task 2 — 근거 병합과 파생 DB</label>
<label><input type=\"checkbox\" checked disabled> Task 3 — family 균형 정책</label>
<label><input type=\"checkbox\" checked disabled> Task 4 — out-of-family 평가기</label>
<label><input type=\"checkbox\"{task5_checked} disabled> Task 5 — 실행기와 smoke 검증</label>
<label><input type=\"checkbox\" disabled> Task 6 — 공식 1,117개 실행 승인</label>
<label><input type=\"checkbox\"{task6_checked} disabled> Task 6 — 공식 결과 검증</label>
</body></html>"""


def run_smoke_antigravity(
    *,
    family_count: int = 8,
    result_path: Path = SMOKE_RESULT_PATH_ANTIGRAVITY,
    dashboard_path: Path = DASHBOARD_PATH_ANTIGRAVITY,
    assessment_path: Path = MLX_ASSESSMENT_PATH_ANTIGRAVITY,
) -> dict[str, object]:
    """Run a small real-family integration smoke without official outputs."""
    rows = load_evidence_database_antigravity(EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    families = tuple(sorted({row.source_family for row in rows}))
    if type(family_count) is not int or not 2 <= family_count <= len(families):
        raise ValueError("family_count must select between 2 and 128 families")
    selected = set(families[:family_count])
    subset = tuple(row for row in rows if row.source_family in selected)
    support = min(31, family_count - 1)
    evaluation = evaluate_out_of_family_antigravity(
        subset,
        family_policy_kwargs={"k": support, "min_support": support},
        row_policy_kwargs={"k": support, "min_support": support},
    )
    acceleration = assess_mlx_antigravity(subset, assessment_path=assessment_path)
    result = {
        "schema": "exp480_smoke_result_v1",
        "state": "smoke_complete",
        "task5_complete": True,
        "family_count": evaluation["family_count"],
        "dataset_count": evaluation["dataset_count"],
        "selected_backend": acceleration["selected_backend"],
        "backend_fallback_reason": acceleration["fallback_reason"],
        "selected_families": sorted(selected),
        "evaluation": evaluation,
    }
    write_json_atomic_antigravity(result_path, result)
    Path(dashboard_path).write_text(render_dashboard_antigravity(result), encoding="utf-8")
    return result


def _official_payload_antigravity(selected_backend: str) -> dict[str, object]:
    rows = load_evidence_database_antigravity(EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    evaluation = evaluate_out_of_family_antigravity(rows)
    return {
        "schema": "exp480_official_qualification_v1",
        "state": "official_complete",
        "task5_complete": True,
        "family_count": evaluation["family_count"],
        "dataset_count": evaluation["dataset_count"],
        "selected_backend": selected_backend,
        "evidence_database_sha256": sha256(
            EVIDENCE_DATABASE_PATH_ANTIGRAVITY.read_bytes()
        ).hexdigest(),
        "evaluation": evaluation,
    }


def run_official_antigravity() -> dict[str, object]:
    """Task 6 entry point; caller must enforce the separate user approval."""
    started = perf_counter()
    rows = load_evidence_database_antigravity(EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    acceleration = assess_mlx_antigravity(
        rows, assessment_path=MLX_ASSESSMENT_PATH_ANTIGRAVITY
    )
    result = _official_payload_antigravity(str(acceleration["selected_backend"]))
    write_json_atomic_antigravity(OFFICIAL_RESULT_PATH_ANTIGRAVITY, result)
    write_json_atomic_antigravity(
        RUN_MANIFEST_PATH_ANTIGRAVITY,
        {
            "schema": "exp480_run_manifest_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "result_sha256": sha256(_json_bytes_antigravity(result)).hexdigest(),
        },
    )
    DASHBOARD_PATH_ANTIGRAVITY.write_text(
        render_dashboard_antigravity(result), encoding="utf-8"
    )
    return result


def verify_official_antigravity() -> dict[str, object]:
    """Recompute deterministic official bytes without overwriting the stored result."""
    stored = OFFICIAL_RESULT_PATH_ANTIGRAVITY.read_bytes()
    payload = json.loads(stored)
    recomputed = _json_bytes_antigravity(
        _official_payload_antigravity(str(payload["selected_backend"]))
    )
    return {
        "stored_result_sha256": sha256(stored).hexdigest(),
        "recomputed_result_sha256": sha256(recomputed).hexdigest(),
        "match": stored == recomputed,
    }


__all__ = [
    "render_dashboard_antigravity",
    "run_official_antigravity",
    "run_smoke_antigravity",
    "run_validate_antigravity",
    "verify_official_antigravity",
    "write_json_atomic_antigravity",
]
