"""Validated smoke and gated official execution for Exp480."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from .acceleration_antigravity import assess_mlx_antigravity
from .config_antigravity import (
    DASHBOARD_PATH_ANTIGRAVITY,
    EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    IMPLEMENTATION_REPORT_PATH_ANTIGRAVITY,
    MLX_ASSESSMENT_PATH_ANTIGRAVITY,
    OFFICIAL_RESULT_PATH_ANTIGRAVITY,
    RUN_MANIFEST_PATH_ANTIGRAVITY,
    SMOKE_RESULT_PATH_ANTIGRAVITY,
)
from .contracts_antigravity import EvidenceRowAntigravity
from .evidence_antigravity import (
    load_evidence_database_antigravity,
    validate_evidence_database_antigravity,
)
from .evaluation_antigravity import (
    EVALUATION_BACKEND_ANTIGRAVITY,
    evaluate_out_of_family_antigravity,
)


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


def _write_text_atomic_antigravity(path: Path, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + "_temporary_antigravity" + target.suffix)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)


def run_validate_antigravity() -> dict[str, object]:
    """Validate the exact derived evidence census without evaluating policies."""
    validation = validate_evidence_database_antigravity(
        EVIDENCE_DATABASE_PATH_ANTIGRAVITY
    )
    return {
        "status": "ready",
        "datasets": validation["datasets"],
        "families": validation["families"],
        "candidate_metrics": validation["candidate_metrics"],
    }


def render_dashboard_antigravity(result: Mapping[str, object]) -> str:
    """Render truthful task state; smoke is never displayed as official evidence."""
    state = str(result.get("state", "task5_pending"))
    task5_complete = bool(result.get("task5_complete", False))
    official_approved = state in {"official_pending_verification", "official_verified"}
    result_verified = state == "official_verified"
    evaluation = result.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    gates = evaluation.get("gates", {})
    summaries = evaluation.get("summaries", {})
    if not isinstance(gates, Mapping):
        gates = {}
    if not isinstance(summaries, Mapping):
        summaries = {}
    qualified = bool(gates.get("overall_pass", False))
    if state == "smoke_complete":
        status = "Task 5 smoke 완료 — 공식 성능 결과가 아닙니다."
    elif state == "official_pending_verification":
        status = "Task 6 실행 승인 반영 — 공식 결과 독립 검증 대기"
    elif result_verified:
        status = "공식 반영 준비" if qualified else "개선 실패 보존"
    else:
        status = "Task 5 구현 중"
    task5_checked = " checked" if task5_complete else ""
    task6_approval_checked = " checked" if official_approved else ""
    task6_verification_checked = " checked" if result_verified else ""
    fallback_reason = str(result.get("backend_fallback_reason", ""))
    if fallback_reason.startswith("mlx_unavailable:"):
        backend_note = "MLX 정확 연산 불가 → NumPy CPU 유지"
    elif fallback_reason:
        backend_note = f"MLX 기준 미충족 → NumPy CPU 유지 ({fallback_reason})"
    elif result.get("selected_backend") == EVALUATION_BACKEND_ANTIGRAVITY:
        backend_note = "공식 평가 계산은 NumPy CPU 고정"
    else:
        backend_note = "MLX 기준 통과" if result.get("selected_backend") == "mlx" else "가속 판정 대기"

    exp480 = summaries.get("exp480", {})
    if not isinstance(exp480, Mapping):
        exp480 = {}
    zf_total = int(exp480.get("zf_total", 0))
    family_mean_zf = float(exp480.get("family_mean_zf", 0.0))
    dataset_mean_auc = float(exp480.get("dataset_mean_auc", 0.0))
    family_mean_auc = float(exp480.get("family_mean_auc", 0.0))
    return f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Exp480 family 균형 라우팅 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1050px;margin:30px auto;padding:0 20px 60px;line-height:1.62;color:#182230}}
.status{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:9px;padding:14px}}.big{{font-size:1.7rem;font-weight:750}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}
</style></head><body><h1>Exp480 family 균형 경험 라우팅</h1>
<p class=\"status\"><strong>{escape(status)}</strong></p>
<div class=\"cards\"><div class=\"card\"><div>평가 family</div><div class=\"big\">{int(result.get('family_count', 0))}</div></div><div class=\"card\"><div>평가 데이터셋</div><div class=\"big\">{int(result.get('dataset_count', 0))}</div></div><div class=\"card\"><div>backend</div><div class=\"big\">{escape(str(result.get('selected_backend', '미판정')))}</div></div></div>
<p><strong>가속 판정:</strong> {escape(backend_note)}</p>
<h2>핵심 결과</h2><div class=\"cards\"><div class=\"card\"><div>ZF 합계</div><div class=\"big\">{zf_total}</div></div><div class=\"card\"><div>family 평균 ZF</div><div class=\"big\">{family_mean_zf:.6g}</div></div><div class=\"card\"><div>평균 AUC</div><div class=\"big\">{dataset_mean_auc:.6g}</div></div><div class=\"card\"><div>family 평균 AUC</div><div class=\"big\">{family_mean_auc:.6g}</div></div></div>
<h2>이름 사용 경계</h2><p>과거 family ID는 중복 투표 제거와 검증 분리에만 씁니다. 새 대상의 이름이나 family는 라우팅 입력이 아니며 TRAIN 데이터의 7개 숫자 특징만 사용합니다.</p>
<h2>진행 체크</h2>
<label><input type=\"checkbox\" checked disabled> 설계 및 구현 계획 승인</label>
<label><input type=\"checkbox\" checked disabled> Task 1 — 코드맵과 변경 경계 고정</label>
<label><input type=\"checkbox\" checked disabled> Task 2 — 근거 병합과 파생 DB</label>
<label><input type=\"checkbox\" checked disabled> Task 3 — family 균형 정책</label>
<label><input type=\"checkbox\" checked disabled> Task 4 — out-of-family 평가기</label>
<label><input type=\"checkbox\"{task5_checked} disabled> Task 5 — 실행기와 smoke 검증</label>
<label><input type=\"checkbox\"{task6_approval_checked} disabled> Task 6 — 공식 1,117개 실행 승인</label>
<label><input type=\"checkbox\"{task6_verification_checked} disabled> Task 6 — 공식 결과 검증</label>
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


def build_official_result_antigravity(
    rows: Sequence[EvidenceRowAntigravity],
    *,
    evidence_database_sha256: str,
) -> dict[str, object]:
    """Build deterministic qualification bytes using the actual NumPy-only backend."""
    evaluation = evaluate_out_of_family_antigravity(rows)
    if evaluation.get("backend") != EVALUATION_BACKEND_ANTIGRAVITY:
        raise RuntimeError("official evaluation backend changed from NumPy CPU")
    return {
        "schema": "exp480_official_qualification_v1",
        "state": "official_pending_verification",
        "task5_complete": True,
        "family_count": evaluation["family_count"],
        "dataset_count": evaluation["dataset_count"],
        "selected_backend": EVALUATION_BACKEND_ANTIGRAVITY,
        "evidence_database_sha256": evidence_database_sha256,
        "evaluation": evaluation,
    }


def build_run_manifest_antigravity(
    *,
    result_bytes: bytes,
    acceleration: Mapping[str, object],
    started_at_utc: str,
    completed_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Record full non-deterministic execution and acceleration evidence separately."""
    required = {
        "mlx_version",
        "mlx_metal_version",
        "device",
        "cpu_seconds",
        "mlx_seconds",
        "finite",
        "candidate_indices_equal",
        "tolerance",
        "max_abs_error",
        "speedup",
        "selected_backend",
        "fallback_reason",
    }
    missing = sorted(required - set(acceleration))
    if missing:
        raise RuntimeError(
            "acceleration assessment is incomplete: " + ", ".join(missing)
        )
    recommended_backend = str(acceleration["selected_backend"])
    if recommended_backend == EVALUATION_BACKEND_ANTIGRAVITY:
        fallback_reason = acceleration.get("fallback_reason")
        fallback_detail = acceleration.get("fallback_detail")
    else:
        fallback_reason = "mlx_recommendation_unwired_numpy_evaluation"
        fallback_detail = (
            "The official evaluation is NumPy-only; MLX benchmark output is not "
            "wired into result computation."
        )
    return {
        "schema": "exp480_run_manifest_v1",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "result_sha256": sha256(result_bytes).hexdigest(),
        "acceleration_evidence": {
            "mlx_version": acceleration["mlx_version"],
            "mlx_metal_version": acceleration["mlx_metal_version"],
            "device": acceleration["device"],
            "cpu_seconds": acceleration["cpu_seconds"],
            "mlx_seconds": acceleration["mlx_seconds"],
            "finite": acceleration["finite"],
            "candidate_indices_equal": acceleration["candidate_indices_equal"],
            "tolerance": acceleration["tolerance"],
            "max_abs_error": acceleration["max_abs_error"],
            "speedup": acceleration["speedup"],
            "recommended_backend": recommended_backend,
            "actual_result_backend": EVALUATION_BACKEND_ANTIGRAVITY,
            "fallback_reason": fallback_reason,
            "fallback_detail": fallback_detail,
        },
    }


def _summary_cell_antigravity(summary: Mapping[str, object], key: str) -> str:
    value = summary.get(key, 0)
    if key == "zf_total":
        return str(int(value))
    return f"{float(value):.6g}"


def render_implementation_report_antigravity(result: Mapping[str, object]) -> str:
    """Explain pass/fail comparisons in simple Korean without changing the router."""
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("official result requires evaluation")
    summaries = evaluation.get("summaries")
    gates = evaluation.get("gates")
    if not isinstance(summaries, Mapping) or not isinstance(gates, Mapping):
        raise ValueError("official result requires summaries and gates")
    passed = bool(gates.get("overall_pass", False))
    verdict = "공식 반영 준비" if passed else "개선 실패 보존"
    labels = (
        ("mini", "항상 MiniRocket"),
        ("row_knn", "현재 row-KNN"),
        ("exp480", "Exp480"),
    )
    rows = []
    for key, label in labels:
        summary = summaries.get(key)
        if not isinstance(summary, Mapping):
            raise ValueError(f"official result is missing {key} summary")
        rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{_summary_cell_antigravity(summary, 'zf_total')}</td>"
            f"<td>{_summary_cell_antigravity(summary, 'family_mean_zf')}</td>"
            f"<td>{_summary_cell_antigravity(summary, 'dataset_mean_auc')}</td>"
            f"<td>{_summary_cell_antigravity(summary, 'family_mean_auc')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Exp480 구현 결과 보고서</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:960px;margin:30px auto;padding:0 20px 60px;line-height:1.65;color:#182230}}.verdict{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee8;padding:8px;text-align:left}}
</style></head><body><h1>Exp480 구현 결과</h1>
<p class=\"verdict\"><strong>{verdict}</strong></p>
<p>FN(False Negative)은 실제 이상을 놓친 수라서 낮을수록 좋습니다. ZF는 거짓 경보 없이 잡은 이상 수를 나타내며 높을수록 좋습니다. 이 실험은 ZF 합계와 family 평균을 두 기준 방식과 비교했습니다.</p>
<p>AUC는 이상과 정상을 구분하는 정도입니다. 전체 평균 AUC와 family 평균 AUC가 두 기준 방식보다 낮아지지 않는지도 함께 확인했습니다.</p>
<table><thead><tr><th>방식</th><th>ZF 합계</th><th>family 평균 ZF</th><th>평균 AUC</th><th>family 평균 AUC</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>공식 라우터는 바꾸지 않았습니다. 이 보고서는 자격 검증 결과만 설명합니다.</p>
</body></html>"""


def run_official_antigravity(
    *,
    evidence_path: Path = EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    result_path: Path = OFFICIAL_RESULT_PATH_ANTIGRAVITY,
    manifest_path: Path = RUN_MANIFEST_PATH_ANTIGRAVITY,
    dashboard_path: Path = DASHBOARD_PATH_ANTIGRAVITY,
    assessment_path: Path = MLX_ASSESSMENT_PATH_ANTIGRAVITY,
) -> dict[str, object]:
    """Task 6 entry point; caller must enforce the separate user approval."""
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    validate_evidence_database_antigravity(evidence_path)
    rows = load_evidence_database_antigravity(evidence_path)
    acceleration = assess_mlx_antigravity(
        rows, assessment_path=assessment_path
    )
    evidence_digest = sha256(Path(evidence_path).read_bytes()).hexdigest()
    result = build_official_result_antigravity(
        rows, evidence_database_sha256=evidence_digest
    )
    result_bytes = _json_bytes_antigravity(result)
    completed_at = datetime.now(timezone.utc)
    manifest = build_run_manifest_antigravity(
        result_bytes=result_bytes,
        acceleration=acceleration,
        started_at_utc=started_at.isoformat(),
        completed_at_utc=completed_at.isoformat(),
        elapsed_seconds=perf_counter() - started,
    )
    write_json_atomic_antigravity(result_path, result)
    write_json_atomic_antigravity(manifest_path, manifest)
    _write_text_atomic_antigravity(
        dashboard_path, render_dashboard_antigravity(result)
    )
    return result


def verify_official_antigravity(
    *,
    evidence_path: Path = EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    result_path: Path = OFFICIAL_RESULT_PATH_ANTIGRAVITY,
    manifest_path: Path = RUN_MANIFEST_PATH_ANTIGRAVITY,
    dashboard_path: Path = DASHBOARD_PATH_ANTIGRAVITY,
    report_path: Path = IMPLEMENTATION_REPORT_PATH_ANTIGRAVITY,
) -> dict[str, object]:
    """Independently recompute CPU bytes, fail on drift, then publish final HTML."""
    validate_evidence_database_antigravity(evidence_path)
    rows = load_evidence_database_antigravity(evidence_path)
    evidence_digest = sha256(Path(evidence_path).read_bytes()).hexdigest()
    recomputed_payload = build_official_result_antigravity(
        rows, evidence_database_sha256=evidence_digest
    )
    recomputed = _json_bytes_antigravity(recomputed_payload)
    stored = Path(result_path).read_bytes()
    if stored != recomputed:
        raise RuntimeError("official result bytes do not match CPU recomputation")
    stored_digest = sha256(stored).hexdigest()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("result_sha256") != stored_digest:
        raise RuntimeError("manifest result SHA does not match official result")
    final_view = dict(recomputed_payload)
    final_view["state"] = "official_verified"
    _write_text_atomic_antigravity(
        dashboard_path, render_dashboard_antigravity(final_view)
    )
    _write_text_atomic_antigravity(
        report_path, render_implementation_report_antigravity(recomputed_payload)
    )
    return {
        "stored_result_sha256": sha256(stored).hexdigest(),
        "recomputed_result_sha256": sha256(recomputed).hexdigest(),
        "manifest_result_sha256": str(manifest["result_sha256"]),
        "match": True,
        "qualification_pass": bool(
            recomputed_payload["evaluation"]["gates"]["overall_pass"]
        ),
    }


__all__ = [
    "build_official_result_antigravity",
    "build_run_manifest_antigravity",
    "render_dashboard_antigravity",
    "render_implementation_report_antigravity",
    "run_official_antigravity",
    "run_smoke_antigravity",
    "run_validate_antigravity",
    "verify_official_antigravity",
    "write_json_atomic_antigravity",
]
