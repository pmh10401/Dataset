"""Validated smoke and separately gated official execution for Exp481."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from experiments.exp480_family_balanced_evidence_routing_antigravity.contracts_antigravity import (
    EvidenceRowAntigravity,
)
from experiments.exp480_family_balanced_evidence_routing_antigravity.evidence_antigravity import (
    load_evidence_database_antigravity,
    validate_evidence_database_antigravity,
)

from .config_antigravity import (
    ACCELERATION_ASSESSMENT_PATH_ANTIGRAVITY,
    DASHBOARD_PATH_ANTIGRAVITY,
    EXPECTED_DATASETS_ANTIGRAVITY,
    EXPECTED_EXP480_EVIDENCE_SHA256_ANTIGRAVITY,
    EXPECTED_EXP480_RESULT_SHA256_ANTIGRAVITY,
    EXPECTED_FAMILIES_ANTIGRAVITY,
    EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    EXP480_RESULT_PATH_ANTIGRAVITY,
    OFFICIAL_RESULT_PATH_ANTIGRAVITY,
    RUN_MANIFEST_PATH_ANTIGRAVITY,
    SMOKE_RESULT_PATH_ANTIGRAVITY,
)
from .evaluation_antigravity import evaluate_nested_out_of_family_antigravity


_BACKEND_KEYS_ANTIGRAVITY = (
    "selected_backend",
    "fallback_reason",
    "candidate_index_parity",
    "decision_parity",
    "finite_outputs",
    "exact_frozen_stage_benchmarked",
    "numba_cold_jit_verified",
)
_NUMBA_GATE_KEYS_ANTIGRAVITY = (
    "candidate_index_parity",
    "statistics_parity",
    "decision_parity",
    "numerical_agreement",
    "finite_outputs",
    "exact_frozen_stage_benchmarked",
    "numba_cold_jit_verified",
    "numba_speed_gate_pass",
)


def json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_atomic_antigravity(path: Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + "_temporary_antigravity" + target.suffix)
    temporary.write_bytes(json_bytes_antigravity(payload))
    temporary.replace(target)


def _write_text_atomic_antigravity(path: Path, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + "_temporary_antigravity" + target.suffix)
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)


def _sha256_antigravity(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _load_backend_assessment_antigravity(
    path: Path = ACCELERATION_ASSESSMENT_PATH_ANTIGRAVITY,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise RuntimeError("backend assessment must be a JSON object")
    _validate_backend_payload_antigravity(payload)
    return payload


def _backend_result_antigravity(assessment: Mapping[str, object]) -> dict[str, object]:
    return {key: assessment[key] for key in _BACKEND_KEYS_ANTIGRAVITY}


def run_validate_antigravity() -> dict[str, object]:
    validation = validate_evidence_database_antigravity(
        EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY
    )
    if _sha256_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY) != EXPECTED_EXP480_EVIDENCE_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp480 evidence SHA changed")
    if _sha256_antigravity(EXP480_RESULT_PATH_ANTIGRAVITY) != EXPECTED_EXP480_RESULT_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp480 result SHA changed")
    assessment = _load_backend_assessment_antigravity()
    state = "ready_for_smoke"
    if SMOKE_RESULT_PATH_ANTIGRAVITY.exists():
        smoke = json.loads(SMOKE_RESULT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
        if (
            smoke.get("state") != "smoke_nonofficial"
            or smoke.get("family_count") != 8
            or smoke.get("selected_backend") != assessment["selected_backend"]
            or smoke.get("fallback_reason") != assessment["fallback_reason"]
        ):
            raise RuntimeError("Exp481 smoke result is stale or malformed")
        _validate_evaluation_antigravity(smoke["evaluation"])
        state = "ready_for_official"
    return {
        "state": state,
        "datasets": validation["datasets"],
        "families": validation["families"],
        "selected_backend": assessment["selected_backend"],
        "fallback_reason": assessment["fallback_reason"],
    }


def _evaluate_antigravity(
    rows: Sequence[EvidenceRowAntigravity], assessment: Mapping[str, object]
) -> dict[str, object]:
    backend = "numba_cpu" if assessment["selected_backend"] == "numba_cpu" else "numpy_reference"
    return evaluate_nested_out_of_family_antigravity(
        rows, residual_policy_kwargs={"backend": backend}
    )


def _validate_evaluation_antigravity(evaluation: object) -> None:
    if not isinstance(evaluation, Mapping):
        raise RuntimeError("evaluation must be an object")
    leakage = evaluation.get("leakage")
    if not isinstance(leakage, Mapping):
        raise RuntimeError("evaluation leakage audit is missing")
    expected = {
        "family_overlap_count": 0,
        "outer_family_seen_by_inner_count": 0,
        "route_input_key_count": 7,
        "dataset_name_feature_count": 0,
        "held_out_score_feature_count": 0,
        "held_out_scores_read_after_routing": True,
    }
    if any(leakage.get(key) != value for key, value in expected.items()):
        raise RuntimeError("evaluation leakage boundary failed")


def _validate_official_result_antigravity(result: Mapping[str, object]) -> None:
    if (result.get("dataset_count"), result.get("family_count")) != (
        EXPECTED_DATASETS_ANTIGRAVITY,
        EXPECTED_FAMILIES_ANTIGRAVITY,
    ):
        raise RuntimeError("official evaluation population changed")
    evaluation = result.get("evaluation")
    _validate_evaluation_antigravity(evaluation)
    assert isinstance(evaluation, Mapping)
    if len(evaluation.get("decisions", ())) != EXPECTED_DATASETS_ANTIGRAVITY:
        raise RuntimeError("official decision count changed")
    if len(evaluation.get("inner_gates_by_outer_family", {})) != EXPECTED_FAMILIES_ANTIGRAVITY:
        raise RuntimeError("official inner family count changed")


def _result_antigravity(
    *, state: str, evaluation: Mapping[str, object], assessment: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema": "exp481_qualification_v1",
        "state": state,
        "task5_complete": True,
        "family_count": int(evaluation["family_count"]),
        "dataset_count": int(evaluation["dataset_count"]),
        **_backend_result_antigravity(assessment),
        "exp480_evidence_sha256": EXPECTED_EXP480_EVIDENCE_SHA256_ANTIGRAVITY,
        "exp480_result_sha256": EXPECTED_EXP480_RESULT_SHA256_ANTIGRAVITY,
        "evaluation": dict(evaluation),
    }


def run_smoke_antigravity(
    *, family_count: int = 8, output_dir: Path | None = None
) -> dict[str, object]:
    run_validate_antigravity()
    rows = load_evidence_database_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    families = tuple(sorted({row.source_family for row in rows}))
    if type(family_count) is not int or not 3 <= family_count <= len(families):
        raise ValueError("family_count must select between 3 and 128 families")
    selected = set(families[:family_count])
    subset = tuple(row for row in rows if row.source_family in selected)
    assessment = _load_backend_assessment_antigravity()
    evaluation = _evaluate_antigravity(subset, assessment)
    _validate_evaluation_antigravity(evaluation)
    result = _result_antigravity(
        state="smoke_nonofficial", evaluation=evaluation, assessment=assessment
    )
    result["selected_families"] = sorted(selected)
    result_path = (
        Path(output_dir) / "smoke_results_antigravity.json"
        if output_dir is not None
        else SMOKE_RESULT_PATH_ANTIGRAVITY
    )
    dashboard_path = (
        Path(output_dir) / "EXP481_DASHBOARD_antigravity.html"
        if output_dir is not None
        else DASHBOARD_PATH_ANTIGRAVITY
    )
    write_json_atomic_antigravity(result_path, result)
    _write_text_atomic_antigravity(dashboard_path, render_dashboard_antigravity(result))
    return result


def build_official_result_antigravity(
    evaluation: Mapping[str, object],
    *,
    backend_assessment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    assessment = backend_assessment or _load_backend_assessment_antigravity()
    return _result_antigravity(
        state="official_pending_verification",
        evaluation=evaluation,
        assessment=assessment,
    )


def build_run_manifest_antigravity(
    *,
    result_bytes: bytes,
    backend_assessment: Mapping[str, object],
    started_at_utc: str,
    completed_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be nonnegative")
    manifest = {
        "schema": "exp481_run_manifest_v1",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "result_sha256": sha256(result_bytes).hexdigest(),
        "backend_assessment_sha256": sha256(
            json_bytes_antigravity(dict(backend_assessment))
        ).hexdigest(),
        "backend_assessment": dict(backend_assessment),
    }
    validate_run_manifest_antigravity(
        manifest,
        expected_result_sha256=manifest["result_sha256"],
        expected_backend_assessment=backend_assessment,
    )
    return manifest


def validate_run_manifest_antigravity(
    manifest: object,
    *,
    expected_result_sha256: str,
    expected_backend_assessment: Mapping[str, object] | None = None,
) -> None:
    if type(manifest) is not dict or set(manifest) != {
        "schema",
        "started_at_utc",
        "completed_at_utc",
        "elapsed_seconds",
        "result_sha256",
        "backend_assessment_sha256",
        "backend_assessment",
    }:
        raise RuntimeError("manifest fields are incomplete")
    if manifest["schema"] != "exp481_run_manifest_v1":
        raise RuntimeError("manifest schema changed")
    times = []
    for key in ("started_at_utc", "completed_at_utc"):
        try:
            parsed = datetime.fromisoformat(str(manifest[key]))
        except ValueError as error:
            raise RuntimeError(f"manifest {key} is invalid") from error
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise RuntimeError(f"manifest {key} must use UTC")
        times.append(parsed)
    elapsed = manifest["elapsed_seconds"]
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("manifest elapsed_seconds is invalid")
    if times[1] < times[0]:
        raise RuntimeError("manifest completion precedes start")
    if manifest["result_sha256"] != expected_result_sha256:
        raise RuntimeError("manifest result SHA does not match")
    assessment = manifest["backend_assessment"]
    if type(assessment) is not dict:
        raise RuntimeError("manifest backend assessment is invalid")
    assessment_digest = sha256(json_bytes_antigravity(assessment)).hexdigest()
    if manifest["backend_assessment_sha256"] != assessment_digest:
        raise RuntimeError("manifest backend assessment SHA does not match")
    _validate_backend_payload_antigravity(assessment)
    if expected_backend_assessment is not None and assessment != dict(expected_backend_assessment):
        raise RuntimeError("manifest backend assessment is not current")


def _validate_backend_payload_antigravity(payload: dict[str, object]) -> None:
    missing = [
        key
        for key in (*_BACKEND_KEYS_ANTIGRAVITY, *_NUMBA_GATE_KEYS_ANTIGRAVITY)
        if key not in payload
    ]
    if missing:
        raise RuntimeError("backend assessment fields missing: " + ", ".join(missing))
    if payload["selected_backend"] not in {"numpy_reference", "numba_cpu"}:
        raise RuntimeError("backend assessment selection is invalid")
    if type(payload["fallback_reason"]) is not str or not payload["fallback_reason"]:
        raise RuntimeError("backend assessment fallback reason is missing")
    if any(type(payload[key]) is not bool for key in _NUMBA_GATE_KEYS_ANTIGRAVITY):
        raise RuntimeError("backend assessment gate flags must be boolean")
    if payload["selected_backend"] == "numba_cpu" and not all(
        payload[key] for key in _NUMBA_GATE_KEYS_ANTIGRAVITY
    ):
        raise RuntimeError("Numba selected without every required gate")


def render_dashboard_antigravity(result: Mapping[str, object]) -> str:
    state = str(result.get("state", "task5_pending"))
    task5_checked = " checked" if result.get("task5_complete") else ""
    task6_checked = " checked" if state in {"official_pending_verification", "official_verified"} else ""
    verified_checked = " checked" if state == "official_verified" else ""
    status = {
        "smoke_nonofficial": "Task 5 smoke 완료 — 공식 성능 결과가 아닙니다.",
        "official_pending_verification": "Task 6 공식 실행 완료 — 검증 대기",
        "official_verified": "Task 6 공식 결과 검증 완료",
    }.get(state, "Task 5 구현 중")
    evaluation = result.get("evaluation", {})
    summaries = evaluation.get("summaries", {}) if isinstance(evaluation, Mapping) else {}
    exp481 = summaries.get("exp481", {}) if isinstance(summaries, Mapping) else {}
    changed = summaries.get("changed", {}) if isinstance(summaries, Mapping) else {}
    gates = evaluation.get("gates", {}) if isinstance(evaluation, Mapping) else {}
    passed = bool(gates.get("overall_pass", False)) if isinstance(gates, Mapping) else False
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp481 엄격 근거 잔여 보정 대시보드</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:960px;margin:30px auto;padding:0 20px 60px;line-height:1.62;color:#182230}}.status{{border-left:5px solid #175cd3;background:#eff8ff;padding:12px 16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}.card{{border:1px solid #d8dee8;border-radius:9px;padding:14px}}.big{{font-size:1.5rem;font-weight:750}}label{{display:block;margin:8px 0}}input{{margin-right:8px}}
</style></head><body><h1>Exp481 엄격 근거 잔여 보정</h1>
<p class="status"><strong>{escape(status)}</strong></p>
<div class="cards"><div class="card"><div>평가 데이터셋</div><div class="big">{int(result.get('dataset_count', 0))}</div></div><div class="card"><div>평가 family</div><div class="big">{int(result.get('family_count', 0))}</div></div><div class="card"><div>backend</div><div class="big">{escape(str(result.get('selected_backend', '미판정')))}</div></div><div class="card"><div>공식 기준 통과</div><div class="big">{'예' if passed else '아니요'}</div></div></div>
<h2>smoke 결과</h2><p>ZF 합계: <strong>{int(exp481.get('zf_total', 0))}</strong>, 평균 AUC: <strong>{float(exp481.get('dataset_mean_auc', 0.0)):.6g}</strong>, 실제 변경: <strong>{int(changed.get('count', 0))}</strong></p>
<p>가속 판정: <code>{escape(str(result.get('fallback_reason', '미판정')))}</code></p>
<h2>진행 체크</h2>
<label><input type="checkbox" checked disabled> Task 1 — 코드맵 및 경계 고정</label>
<label><input type="checkbox" checked disabled> Task 2 — strict evidence 계약</label>
<label><input type="checkbox" checked disabled> Task 3 — nested family 평가기</label>
<label><input type="checkbox" checked disabled> Task 4 — 가속 적합성 평가</label>
<label><input type="checkbox"{task5_checked} disabled> Task 5 — smoke 및 재현성 검증</label>
<label><input type="checkbox"{task6_checked} disabled> Task 6 — 공식 1,117개 실행</label>
<label><input type="checkbox"{verified_checked} disabled> Task 6 — 공식 결과 검증</label>
</body></html>"""


def run_official_antigravity() -> dict[str, object]:
    """Task 6 entry point. Call only after the separate user approval."""
    run_validate_antigravity()
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    rows = load_evidence_database_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    assessment = _load_backend_assessment_antigravity()
    result = build_official_result_antigravity(
        _evaluate_antigravity(rows, assessment), backend_assessment=assessment
    )
    _validate_official_result_antigravity(result)
    result_bytes = json_bytes_antigravity(result)
    completed_at = datetime.now(timezone.utc)
    manifest = build_run_manifest_antigravity(
        result_bytes=result_bytes,
        backend_assessment=assessment,
        started_at_utc=started_at.isoformat(),
        completed_at_utc=completed_at.isoformat(),
        elapsed_seconds=perf_counter() - started,
    )
    write_json_atomic_antigravity(OFFICIAL_RESULT_PATH_ANTIGRAVITY, result)
    write_json_atomic_antigravity(RUN_MANIFEST_PATH_ANTIGRAVITY, manifest)
    _write_text_atomic_antigravity(DASHBOARD_PATH_ANTIGRAVITY, render_dashboard_antigravity(result))
    return result


def verify_official_antigravity() -> dict[str, object]:
    """Recompute deterministic bytes and verify the separate manifest."""
    run_validate_antigravity()
    rows = load_evidence_database_antigravity(EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY)
    assessment = _load_backend_assessment_antigravity()
    recomputed = build_official_result_antigravity(
        _evaluate_antigravity(rows, assessment), backend_assessment=assessment
    )
    _validate_official_result_antigravity(recomputed)
    expected = json_bytes_antigravity(recomputed)
    stored = OFFICIAL_RESULT_PATH_ANTIGRAVITY.read_bytes()
    if stored != expected:
        raise RuntimeError("official result bytes do not match recomputation")
    manifest = json.loads(RUN_MANIFEST_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    validate_run_manifest_antigravity(
        manifest,
        expected_result_sha256=sha256(stored).hexdigest(),
        expected_backend_assessment=assessment,
    )
    verified = dict(recomputed)
    verified["state"] = "official_verified"
    _write_text_atomic_antigravity(DASHBOARD_PATH_ANTIGRAVITY, render_dashboard_antigravity(verified))
    return verified


__all__ = [
    "build_official_result_antigravity",
    "build_run_manifest_antigravity",
    "json_bytes_antigravity",
    "render_dashboard_antigravity",
    "run_official_antigravity",
    "run_smoke_antigravity",
    "run_validate_antigravity",
    "validate_run_manifest_antigravity",
    "verify_official_antigravity",
    "write_json_atomic_antigravity",
]
