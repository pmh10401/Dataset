from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


EXPECTED_POLICY_VERSION = "exp265-shadow-v1"


@dataclass(frozen=True)
class ReportPaths:
    json: Path
    markdown: Path
    manifest: Path
    payload: Mapping[str, object]


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(child) for child in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def report_sha256(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("report_sha256", None)
    return hashlib.sha256(_json_bytes(unsigned)).hexdigest()


def _validate(payload: Mapping[str, object], expected_policy_version: str) -> None:
    if payload.get("collection_mode") != "prospective_shadow" or payload.get("prospective") is not True or payload.get("retrospective") is not False:
        raise ValueError("report is not a prospective shadow collection")
    if payload.get("policy_version") != expected_policy_version:
        raise ValueError("unexpected policy version")
    if payload.get("automatic_policy_changed") is not False:
        raise ValueError("report claims an automatic policy change")
    if payload.get("deployment_validated") is not False:
        raise ValueError("report must not claim deployment validation")
    if payload.get("autonomous_scope") != "Hard_only":
        raise ValueError("autonomous scope mixes Hard with review lanes")
    recall, f1 = payload.get("recall"), payload.get("f1")
    if (recall is not None or f1 is not None) and (
        payload.get("metric_scope") != "fully_annotated_human_assisted"
        or not isinstance(payload.get("fully_annotated_runs"), int)
        or int(payload["fully_annotated_runs"]) < 1
    ):
        raise ValueError("Recall/F1 require fully annotated runs")
    stated_hash = payload.get("report_sha256")
    if not isinstance(stated_hash, str) or stated_hash != report_sha256(payload):
        raise ValueError("report SHA256 mismatch")
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    if "historical_test" in serialized or "historical test" in serialized:
        raise ValueError("historical TEST rows cannot be presented as prospective")


def _percent(value: object) -> str:
    return "계산 불가" if value is None else f"{float(value) * 100:.1f}%"


def _markdown(payload: Mapping[str, object]) -> str:
    limitations = [
        "Shadow/Priority/Standard는 사용자 검토 범위이며 자동 경보가 아닙니다.",
        "동일 평가셋 재분석이 아니라 신규 prospective run만 집계합니다.",
        "100 run 수집은 정책 자동 승격이나 실제 설비 성능 보장을 의미하지 않습니다.",
    ]
    return "\n".join([
        "# Exp265 신규 run Shadow 검증 보고서",
        "",
        f"- 수집 진행: {payload.get('runs_collected', 0)} / {payload.get('target_runs', 100)} run",
        f"- 후보 판정 Precision: {_percent(payload.get('candidate_precision'))}",
        f"- Recall: {_percent(payload.get('recall'))}",
        f"- F1: {_percent(payload.get('f1'))}",
        f"- 지표 범위: {payload.get('metric_scope', '확인 불가')}",
        f"- 수집 기간: {payload.get('collected_from') or '없음'} ~ {payload.get('collected_to') or '없음'}",
        "",
        "## 해석 제한",
        *(f"- {item}" for item in limitations),
        "",
    ])


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(
    output_dir: Path,
    payload: Mapping[str, object],
    *,
    api_url: str,
    expected_policy_version: str = EXPECTED_POLICY_VERSION,
) -> ReportPaths:
    _validate(payload, expected_policy_version)
    output_dir = output_dir.expanduser().resolve()
    json_path = output_dir / "shadow_report.json"
    markdown_path = output_dir / "SHADOW_REPORT.md"
    manifest_path = output_dir / "MANIFEST.json"
    _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    _atomic_write(markdown_path, _markdown(payload).encode("utf-8"))
    code_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "response_sha256": payload["report_sha256"],
        "code_sha256": _file_hash(code_path),
        "policy_version": payload["policy_version"],
        "config_sha256": payload.get("config_sha256", []),
        "collected_from": payload.get("collected_from"),
        "collected_to": payload.get("collected_to"),
        "retrospective_or_prospective": "prospective",
        "autonomous_scope": "Hard_only",
        "human_assisted_scope": "Shadow/Priority/Standard and fully annotated Recall/F1",
        "known_limitations": [
            "Not deployment validated",
            "No automatic policy promotion",
            "Recall/F1 only when the report marks fully annotated human-assisted scope",
        ],
        "files": {
            json_path.name: _file_hash(json_path),
            markdown_path.name: _file_hash(markdown_path),
        },
    }
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return ReportPaths(json_path, markdown_path, manifest_path, dict(payload))
