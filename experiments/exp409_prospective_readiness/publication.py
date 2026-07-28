"""Fail-closed, atomic publication for Exp409 readiness artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from collections.abc import Mapping
import uuid

from experiments.exp403_408.artifact_io import (
    canonical_json_bytes,
    sha256_file,
)

from .config import FROZEN_CONFIG
from .contracts import ReadinessBatch, validate_batch
from .guard import validate_label_blind_payload
from .provenance import (
    provenance_closure_sha256,
    validate_canonical_provenance,
)


_CONFIG_FILE = "exp409_config.json"
_CANDIDATE_FILE = "exp409_shadow_candidates.jsonl"
_STATUS_FILE = "exp409_source_statuses.jsonl"
_PROVENANCE_FILE = "exp409_provenance_manifest.json"
_SUMMARY_FILE = "exp409_readiness_summary.json"
_REPORT_FILE = "exp409_readiness_report_ko.md"
_RUNTIME_FILE = "exp409_runtime_envelope.json"
_MANIFEST_FILE = "exp409_artifact_manifest.json"


def candidate_payload_sha256(batch: ReadinessBatch) -> str:
    """Hash the exact canonical candidate JSONL bytes used by publication."""
    rows = _candidate_rows(batch)
    validate_label_blind_payload(rows, "exp409_candidates")
    return hashlib.sha256(_jsonl_bytes(rows)).hexdigest()


def publish_batch(
    batch: ReadinessBatch,
    output_dir: Path,
    canonical_provenance: Mapping[str, object],
    runtime_envelope: Mapping[str, object],
) -> Mapping[str, str]:
    """Publish a complete validated directory or leave no directory at all."""
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp409 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp409 output directory already exists")

    validate_batch(batch, review_budget=FROZEN_CONFIG.review_budget)
    validate_canonical_provenance(canonical_provenance)
    if canonical_provenance["candidate_sha256"] != candidate_payload_sha256(
        batch
    ):
        raise ValueError("Exp409 candidate provenance digest mismatch")
    _validate_runtime_envelope(runtime_envelope)
    payloads = _render_payloads(
        batch, canonical_provenance, runtime_envelope
    )

    staging = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.staging"
    )
    try:
        staging.mkdir(parents=False, exist_ok=False)
        artifact_hashes = _write_staged_files(staging, payloads)
        manifest = {
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "provenance_closure_sha256": provenance_closure_sha256(
                canonical_provenance
            ),
            "artifact_sha256": dict(sorted(artifact_hashes.items())),
        }
        manifest_payload = canonical_json_bytes(manifest) + b"\n"
        (staging / _MANIFEST_FILE).write_bytes(manifest_payload)
        artifact_hashes[_MANIFEST_FILE] = sha256_file(
            staging / _MANIFEST_FILE
        )
        _verify_staged_files(staging, artifact_hashes)
        if output.exists():
            raise FileExistsError("Exp409 output directory appeared during staging")
        os.replace(staging, output)
        return dict(sorted(artifact_hashes.items()))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_failure_record(
    output_dir: Path, *, status: str, reason: str
) -> Path:
    """Atomically write a candidate-free failure record beside the output."""
    if status not in {"blocked", "error"}:
        raise ValueError("Exp409 failure status must be blocked or error")
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp409 failure-record parent does not exist")
    destination = output.with_name(f"{output.name}.failure.json")
    if destination.exists():
        raise FileExistsError("Exp409 failure record already exists")
    safe_reason = " ".join(str(reason).split())[:500]
    if not safe_reason:
        raise ValueError("Exp409 failure reason is required")
    payload = canonical_json_bytes(
        {
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "status": status,
            "reason": safe_reason,
        }
    ) + b"\n"
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError("Exp409 failure record appeared during staging")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _render_payloads(
    batch: ReadinessBatch,
    canonical_provenance: Mapping[str, object],
    runtime_envelope: Mapping[str, object],
) -> dict[str, bytes]:
    candidates = _candidate_rows(batch)
    statuses = _status_rows(batch)
    summary = _readiness_summary(batch)
    config = FROZEN_CONFIG.to_payload()
    provenance = {
        "canonical_provenance": dict(canonical_provenance),
        "closure_sha256": provenance_closure_sha256(canonical_provenance),
    }
    for name, payload in (
        ("candidates", candidates),
        ("source_statuses", statuses),
    ):
        validate_label_blind_payload(payload, f"exp409_{name}")

    return {
        _CONFIG_FILE: canonical_json_bytes(config) + b"\n",
        _CANDIDATE_FILE: _jsonl_bytes(candidates),
        _STATUS_FILE: _jsonl_bytes(statuses),
        _PROVENANCE_FILE: canonical_json_bytes(provenance) + b"\n",
        _SUMMARY_FILE: canonical_json_bytes(summary) + b"\n",
        _REPORT_FILE: _readiness_report(summary).encode("utf-8"),
        _RUNTIME_FILE: canonical_json_bytes(runtime_envelope) + b"\n",
    }


def _candidate_rows(batch: ReadinessBatch) -> list[dict[str, object]]:
    return [
        candidate.to_payload()
        for candidate in sorted(
            batch.candidates,
            key=lambda row: (row.dataset, row.candidate_index),
        )
    ]


def _status_rows(batch: ReadinessBatch) -> list[dict[str, object]]:
    rows = [
        {
            "dataset": result.dataset,
            **asdict(status),
            "output_sha256": result.source_output_sha256[status.source],
        }
        for result in batch.dataset_results
        for status in result.source_statuses
    ]
    return sorted(rows, key=lambda row: (str(row["dataset"]), str(row["source"])))


def _readiness_summary(batch: ReadinessBatch) -> dict[str, object]:
    available = sum(
        status.status == "available"
        for result in batch.dataset_results
        for status in result.source_statuses
    )
    abstained = sum(
        status.status == "abstained"
        for result in batch.dataset_results
        for status in result.source_statuses
    )
    return {
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "variant": FROZEN_CONFIG.variant,
        "alpha": FROZEN_CONFIG.alpha,
        "review_budget": FROZEN_CONFIG.review_budget,
        "disposition": FROZEN_CONFIG.disposition,
        "dataset_count": len(batch.dataset_results),
        "candidate_count": len(batch.candidates),
        "available_source_count": available,
        "abstained_source_count": abstained,
        "performance_evaluation": FROZEN_CONFIG.performance_evaluation,
        "prospective_validated": FROZEN_CONFIG.prospective_validated,
        "deployment_eligible": FROZEN_CONFIG.deployment_eligible,
        "operational_policy_changed": FROZEN_CONFIG.operational_policy_changed,
        "automatic_lanes_changed": False,
    }


def _readiness_report(summary: Mapping[str, object]) -> str:
    return "\n".join(
        (
            "# Exp409 전향적 준비 상태 보고서",
            "",
            f"- 데이터셋 수: `{summary['dataset_count']}`",
            f"- review-only 후보 수: `{summary['candidate_count']}`",
            f"- 공개 예산: `{summary['review_budget']}`",
            f"- 변형 / alpha: `{summary['variant']}` / `{summary['alpha']}`",
            "- 성능 평가: `not_evaluated`",
            "- 전향 검증: `false`",
            "- 배포 적격: `false`",
            "- 운영 정책 변경: `false`",
            "- Hard / Standard / Priority 자동 lane 변경: `false`",
            "",
            "이 산출물은 새 데이터 도착 전 재현성과 실행 준비 상태만 확인한다.",
            "모든 후보는 사람 검토 전용이며 자동 경보로 승격되지 않는다.",
            "",
        )
    )


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _write_staged_files(
    staging: Path, payloads: Mapping[str, bytes]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, payload in sorted(payloads.items()):
        path = staging / name
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return hashes


def _verify_staged_files(
    staging: Path, expected: Mapping[str, str]
) -> None:
    if {path.name for path in staging.iterdir()} != set(expected):
        raise ValueError("Exp409 staged artifact coverage mismatch")
    for name, digest in expected.items():
        if sha256_file(staging / name) != digest:
            raise ValueError(f"Exp409 staged artifact hash mismatch: {name}")


def _validate_runtime_envelope(value: Mapping[str, object]) -> None:
    required = {
        "run_id",
        "started_at",
        "python_version",
        "platform",
        "packages",
        "runtime_envelope_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Exp409 runtime envelope is invalid")
    content = {key: value[key] for key in required if key != "runtime_envelope_sha256"}
    observed = hashlib.sha256(canonical_json_bytes(content)).hexdigest()
    if value["runtime_envelope_sha256"] != observed:
        raise ValueError("Exp409 runtime envelope hash mismatch")
