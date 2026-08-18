"""Create the exact replayable nine-leaf Exp458 Task 4 result artifact."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    PublicationEvidenceAntigravity,
    PublishedResultAntigravity,
    observe_runtime_provenance_antigravity,
)
from .dashboard_export_antigravity import export_dashboard_antigravity
from .evaluation_antigravity import (
    _evidence_for_result_antigravity,
    evaluate_routes_antigravity,
    validate_evaluation_result_semantics_antigravity,
)
from .router_antigravity import route_normal2_antigravity


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "train_qualification_antigravity.jsonl",
    "features_antigravity.jsonl",
    "route_decisions_antigravity.jsonl",
    "comparison_metrics_antigravity.json",
    "family_metrics_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_EXP456_RESULT_ROOT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/exp456_exp455_dataset_key_order_repair_antigravity_20260817"
)


def _json_bytes_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _jsonl_bytes_antigravity(rows: tuple[object, ...]) -> bytes:
    return b"".join(_json_bytes_antigravity(row) for row in rows)


def _confusion_row_antigravity(value: object) -> dict[str, int]:
    return {"tp": value.tp, "fp": value.fp, "fn": value.fn, "tn": value.tn}


def _qualification_row_antigravity(value: object) -> dict[str, object]:
    return {
        "dataset": value.dataset,
        "seed_evidence": [
            {
                "dataset": seed.dataset,
                "seed": seed.seed,
                "normal_trials": seed.normal_trials,
                "normal_alerts": seed.normal_alerts,
                "normal_wilson_upper": seed.normal_wilson_upper,
                "per_type_auc": dict(seed.per_type_auc),
                "pooled_auc": seed.pooled_auc,
                "passed": seed.passed,
            }
            for seed in value.seed_evidence
        ],
        "passed": value.passed,
    }


def _comparison_row_antigravity(value: object) -> dict[str, object]:
    return {
        "name": value.name,
        "confusion": _confusion_row_antigravity(value.confusion),
        "f1": value.f1,
    }


def _context_antigravity(evidence: PublicationEvidenceAntigravity) -> dict[str, object]:
    authority = evidence.exp456_authority
    rows = [
        {
            "dataset": row.dataset,
            "family": row.family,
            "target_length": row.target_length,
            "method": row.method,
            "baseline_confusion": _confusion_row_antigravity(row.baseline_confusion),
            "candidate_confusion": None if row.candidate_confusion is None else _confusion_row_antigravity(row.candidate_confusion),
        }
        for row in authority.authority.rows
    ]
    return {
        "experiment_id": "Exp458",
        "approved_design_sha256": FROZEN_CONFIG_ANTIGRAVITY.approved_design_sha256,
        "exp456": {
            "result_root": _EXP456_RESULT_ROOT_ANTIGRAVITY,
            "artifact_map_sha256": authority.artifact_map_sha256,
            "comparison_sha256": authority.comparison_sha256,
            "decision_sha256": authority.decision_sha256,
        },
        "rows": rows,
        "runtime": asdict(observe_runtime_provenance_antigravity()),
    }


def _validate_evidence_antigravity(value: object) -> PublicationEvidenceAntigravity:
    if type(value) is not PublicationEvidenceAntigravity:
        raise ValueError("publication evidence type is invalid")
    validate_evaluation_result_semantics_antigravity(value.result)
    authority, qualification, feature = _evidence_for_result_antigravity(value.result)
    if authority != value.exp456_authority or qualification != value.qualification or feature != value.feature:
        raise ValueError("publication evidence differs from evaluated provenance")
    decision = route_normal2_antigravity(value.feature, value.qualification)
    if evaluate_routes_antigravity(value.exp456_authority, decision) != value.result:
        raise ValueError("publication result differs from recomputed evaluation")
    export_dashboard_antigravity(value.result)
    return value


def _payloads_antigravity(evidence: PublicationEvidenceAntigravity) -> dict[str, bytes]:
    value = _validate_evidence_antigravity(evidence)
    result = value.result
    return {
        "context_antigravity.json": _json_bytes_antigravity(_context_antigravity(value)),
        "train_qualification_antigravity.jsonl": _jsonl_bytes_antigravity((_qualification_row_antigravity(value.qualification),)),
        "features_antigravity.jsonl": _jsonl_bytes_antigravity((asdict(value.feature),)),
        "route_decisions_antigravity.jsonl": _jsonl_bytes_antigravity(tuple(asdict(item) for item in result.decisions)),
        "comparison_metrics_antigravity.json": _json_bytes_antigravity({"comparisons": [_comparison_row_antigravity(item) for item in result.comparisons]}),
        "family_metrics_antigravity.json": _json_bytes_antigravity({"families": [asdict(item) for item in result.family_metrics]}),
        "dashboard_snapshot_antigravity.json": _json_bytes_antigravity(export_dashboard_antigravity(result)),
        "decision_antigravity.json": _json_bytes_antigravity(
            {
                "passed": result.passed,
                "status": result.status,
                "criteria": [asdict(item) for item in result.criteria],
            }
        ),
    }


def publish_result_antigravity(
    root: Path, evidence: PublicationEvidenceAntigravity
) -> PublishedResultAntigravity:
    if not isinstance(root, Path) or root.exists() or root.is_symlink():
        raise ValueError("publication destination must not exist")
    payloads = _payloads_antigravity(evidence)
    if tuple(payloads) != _LEAVES_ANTIGRAVITY[:-1]:
        raise ValueError("publication payload inventory differs")
    artifact_map = {name: sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    map_bytes = _json_bytes_antigravity(artifact_map)
    try:
        root.mkdir(mode=0o700)
        for name in _LEAVES_ANTIGRAVITY[:-1]:
            descriptor = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payloads[name])
        descriptor = os.open(root / _LEAVES_ANTIGRAVITY[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(map_bytes)
    except OSError as error:
        raise ValueError("publication destination could not be created") from error
    leaf_sha256 = tuple((name, sha256((root / name).read_bytes()).hexdigest()) for name in _LEAVES_ANTIGRAVITY)
    return PublishedResultAntigravity(root, leaf_sha256, sha256(map_bytes).hexdigest())


__all__ = ("publish_result_antigravity",)
