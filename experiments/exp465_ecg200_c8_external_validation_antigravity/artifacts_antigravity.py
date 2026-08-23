"""Canonical seven-leaf publication for Exp465 Task 4."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .dashboard_export_antigravity import export_dashboard_antigravity
from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_qualification_antigravity
from .qualification_antigravity import Exp465MethodFoldMetricAntigravity, Exp465MethodSummaryAntigravity, Exp465QualificationResultAntigravity, recompute_qualification_antigravity


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json", "method_fold_metrics_antigravity.jsonl", "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.jsonl", "decision_antigravity.json", "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


@dataclass(frozen=True, slots=True)
class PublishedQualificationAntigravity:
    root: Path
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


def canonical_json_bytes_antigravity(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _jsonl_bytes_antigravity(rows: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(canonical_json_bytes_antigravity(row) for row in rows)


def _row_payload_antigravity(row: Exp465MethodFoldMetricAntigravity) -> dict[str, object]:
    return {
        "method": row.method, "seed": row.seed, "fold_index": row.fold_index,
        "normal_scores": list(row.normal_scores),
        "primary_scores": {key: list(row.primary_scores[key]) for key in row.primary_scores},
        "confirmation_counts": dict(row.confirmation_counts),
        "query_cosines": {key: list(row.query_cosines[key]) for key in row.query_cosines},
        "fit_amplitudes": list(row.fit_amplitudes),
        "query_amplitudes": {key: list(row.query_amplitudes[key]) for key in row.query_amplitudes},
        "raw_evidence_kind": row.raw_evidence_kind, "raw_fit_vector_blob": row.raw_fit_vector_blob,
        "raw_query_vector_blobs": dict(row.raw_query_vector_blobs),
        "per_type_auc": list(row.per_type_auc), "mean_auc": row.mean_auc,
        "all_scores_finite": row.all_scores_finite, "normal_wilson_upper": row.normal_wilson_upper,
        "pair_diagnostics": {key: dict(row.pair_diagnostics[key]) for key in row.pair_diagnostics},
        "passed": row.passed, "failure_reasons": list(row.failure_reasons),
        "feature_dimension": row.feature_dimension, "provenance": row.provenance,
        "input_train140_blob": row.input_train140_blob,
        "input_fit_indices": list(row.input_fit_indices),
        "input_query_indices": list(row.input_query_indices),
    }


def _summary_payload_antigravity(summary: Exp465MethodSummaryAntigravity) -> dict[str, object]:
    return {
        "method": summary.method, "fold_count": summary.fold_count, "passed_count": summary.passed_count,
        "failed_count": summary.failed_count, "min_per_type_auc": list(summary.min_per_type_auc),
        "min_mean_auc": summary.min_mean_auc, "max_normal_wilson_upper": summary.max_normal_wilson_upper,
        "normal_confirmation_count": summary.normal_confirmation_count, "all_scores_finite": summary.all_scores_finite,
        "pair_diagnostics": {key: dict(summary.pair_diagnostics[key]) for key in summary.pair_diagnostics},
        "passed": summary.passed,
    }


def _decision_payload_antigravity(decision: QualificationDecisionAntigravity) -> dict[str, object]:
    return {"status": decision.status, "primary_candidate": decision.primary_candidate,
            "eligible_for_routing_research": decision.eligible_for_routing_research,
            "primary_passed_count": decision.primary_passed_count, "primary_required_count": decision.primary_required_count,
            "failure_reasons": list(decision.failure_reasons), "routing_applied": decision.routing_applied,
            "test_labels_opened": decision.test_labels_opened, "label_query_count": decision.label_query_count}


def _context_payload_antigravity(*, method_fold_metrics_sha256: str) -> dict[str, object]:
    return {"experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment, "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
            "data_split": FROZEN_CONFIG_ANTIGRAVITY.data_split, "method_order": list(FROZEN_CONFIG_ANTIGRAVITY.method_order),
            "primary_candidate": "C8", "seeds": list(FROZEN_CONFIG_ANTIGRAVITY.seeds), "folds": FROZEN_CONFIG_ANTIGRAVITY.folds,
            "gates": {"per_type_auc_min": FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min, "mean_auc_min": FROZEN_CONFIG_ANTIGRAVITY.mean_auc_min, "normal_confirmation_must_equal": 0},
            "normal_trials_per_seed": FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed,
            "input_evidence_schema": "train140_blob_plus_fit_query_indices_v1",
            "label_query_count": 0, "official_evaluate_count": 0, "method_fold_metrics_sha256": method_fold_metrics_sha256}


def _payloads_antigravity(qualification: Exp465QualificationResultAntigravity, decision: QualificationDecisionAntigravity) -> dict[str, bytes]:
    replayed = recompute_qualification_antigravity(qualification)
    if decision != evaluate_qualification_antigravity(replayed):
        raise ValueError("publication decision differs from replayed evidence")
    rows = tuple(_row_payload_antigravity(row) for row in replayed.method_rows)
    metrics = _jsonl_bytes_antigravity(rows)
    return {
        "context_antigravity.json": canonical_json_bytes_antigravity({**_context_payload_antigravity(method_fold_metrics_sha256=sha256(metrics).hexdigest()), "seed_normal_wilson_upper": {str(seed): replayed.seed_normal_wilson_upper[seed] for seed in replayed.seed_normal_wilson_upper}}),
        "method_fold_metrics_antigravity.jsonl": metrics,
        "failed_folds_antigravity.jsonl": _jsonl_bytes_antigravity(tuple(row for row in rows if not row["passed"])),
        "method_summary_antigravity.jsonl": _jsonl_bytes_antigravity(tuple(_summary_payload_antigravity(item) for item in replayed.summaries)),
        "decision_antigravity.json": canonical_json_bytes_antigravity(_decision_payload_antigravity(decision)),
        "dashboard_snapshot_antigravity.json": canonical_json_bytes_antigravity(export_dashboard_antigravity(replayed, decision)),
    }


def publish_result_antigravity(root: Path, qualification: Exp465QualificationResultAntigravity, decision: QualificationDecisionAntigravity) -> PublishedQualificationAntigravity:
    if not isinstance(root, Path) or root.exists() or root.is_symlink():
        raise ValueError("publication destination must be a new create-only path")
    payloads = _payloads_antigravity(qualification, decision)
    artifact_map = {name: sha256(payload).hexdigest() for name, payload in payloads.items()}
    try:
        root.mkdir(mode=0o700)
        for name, payload in payloads.items():
            fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
        map_bytes = canonical_json_bytes_antigravity(artifact_map)
        fd = os.open(root / _LEAVES_ANTIGRAVITY[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(map_bytes)
    except OSError as error:
        raise ValueError("publication destination could not be created") from error
    return PublishedQualificationAntigravity(root, tuple((name, sha256((root / name).read_bytes()).hexdigest()) for name in _LEAVES_ANTIGRAVITY), sha256(map_bytes).hexdigest())


__all__ = ("PublishedQualificationAntigravity", "_LEAVES_ANTIGRAVITY", "_context_payload_antigravity", "_decision_payload_antigravity", "_payloads_antigravity", "_row_payload_antigravity", "_summary_payload_antigravity", "canonical_json_bytes_antigravity", "publish_result_antigravity")
