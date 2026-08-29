"""Canonical, synthetic-only Exp470 evidence publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp470FoldResultAntigravity, Exp470QualificationAntigravity
from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_repair_gates_antigravity
from .qualification_antigravity import qualify_records_antigravity
from .scoring_antigravity import FoldMethodScoresAntigravity


LEAF_NAMES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)
_HASHED_LEAF_NAMES_ANTIGRAVITY = LEAF_NAMES_ANTIGRAVITY[1:]
_SCOPES_ANTIGRAVITY = ("normal",) + FROZEN_CONFIG_ANTIGRAVITY.change_types


def _canonical_json_bytes_antigravity(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _array_payload_antigravity(values: np.ndarray) -> list[Any]:
    return values.tolist()


def _json_seed_mapping_antigravity(values: Mapping[int, int]) -> dict[str, int]:
    return {str(seed): count for seed, count in values.items()}


def _fold_result_payload_antigravity(row: Exp470FoldResultAntigravity) -> dict[str, Any]:
    return {
        "c0_per_type_auc": dict(row.c0_per_type_auc),
        "c11_per_type_auc": dict(row.c11_per_type_auc),
        "failure_reasons": list(row.failure_reasons),
        "fallback_exact": row.fallback_exact,
        "finite": row.finite,
        "fold_index": row.fold_index,
        "normal_confirmations": row.normal_confirmations,
        "pair_repairs": {name: {"broken": value.broken, "fixed": value.fixed} for name, value in row.pair_repairs.items()},
        "passed": row.passed,
        "seed": row.seed,
    }


def _metric_row_antigravity(record: FoldMethodScoresAntigravity, result: Exp470FoldResultAntigravity) -> dict[str, Any]:
    return {
        "backend": record.backend,
        "confirmed": {scope: _array_payload_antigravity(record.confirmed[scope]) for scope in _SCOPES_ANTIGRAVITY},
        "fit_normal_max_raw_entropy": record.fit_normal_max_raw_entropy,
        "fit_raw_entropy": _array_payload_antigravity(record.fit_raw_entropy),
        "fold_result": _fold_result_payload_antigravity(result),
        "fold_index": record.fold_index,
        "original": {method: _array_payload_antigravity(record.original[method]) for method in record.methods},
        "primary": {method: {change: _array_payload_antigravity(record.primary[method][change]) for change in FROZEN_CONFIG_ANTIGRAVITY.change_types} for method in record.methods},
        "query_raw_entropy": {scope: _array_payload_antigravity(record.query_raw_entropy[scope]) for scope in _SCOPES_ANTIGRAVITY},
        "seed": record.seed,
    }


def _context_record_antigravity(record: FoldMethodScoresAntigravity) -> dict[str, Any]:
    return {
        "c0_normal": _array_payload_antigravity(record.original["C0"]),
        "c0_primary": {change: _array_payload_antigravity(record.primary["C0"][change]) for change in FROZEN_CONFIG_ANTIGRAVITY.change_types},
        "fit_indices": _array_payload_antigravity(record.input_fit_indices),
        "fold_index": record.fold_index,
        "query_indices": _array_payload_antigravity(record.input_query_indices),
        "seed": record.seed,
    }


def _context_payload_antigravity(records: tuple[FoldMethodScoresAntigravity, ...]) -> dict[str, Any]:
    return {
        "records": [_context_record_antigravity(record) for record in records],
        "schema": "exp470_context_v1",
        "shared_train140": _array_payload_antigravity(records[0].input_train140),
    }


def _decision_payload_antigravity(
    qualification: Exp470QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, Any]:
    return {
        "eligible_for_private_reproducibility": decision.eligible_for_private_reproducibility,
        "failure_reasons": list(decision.failure_reasons),
        "fallback_exact": qualification.fallback_exact,
        "finite": qualification.finite,
        "label_query_count": qualification.label_query_count,
        "normal_confirmations": _json_seed_mapping_antigravity(qualification.normal_confirmations),
        "passed": decision.passed,
        "period_broken": _json_seed_mapping_antigravity(qualification.period_broken),
        "period_fixed": _json_seed_mapping_antigravity(qualification.period_fixed),
        "protected_broken": dict(qualification.protected_broken),
        "schema": "exp470_decision_v1",
    }


def _method_summary_rows_antigravity(metrics: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for method in FROZEN_CONFIG_ANTIGRAVITY.method_order:
        auc_key = "c0_per_type_auc" if method == "C0" else "c11_per_type_auc"
        if method == "E11":
            # E11 is retained as rank evidence; pair-gate AUC is intentionally C0/C11-only.
            rows.append({"fold_count": len(metrics), "method": method, "schema": "exp470_method_summary_v1"})
            continue
        rows.append({
            "fold_count": len(metrics),
            "mean_auc_by_change": {
                change: sum(float(row["fold_result"][auc_key][change]) for row in metrics) / len(metrics)
                for change in FROZEN_CONFIG_ANTIGRAVITY.change_types
            },
            "method": method,
            "schema": "exp470_method_summary_v1",
        })
    return tuple(rows)


def _dashboard_snapshot_payload_antigravity(
    qualification: Exp470QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
    summary_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    summary = {row["method"]: row for row in summary_rows}
    return {
        "c0_mean_auc_by_change": summary["C0"]["mean_auc_by_change"],
        "c11_mean_auc_by_change": summary["C11"]["mean_auc_by_change"],
        "eligible_for_private_reproducibility": decision.eligible_for_private_reproducibility,
        "fallback_exact": qualification.fallback_exact,
        "finite": qualification.finite,
        "mlx_status": "assessed_not_selected",
        "normal_confirmations": _json_seed_mapping_antigravity(qualification.normal_confirmations),
        "numba_backend": "numba",
        "passed": decision.passed,
        "period_broken": _json_seed_mapping_antigravity(qualification.period_broken),
        "period_fixed": _json_seed_mapping_antigravity(qualification.period_fixed),
        "protected_broken": dict(qualification.protected_broken),
        "schema": "exp470_dashboard_snapshot_v1",
        "status": "awaiting_private_checkpoint",
    }


def _failed_rows_antigravity(qualification: Exp470QualificationAntigravity) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"failure_reasons": list(row.failure_reasons), "fold_index": row.fold_index, "seed": row.seed}
        for row in qualification.fold_results if not row.passed
    )


def _validate_publication_inputs_antigravity(
    records: object, qualification: object, decision: object,
) -> tuple[tuple[FoldMethodScoresAntigravity, ...], Exp470QualificationAntigravity, QualificationDecisionAntigravity]:
    if type(qualification) is not Exp470QualificationAntigravity:
        raise ValueError("qualification must use the exact Task 4 type")
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("decision must use the exact Task 4 type")
    recomputed = qualify_records_antigravity(records)
    recomputed_decision = evaluate_repair_gates_antigravity(recomputed)
    if qualification != recomputed:
        raise ValueError("caller qualification disagrees with recomputed records")
    if decision != recomputed_decision:
        raise ValueError("caller decision disagrees with recomputed qualification")
    return tuple(records.values()), recomputed, recomputed_decision


def _require_empty_root_antigravity(root: Path) -> None:
    if root.exists():
        if not root.is_dir():
            raise ValueError("publication root must be a directory path")
        if any(root.iterdir()):
            raise ValueError("publication root must be empty; unrelated files are preserved")
    else:
        root.mkdir(parents=True)


@dataclass(frozen=True, slots=True)
class PublishedResultAntigravity:
    root: Path
    hashes: Mapping[str, str]
    qualification: Exp470QualificationAntigravity
    decision: QualificationDecisionAntigravity

    def __post_init__(self) -> None:
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))


def publish_result_antigravity(root: object, records: object, qualification: object, decision: object) -> PublishedResultAntigravity:
    """Write the exact seven-leaf publication only after independently recomputing its gates."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    values, recomputed, recomputed_decision = _validate_publication_inputs_antigravity(records, qualification, decision)
    _require_empty_root_antigravity(root)
    metrics = tuple(_metric_row_antigravity(record, result) for record, result in zip(values, recomputed.fold_results, strict=True))
    summaries = _method_summary_rows_antigravity(metrics)
    payloads = {
        "context_antigravity.json": _canonical_json_bytes_antigravity(_context_payload_antigravity(values)),
        "dashboard_snapshot_antigravity.json": _canonical_json_bytes_antigravity(_dashboard_snapshot_payload_antigravity(recomputed, recomputed_decision, summaries)),
        "decision_antigravity.json": _canonical_json_bytes_antigravity(_decision_payload_antigravity(recomputed, recomputed_decision)),
        "failed_folds_antigravity.jsonl": b"".join(_canonical_json_bytes_antigravity(row) for row in _failed_rows_antigravity(recomputed)),
        "method_fold_metrics_antigravity.jsonl": b"".join(_canonical_json_bytes_antigravity(row) for row in metrics),
        "method_summary_antigravity.jsonl": b"".join(_canonical_json_bytes_antigravity(row) for row in summaries),
    }
    for name in _HASHED_LEAF_NAMES_ANTIGRAVITY:
        (root / name).write_bytes(payloads[name])
    hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in _HASHED_LEAF_NAMES_ANTIGRAVITY}
    (root / LEAF_NAMES_ANTIGRAVITY[0]).write_bytes(_canonical_json_bytes_antigravity({"hashes": hashes, "schema": "exp470_artifact_sha256_v1"}))
    return PublishedResultAntigravity(root=root, hashes=hashes, qualification=recomputed, decision=recomputed_decision)


__all__ = (
    "LEAF_NAMES_ANTIGRAVITY", "PublishedResultAntigravity", "_canonical_json_bytes_antigravity",
    "_dashboard_snapshot_payload_antigravity", "_decision_payload_antigravity", "_fold_result_payload_antigravity",
    "_metric_row_antigravity", "_method_summary_rows_antigravity", "publish_result_antigravity",
)
