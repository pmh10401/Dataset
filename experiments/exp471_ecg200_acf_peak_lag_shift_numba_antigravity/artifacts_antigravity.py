"""Canonical synthetic-only publication for Exp471 ACF peak-lag evidence."""

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
from .contracts_antigravity import Exp471FoldResultAntigravity, Exp471QualificationAntigravity
from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_repair_gates_antigravity
from .qualification_antigravity import blocked_fold_indices_antigravity, qualify_records_antigravity
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


def evaluate_qualification_antigravity(qualification: Exp471QualificationAntigravity) -> QualificationDecisionAntigravity:
    """Apply the keyword-only Exp471 decision API from an aggregate result."""
    return evaluate_repair_gates_antigravity(
        period_fixed=qualification.period_fixed,
        period_broken=qualification.period_broken,
        protected_broken=qualification.protected_broken,
        normal_confirmations=qualification.normal_confirmations,
        fallback_exact=qualification.fallback_exact,
        finite=qualification.finite,
    )


def _canonical_json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _array_payload_antigravity(values: np.ndarray) -> list[Any]:
    return values.tolist()


def _scope_mapping_payload_antigravity(values: Mapping[str, np.ndarray]) -> dict[str, list[Any]]:
    return {scope: _array_payload_antigravity(values[scope]) for scope in _SCOPES_ANTIGRAVITY}


def _json_seed_mapping_antigravity(values: Mapping[int, int]) -> dict[str, int]:
    return {str(seed): count for seed, count in values.items()}


def _fold_result_payload_antigravity(row: Exp471FoldResultAntigravity) -> dict[str, Any]:
    return {
        "c0_per_type_auc": dict(row.c0_per_type_auc),
        "c12_per_type_auc": dict(row.c12_per_type_auc),
        "failure_reasons": list(row.failure_reasons),
        "fallback_exact": row.fallback_exact,
        "finite": row.finite,
        "fold_index": row.fold_index,
        "normal_confirmations": row.normal_confirmations,
        "pair_repairs": {
            name: {"broken": value.broken, "fixed": value.fixed}
            for name, value in row.pair_repairs.items()
        },
        "passed": row.passed,
        "seed": row.seed,
    }


def _metric_row_antigravity(
    record: FoldMethodScoresAntigravity,
    result: Exp471FoldResultAntigravity,
) -> dict[str, Any]:
    return {
        "backend": record.backend,
        "confirmed": _scope_mapping_payload_antigravity(record.confirmed),
        "fit_normal_max_raw_shift": record.fit_normal_max_raw_shift,
        "fit_raw_shift": _array_payload_antigravity(record.fit_raw_shift),
        "fold_index": record.fold_index,
        "fold_result": _fold_result_payload_antigravity(result),
        "original": {
            method: _array_payload_antigravity(record.original[method])
            for method in record.methods
        },
        "primary": {
            method: {
                change: _array_payload_antigravity(record.primary[method][change])
                for change in FROZEN_CONFIG_ANTIGRAVITY.change_types
            }
            for method in record.methods
        },
        "query_raw_shift": _scope_mapping_payload_antigravity(record.query_raw_shift),
        "seed": record.seed,
    }


def _context_record_antigravity(record: FoldMethodScoresAntigravity) -> dict[str, Any]:
    return {
        "c0_normal": _array_payload_antigravity(record.original["C0"]),
        "c0_primary": {
            change: _array_payload_antigravity(record.primary["C0"][change])
            for change in FROZEN_CONFIG_ANTIGRAVITY.change_types
        },
        "confirmed": _scope_mapping_payload_antigravity(record.confirmed),
        "query_ranks": {
            "normal": _array_payload_antigravity(record.original["E12"]),
            **{
                change: _array_payload_antigravity(record.primary["E12"][change])
                for change in FROZEN_CONFIG_ANTIGRAVITY.change_types
            },
        },
        "fit_early_peak_lags": _array_payload_antigravity(record.fit_early_peak_lags),
        "fit_indices": _array_payload_antigravity(record.input_fit_indices),
        "fit_late_peak_lags": _array_payload_antigravity(record.fit_late_peak_lags),
        "fit_normal_max_raw_shift": record.fit_normal_max_raw_shift,
        "fit_peak_presence": _array_payload_antigravity(record.fit_peak_presence),
        "fit_raw_shift": _array_payload_antigravity(record.fit_raw_shift),
        "fit_shifts": _array_payload_antigravity(record.fit_shifts),
        "fold_index": record.fold_index,
        "query_early_peak_lags": _scope_mapping_payload_antigravity(record.query_early_peak_lags),
        "query_indices": _array_payload_antigravity(record.input_query_indices),
        "query_late_peak_lags": _scope_mapping_payload_antigravity(record.query_late_peak_lags),
        "query_peak_presence": _scope_mapping_payload_antigravity(record.query_peak_presence),
        "query_raw_shift": _scope_mapping_payload_antigravity(record.query_raw_shift),
        "query_shifts": _scope_mapping_payload_antigravity(record.query_shifts),
        "seed": record.seed,
    }


def _context_payload_antigravity(records: tuple[FoldMethodScoresAntigravity, ...]) -> dict[str, Any]:
    return {
        "records": [_context_record_antigravity(record) for record in records],
        "schema": "exp471_context_v1",
        "shared_train140": _array_payload_antigravity(records[0].input_train140),
    }


def _decision_payload_antigravity(
    qualification: Exp471QualificationAntigravity,
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
        "schema": "exp471_decision_v1",
    }


def _method_summary_rows_antigravity(metrics: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for method in FROZEN_CONFIG_ANTIGRAVITY.method_order:
        if method == "E12":
            # E12 remains rank evidence; pair-gate AUC is intentionally C0/C12-only.
            rows.append({"fold_count": len(metrics), "method": method, "schema": "exp471_method_summary_v1"})
            continue
        auc_key = "c0_per_type_auc" if method == "C0" else "c12_per_type_auc"
        rows.append({
            "fold_count": len(metrics),
            "mean_auc_by_change": {
                change: sum(float(row["fold_result"][auc_key][change]) for row in metrics) / len(metrics)
                for change in FROZEN_CONFIG_ANTIGRAVITY.change_types
            },
            "method": method,
            "schema": "exp471_method_summary_v1",
        })
    return tuple(rows)


def _peak_absence_counts_antigravity(records: tuple[FoldMethodScoresAntigravity, ...]) -> dict[str, Any]:
    fit_missing = sum(int(np.logical_not(record.fit_peak_presence).any(axis=1).sum()) for record in records)
    query_missing = sum(
        int(np.logical_not(record.query_peak_presence[scope]).any(axis=1).sum())
        for record in records
        for scope in _SCOPES_ANTIGRAVITY
    )
    return {"fit_rows_with_any_peak_missing": fit_missing, "query_scope_rows_with_any_peak_missing": query_missing}


def _shift_distributions_antigravity(records: tuple[FoldMethodScoresAntigravity, ...]) -> dict[str, Any]:
    fit_values = [int(value) for record in records for value in record.fit_shifts]
    query_values = [
        int(value)
        for record in records
        for scope in _SCOPES_ANTIGRAVITY
        for value in record.query_shifts[scope]
    ]
    return {
        "fit": {str(value): fit_values.count(value) for value in sorted(set(fit_values))},
        "query_all_scopes": {str(value): query_values.count(value) for value in sorted(set(query_values))},
    }


def _dashboard_snapshot_payload_antigravity(
    records: tuple[FoldMethodScoresAntigravity, ...],
    qualification: Exp471QualificationAntigravity,
    decision: QualificationDecisionAntigravity,
    summaries: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    summary = {row["method"]: row for row in summaries}
    return {
        "backend": "numba",
        "c0_mean_auc_by_change": summary["C0"]["mean_auc_by_change"],
        "c12_mean_auc_by_change": summary["C12"]["mean_auc_by_change"],
        "eligible_for_private_reproducibility": decision.eligible_for_private_reproducibility,
        "fallback_exact": qualification.fallback_exact,
        "finite": qualification.finite,
        "mlx_status": "assessed_not_selected",
        "normal_confirmations": _json_seed_mapping_antigravity(qualification.normal_confirmations),
        "passed": decision.passed,
        "peak_absence": _peak_absence_counts_antigravity(records),
        "period_broken": _json_seed_mapping_antigravity(qualification.period_broken),
        "period_fixed": _json_seed_mapping_antigravity(qualification.period_fixed),
        "protected_broken": dict(qualification.protected_broken),
        "schema": "exp471_dashboard_snapshot_v1",
        "shift_distribution": _shift_distributions_antigravity(records),
        "status": "awaiting_private_checkpoint",
    }


def _failed_rows_antigravity(qualification: Exp471QualificationAntigravity) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "failure_reasons": list(row.failure_reasons),
            "fold_index": row.fold_index,
            "seed": row.seed,
        }
        for row in qualification.fold_results
        if not row.passed
    )


def _validate_publication_inputs_antigravity(
    records: object,
    qualification: object,
    decision: object,
) -> tuple[tuple[FoldMethodScoresAntigravity, ...], Exp471QualificationAntigravity, QualificationDecisionAntigravity]:
    if type(qualification) is not Exp471QualificationAntigravity:
        raise ValueError("qualification must use the exact Task 4 type")
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("decision must use the exact Task 4 type")
    if not isinstance(records, Mapping):
        raise ValueError("records must be an identity mapping")
    recomputed = qualify_records_antigravity(records)
    recomputed_decision = evaluate_qualification_antigravity(recomputed)
    if qualification != recomputed:
        raise ValueError("caller qualification disagrees with recomputed records")
    if decision != recomputed_decision:
        raise ValueError("caller decision disagrees with recomputed qualification")
    identities = tuple((seed, fold) for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds))
    if tuple(records) != identities:
        raise ValueError("record identities differ from frozen seed/fold order")
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
    qualification: Exp471QualificationAntigravity
    decision: QualificationDecisionAntigravity

    def __post_init__(self) -> None:
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))


def publish_result_antigravity(
    root: object,
    records: object,
    qualification: object,
    decision: object,
) -> PublishedResultAntigravity:
    """Write exactly seven canonical leaves only after independently recomputing gates."""
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    values, recomputed, recomputed_decision = _validate_publication_inputs_antigravity(
        records, qualification, decision
    )
    _require_empty_root_antigravity(root)
    folds = blocked_fold_indices_antigravity()
    for record in values:
        fit, query = folds[record.fold_index]
        if not np.array_equal(record.input_fit_indices, fit) or not np.array_equal(record.input_query_indices, query):
            raise ValueError("record indices differ from frozen blocked folds")
    metrics = tuple(
        _metric_row_antigravity(record, result)
        for record, result in zip(values, recomputed.fold_results, strict=True)
    )
    summaries = _method_summary_rows_antigravity(metrics)
    payloads = {
        "context_antigravity.json": _canonical_json_bytes_antigravity(_context_payload_antigravity(values)),
        "dashboard_snapshot_antigravity.json": _canonical_json_bytes_antigravity(
            _dashboard_snapshot_payload_antigravity(values, recomputed, recomputed_decision, summaries)
        ),
        "decision_antigravity.json": _canonical_json_bytes_antigravity(_decision_payload_antigravity(recomputed, recomputed_decision)),
        "failed_folds_antigravity.jsonl": b"".join(_canonical_json_bytes_antigravity(row) for row in _failed_rows_antigravity(recomputed)),
        "method_fold_metrics_antigravity.jsonl": b"".join(_canonical_json_bytes_antigravity(row) for row in metrics),
        "method_summary_antigravity.jsonl": b"".join(_canonical_json_bytes_antigravity(row) for row in summaries),
    }
    for name in _HASHED_LEAF_NAMES_ANTIGRAVITY:
        (root / name).write_bytes(payloads[name])
    hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in _HASHED_LEAF_NAMES_ANTIGRAVITY}
    (root / LEAF_NAMES_ANTIGRAVITY[0]).write_bytes(
        _canonical_json_bytes_antigravity({"hashes": hashes, "schema": "exp471_artifact_sha256_v1"})
    )
    return PublishedResultAntigravity(root=root, hashes=hashes, qualification=recomputed, decision=recomputed_decision)


__all__ = (
    "evaluate_qualification_antigravity",
    "LEAF_NAMES_ANTIGRAVITY",
    "PublishedResultAntigravity",
    "_canonical_json_bytes_antigravity",
    "_context_record_antigravity",
    "_dashboard_snapshot_payload_antigravity",
    "_decision_payload_antigravity",
    "_failed_rows_antigravity",
    "_fold_result_payload_antigravity",
    "_metric_row_antigravity",
    "_method_summary_rows_antigravity",
    "publish_result_antigravity",
)
