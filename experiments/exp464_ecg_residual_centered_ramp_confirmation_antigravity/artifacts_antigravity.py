"""Canonical seven-leaf publication for Exp464 Task 4."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp463AuthorityAntigravity,
    validate_exp463_authority_antigravity,
)
from .dashboard_export_antigravity import export_dashboard_antigravity
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import (
    Exp464MethodFoldMetricAntigravity,
    Exp464MethodSummaryAntigravity,
    Exp464QualificationContractAntigravity,
    normalize_qualification_antigravity,
)


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "method_fold_metrics_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


@dataclass(frozen=True, slots=True)
class PublishedQualificationAntigravity:
    root: Path
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


def canonical_json_bytes_antigravity(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _jsonl_bytes_antigravity(values: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(canonical_json_bytes_antigravity(value) for value in values)


def _row_payload_antigravity(row: Exp464MethodFoldMetricAntigravity) -> dict[str, object]:
    return {
        "method": row.method,
        "seed": row.seed,
        "fold_index": row.fold_index,
        "normal_trials": row.normal_trials,
        "normal_alerts": row.normal_alerts,
        "normal_wilson_upper": row.normal_wilson_upper,
        "per_type_auc": dict(row.per_type_auc),
        "mean_auc": row.mean_auc,
        "scores_finite": row.scores_finite,
        "normal_scores": list(row.normal_scores),
        "primary_scores": {
            change_type: list(row.primary_scores[change_type])
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
        "confirmation_counts": dict(row.confirmation_counts),
        "confirmation_rates": dict(row.confirmation_rates),
        "pair_diagnostics": {
            change_type: dict(row.pair_diagnostics[change_type])
            for change_type in row.pair_diagnostics
        },
        "passed": row.passed,
        "failure_reasons": list(row.failure_reasons),
        "feature_dimension": row.feature_dimension,
        "provenance": row.provenance,
    }


def _summary_payload_antigravity(
    summary: Exp464MethodSummaryAntigravity,
) -> dict[str, object]:
    return {
        "method": summary.method,
        "fold_count": summary.fold_count,
        "passed_count": summary.passed_count,
        "failed_count": summary.failed_count,
        "min_per_type_auc": dict(summary.min_per_type_auc),
        "min_gradual_drift_auc": summary.min_gradual_drift_auc,
        "min_period_change_auc": summary.min_period_change_auc,
        "min_mean_auc": summary.min_mean_auc,
        "max_normal_wilson_upper": summary.max_normal_wilson_upper,
        "all_scores_finite": summary.all_scores_finite,
        "passed": summary.passed,
    }


def _decision_payload_antigravity(
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    return {
        "status": decision.status,
        "primary_candidate": decision.primary_candidate,
        "eligible_for_routing_research": decision.eligible_for_routing_research,
        "primary_passed_count": decision.primary_passed_count,
        "primary_required_count": decision.primary_required_count,
        "failure_reasons": list(decision.failure_reasons),
        "routing_applied": decision.routing_applied,
        "test_labels_opened": decision.test_labels_opened,
        "label_query_count": decision.label_query_count,
    }


def _context_payload_antigravity(
    authority: Exp463AuthorityAntigravity,
    *,
    method_fold_metrics_sha256: str,
) -> dict[str, object]:
    return {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "kind": FROZEN_CONFIG_ANTIGRAVITY.kind,
        "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
        "data_split": FROZEN_CONFIG_ANTIGRAVITY.data_split,
        "train_shape": list(FROZEN_CONFIG_ANTIGRAVITY.train_shape),
        "method_order": list(FROZEN_CONFIG_ANTIGRAVITY.method_order),
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "seeds": list(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds),
        "blocked_fold_count": FROZEN_CONFIG_ANTIGRAVITY.folds,
        "onset": FROZEN_CONFIG_ANTIGRAVITY.onset,
        "cosine_min": FROZEN_CONFIG_ANTIGRAVITY.cosine_min,
        "gates": {
            "per_type_auc_min": FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min,
            "mean_auc_min": FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min,
            "normal_wilson_upper_max": FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max,
            "gradual_drift_auc_min": FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_min,
            "gradual_drift_strict_floor": FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_strictly_above_exp463,
            "period_change_auc_global_min": FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_global_min,
            "historical_period_fold_thresholds": [
                {
                    "seed": item.seed,
                    "fold_index": item.fold_index,
                    "period_change_auc_min": item.period_change_auc_min,
                }
                for item in FROZEN_CONFIG_ANTIGRAVITY.historical_period_fold_thresholds
            ],
        },
        "predecessor": {
            "experiment": authority.predecessor_experiment,
            "terminal_closeout_commit": authority.terminal_closeout_commit,
            "task5_report_sha256": authority.task5_report_sha256,
            "artifact_map_sha256": authority.artifact_map_sha256,
        },
        "label_query_count": FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        "method_fold_metrics_sha256": method_fold_metrics_sha256,
    }


def _payloads_antigravity(
    qualification: Exp464QualificationContractAntigravity,
    decision: QualificationDecisionAntigravity,
    authority: Exp463AuthorityAntigravity,
) -> dict[str, bytes]:
    validate_exp463_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(qualification, authority)
    recomputed_decision = evaluate_qualification_antigravity(normalized, authority)
    if decision != recomputed_decision:
        raise ValueError("publication decision differs from evidence-derived decision")
    row_payloads = tuple(_row_payload_antigravity(row) for row in normalized.fold_metrics)
    rows_bytes = _jsonl_bytes_antigravity(row_payloads)
    payloads = {
        "context_antigravity.json": canonical_json_bytes_antigravity(
            _context_payload_antigravity(
                authority,
                method_fold_metrics_sha256=sha256(rows_bytes).hexdigest(),
            )
        ),
        "method_fold_metrics_antigravity.jsonl": rows_bytes,
        "failed_folds_antigravity.jsonl": _jsonl_bytes_antigravity(
            tuple(
                payload for payload in row_payloads if not payload["passed"]
            )
        ),
        "method_summary_antigravity.jsonl": _jsonl_bytes_antigravity(
            tuple(_summary_payload_antigravity(summary) for summary in normalized.summaries)
        ),
        "decision_antigravity.json": canonical_json_bytes_antigravity(
            _decision_payload_antigravity(decision)
        ),
        "dashboard_snapshot_antigravity.json": canonical_json_bytes_antigravity(
            export_dashboard_antigravity(normalized, decision)
        ),
    }
    if tuple(payloads) != _LEAVES_ANTIGRAVITY[:-1]:
        raise ValueError("publication payload inventory differs")
    return payloads


def publish_result_antigravity(
    root: Path,
    qualification: Exp464QualificationContractAntigravity,
    decision: QualificationDecisionAntigravity,
    authority: Exp463AuthorityAntigravity,
) -> PublishedQualificationAntigravity:
    official_root = Path(FROZEN_CONFIG_ANTIGRAVITY.official_result_root)
    if not isinstance(root, Path) or root == official_root or root.exists() or root.is_symlink():
        raise ValueError("publication destination must be a new private path")
    payloads = _payloads_antigravity(qualification, decision, authority)
    artifact_map = {
        name: sha256(payloads[name]).hexdigest()
        for name in _LEAVES_ANTIGRAVITY[:-1]
    }
    map_bytes = canonical_json_bytes_antigravity(artifact_map)
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
    leaf_sha256 = tuple(
        (name, sha256((root / name).read_bytes()).hexdigest())
        for name in _LEAVES_ANTIGRAVITY
    )
    return PublishedQualificationAntigravity(
        root=root,
        leaf_sha256=leaf_sha256,
        artifact_map_sha256=sha256(map_bytes).hexdigest(),
    )


__all__ = (
    "PublishedQualificationAntigravity",
    "_LEAVES_ANTIGRAVITY",
    "_context_payload_antigravity",
    "_decision_payload_antigravity",
    "_payloads_antigravity",
    "_row_payload_antigravity",
    "_summary_payload_antigravity",
    "canonical_json_bytes_antigravity",
    "publish_result_antigravity",
)
