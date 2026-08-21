"""Canonical seven-leaf create-only publication for Exp463 Task 4."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp462AuthorityAntigravity,
    Exp463MethodFoldMetricAntigravity,
    Exp463QualificationContractAntigravity,
    validate_exp462_authority_antigravity,
)
from .dashboard_export_antigravity import export_dashboard_antigravity
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import normalize_qualification_antigravity


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "method_fold_metrics_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


@dataclass(frozen=True, slots=True)
class PublishedQualificationAntigravity:
    root: Path
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


def _json_bytes_antigravity(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _jsonl_bytes_antigravity(values: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(_json_bytes_antigravity(value) for value in values)


def _row_payload_antigravity(row: Exp463MethodFoldMetricAntigravity) -> dict[str, object]:
    return {
        "method": row.method,
        "seed": row.seed,
        "fold_index": row.fold_index,
        "normal_trials": row.normal_trials,
        "normal_alerts": row.normal_alerts,
        "normal_wilson_upper": row.normal_wilson_upper,
        "per_type_auc": dict(row.per_type_auc),
        "pooled_auc": row.pooled_auc,
        "scores_finite": row.scores_finite,
        "normal_scores": list(row.normal_scores),
        "primary_scores": {
            change_type: list(row.primary_scores[change_type])
            for change_type in FROZEN_CONFIG_ANTIGRAVITY.change_order
        },
        "passed": row.passed,
        "failure_reasons": list(row.failure_reasons),
    }


def _score_evidence_bytes_antigravity(
    rows: tuple[Exp463MethodFoldMetricAntigravity, ...],
) -> bytes:
    return _jsonl_bytes_antigravity(
        tuple(
            {
                "method": row.method,
                "seed": row.seed,
                "fold_index": row.fold_index,
                "normal_scores": list(row.normal_scores),
                "primary_scores": {
                    change_type: list(row.primary_scores[change_type])
                    for change_type in FROZEN_CONFIG_ANTIGRAVITY.change_order
                },
            }
            for row in rows
        )
    )


def _context_payload_antigravity(
    authority: Exp462AuthorityAntigravity,
    *,
    score_evidence_sha256: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "kind": FROZEN_CONFIG_ANTIGRAVITY.kind,
        "run_identity": FROZEN_CONFIG_ANTIGRAVITY.run_identity,
        "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
        "data_split": FROZEN_CONFIG_ANTIGRAVITY.data_split,
        "train_shape": list(FROZEN_CONFIG_ANTIGRAVITY.train_shape),
        "method_order": list(FROZEN_CONFIG_ANTIGRAVITY.method_order),
        "method_definitions": dict(FROZEN_CONFIG_ANTIGRAVITY.method_definitions),
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "seeds": list(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds),
        "blocked_fold_count": FROZEN_CONFIG_ANTIGRAVITY.folds,
        "onset": FROZEN_CONFIG_ANTIGRAVITY.onset,
        "alpha": FROZEN_CONFIG_ANTIGRAVITY.alpha,
        "gate": FROZEN_CONFIG_ANTIGRAVITY.gate,
        "gates": {
            "per_type_auc_min": FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min,
            "pooled_auc_min": FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min,
            "normal_wilson_upper_max": FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max,
            "gradual_drift_auc_min": FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_min,
            "gradual_drift_auc_strictly_above_exp462": FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_strictly_above_exp462,
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
            "reviewed_code_commit": authority.reviewed_code_commit,
            "terminal_closeout_commit": authority.terminal_closeout_commit,
            "status": authority.status,
            "primary_candidate": authority.primary_candidate,
            "passed_record_count": authority.passed_record_count,
            "required_record_count": authority.required_record_count,
            "failed_gradual_drift_auc": authority.failed_gradual_drift_auc,
            "period_min_auc": authority.period_min_auc,
            "historical_period_folds": [
                {
                    "seed": item.seed,
                    "fold_index": item.fold_index,
                    "period_change_auc": item.period_change_auc,
                }
                for item in authority.historical_period_folds
            ],
            "result_row_count": authority.result_snapshot.row_count,
            "result_summary_count": authority.result_snapshot.summary_count,
            "result_leaf_count": authority.result_snapshot.leaf_count,
            "leaf_sha256": dict(authority.private_sha256),
        },
        "label_query_count": FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        "official_command_count": FROZEN_CONFIG_ANTIGRAVITY.official_command_invocations_must_remain,
    }
    if score_evidence_sha256 is not None:
        payload["score_evidence_sha256"] = score_evidence_sha256
    return payload


def _summary_payload_antigravity(
    qualification: Exp463QualificationContractAntigravity,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "method": summary.method,
            "fold_count": summary.fold_count,
            "passed_count": summary.passed_count,
            "failed_count": summary.failed_count,
            "min_per_type_auc": dict(summary.min_per_type_auc),
            "min_gradual_drift_auc": summary.min_gradual_drift_auc,
            "min_period_change_auc": summary.min_period_change_auc,
            "min_pooled_auc": summary.min_pooled_auc,
            "max_normal_wilson_upper": summary.max_normal_wilson_upper,
            "all_scores_finite": summary.all_scores_finite,
            "passed": summary.passed,
        }
        for summary in qualification.summaries
    )


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


def _payloads_antigravity(
    authority: Exp462AuthorityAntigravity,
    qualification: Exp463QualificationContractAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, bytes]:
    trusted_authority = validate_exp462_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(trusted_authority, qualification)
    if any(
        not row.scores_finite
        or any(not all(isfinite(value) for value in row.primary_scores[key]) for key in row.primary_scores)
        or not all(isfinite(value) for value in row.normal_scores)
        for row in normalized.fold_metrics
    ):
        raise ValueError("non-finite score evidence cannot be published canonically")
    recomputed_decision = evaluate_qualification_antigravity(trusted_authority, normalized)
    if decision != recomputed_decision:
        raise ValueError("publication decision differs from evidence-derived decision")
    rows_payload = tuple(_row_payload_antigravity(row) for row in normalized.fold_metrics)
    score_evidence_sha256 = sha256(_score_evidence_bytes_antigravity(normalized.fold_metrics)).hexdigest()
    payloads = {
        "context_antigravity.json": _json_bytes_antigravity(
            _context_payload_antigravity(trusted_authority, score_evidence_sha256=score_evidence_sha256)
        ),
        "method_fold_metrics_antigravity.jsonl": _jsonl_bytes_antigravity(rows_payload),
        "failed_folds_antigravity.jsonl": _jsonl_bytes_antigravity(
            tuple(row for row in rows_payload if not row["passed"])
        ),
        "method_summary_antigravity.jsonl": _jsonl_bytes_antigravity(_summary_payload_antigravity(normalized)),
        "decision_antigravity.json": _json_bytes_antigravity(_decision_payload_antigravity(decision)),
        "dashboard_snapshot_antigravity.json": _json_bytes_antigravity(
            export_dashboard_antigravity(trusted_authority, normalized, decision)
        ),
    }
    if tuple(payloads) != _LEAVES_ANTIGRAVITY[:-1]:
        raise ValueError("publication payload inventory differs")
    return payloads


def publish_result_antigravity(
    root: Path,
    authority: Exp462AuthorityAntigravity,
    qualification: Exp463QualificationContractAntigravity,
    decision: QualificationDecisionAntigravity,
) -> PublishedQualificationAntigravity:
    """Create exactly seven canonical leaves at a new private destination."""

    official_root = Path(FROZEN_CONFIG_ANTIGRAVITY.official_result_root)
    if (
        not isinstance(root, Path)
        or root == official_root
        or root.exists()
        or root.is_symlink()
    ):
        raise ValueError("publication destination must be a new private path")
    payloads = _payloads_antigravity(authority, qualification, decision)
    artifact_map = {
        name: sha256(payloads[name]).hexdigest()
        for name in _LEAVES_ANTIGRAVITY[:-1]
    }
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
    "_json_bytes_antigravity",
    "_jsonl_bytes_antigravity",
    "_payloads_antigravity",
    "_row_payload_antigravity",
    "_score_evidence_bytes_antigravity",
    "_summary_payload_antigravity",
    "publish_result_antigravity",
)
