"""Source-free replay for Exp461 Task 4 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _json_bytes_antigravity,
    _payloads_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .predecessor_antigravity import capture_exp460_period_authority_antigravity
from .qualification_antigravity import (
    MethodFoldMetricAntigravity,
    MethodQualificationAntigravity,
    summarize_methods_antigravity,
)


@dataclass(frozen=True, slots=True)
class QualificationReplayAntigravity:
    qualification: MethodQualificationAntigravity
    decision: QualificationDecisionAntigravity
    dashboard: dict[str, object]
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


def _strict_json_antigravity(payload: bytes) -> object:
    def no_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate artifact JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates_antigravity)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact JSON is invalid") from error
    if payload != _json_bytes_antigravity(value):
        raise ValueError("artifact JSON is not canonical")
    return value


def _strict_jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    return tuple(_strict_json_antigravity(line + b"\n") for line in payload.splitlines())


def _load_antigravity(root: Path) -> dict[str, bytes]:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("artifact root is invalid")
    if {entry.name for entry in root.iterdir()} != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("artifact inventory differs")
    leaves: dict[str, bytes] = {}
    for name in _LEAVES_ANTIGRAVITY:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact leaf is invalid")
        leaves[name] = path.read_bytes()
    artifact_map = _strict_json_antigravity(leaves["artifact_sha256_antigravity.json"])
    expected = {name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    if artifact_map != expected:
        raise ValueError("artifact hash map differs")
    return leaves


def _context_antigravity(payload: bytes) -> None:
    value = _strict_json_antigravity(payload)
    expected = {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "kind": FROZEN_CONFIG_ANTIGRAVITY.kind,
        "run_identity": FROZEN_CONFIG_ANTIGRAVITY.run_identity,
        "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
        "data_split": FROZEN_CONFIG_ANTIGRAVITY.data_split,
        "approved_design_sha256": FROZEN_CONFIG_ANTIGRAVITY.approved_design_sha256,
        "method_order": list(FROZEN_CONFIG_ANTIGRAVITY.method_order),
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "primary_types": list(FROZEN_CONFIG_ANTIGRAVITY.primary_types),
        "hydra_seeds": list(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds),
        "folds": FROZEN_CONFIG_ANTIGRAVITY.folds,
        "per_type_auc_min": FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min,
        "pooled_auc_min": FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min,
        "normal_wilson_upper_max": FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max,
        "label_query_count": FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        "official_command_count": FROZEN_CONFIG_ANTIGRAVITY.official_command_invocations_must_remain,
    }
    for key, expected_value in expected.items():
        if not isinstance(value, dict) or value.get(key) != expected_value:
            raise ValueError("context artifact differs from frozen Exp461 config")
    authority = capture_exp460_period_authority_antigravity()
    predecessor = value.get("predecessor") if isinstance(value, dict) else None
    if predecessor != {
        "experiment": authority.predecessor_experiment,
        "truth_commit": authority.truth_commit,
        "artifact_map_sha256": authority.artifact_map_sha256,
        "decision_sha256": authority.decision_sha256,
        "strict_route": authority.strict_route,
        "failed_folds": [
            {
                "seed": row.seed,
                "fold_index": row.fold_index,
                "period_change_auc": row.period_change_auc,
            }
            for row in authority.failed_folds
        ],
    }:
        raise ValueError("context predecessor differs")


def _metric_row_antigravity(value: object) -> MethodFoldMetricAntigravity:
    keys = {
        "method",
        "seed",
        "fold_index",
        "normal_trials",
        "normal_alerts",
        "normal_wilson_upper",
        "per_type_auc",
        "pooled_auc",
        "scores_finite",
        "passed",
        "failure_reasons",
    }
    if type(value) is not dict or set(value) != keys or type(value["failure_reasons"]) is not list:
        raise ValueError("method metric row is invalid")
    return MethodFoldMetricAntigravity(
        method=value["method"],
        seed=value["seed"],
        fold_index=value["fold_index"],
        normal_trials=value["normal_trials"],
        normal_alerts=value["normal_alerts"],
        normal_wilson_upper=value["normal_wilson_upper"],
        per_type_auc=value["per_type_auc"],
        pooled_auc=value["pooled_auc"],
        scores_finite=value["scores_finite"],
        passed=value["passed"],
        failure_reasons=tuple(value["failure_reasons"]),
    )


def _decision_antigravity(value: object) -> QualificationDecisionAntigravity:
    keys = {
        "status",
        "primary_candidate",
        "eligible_for_routing_research",
        "primary_passed_count",
        "primary_required_count",
        "failure_reasons",
        "routing_applied",
        "test_labels_opened",
        "label_query_count",
    }
    if type(value) is not dict or set(value) != keys or type(value["failure_reasons"]) is not list:
        raise ValueError("decision artifact is invalid")
    return QualificationDecisionAntigravity(
        status=value["status"],
        primary_candidate=value["primary_candidate"],
        eligible_for_routing_research=value["eligible_for_routing_research"],
        primary_passed_count=value["primary_passed_count"],
        primary_required_count=value["primary_required_count"],
        failure_reasons=tuple(value["failure_reasons"]),
        routing_applied=value["routing_applied"],
        test_labels_opened=value["test_labels_opened"],
        label_query_count=value["label_query_count"],
    )


def inspect_replay_at_antigravity(root: Path) -> QualificationReplayAntigravity:
    leaves = _load_antigravity(root)
    _context_antigravity(leaves["context_antigravity.json"])
    rows = tuple(_metric_row_antigravity(row) for row in _strict_jsonl_antigravity(leaves["method_fold_metrics_antigravity.jsonl"]))
    qualification = MethodQualificationAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        method_order=FROZEN_CONFIG_ANTIGRAVITY.method_order,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        fold_metrics=rows,
        summaries=summarize_methods_antigravity(rows),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )
    authority = capture_exp460_period_authority_antigravity()
    decision = evaluate_qualification_antigravity(authority, qualification)
    stored_decision = _decision_antigravity(_strict_json_antigravity(leaves["decision_antigravity.json"]))
    if stored_decision != decision:
        raise ValueError("decision artifact differs from replay")
    expected_payloads = _payloads_antigravity(authority, qualification, decision)
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if leaves[name] != expected_payloads[name]:
            raise ValueError("artifact payload differs from replay")
    dashboard = _strict_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])
    return QualificationReplayAntigravity(
        qualification=qualification,
        decision=decision,
        dashboard=dashboard if isinstance(dashboard, dict) else {},
        leaf_sha256=tuple((name, sha256(leaves[name]).hexdigest()) for name in _LEAVES_ANTIGRAVITY),
        artifact_map_sha256=sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest(),
    )


__all__ = ("QualificationReplayAntigravity", "inspect_replay_at_antigravity")
