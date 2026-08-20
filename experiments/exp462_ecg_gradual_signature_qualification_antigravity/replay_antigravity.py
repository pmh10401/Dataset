"""Source-free semantic replay and strict inspection for Exp462 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _context_payload_antigravity,
    _decision_payload_antigravity,
    _json_bytes_antigravity,
    _payloads_antigravity,
    _row_payload_antigravity,
    _score_evidence_bytes_antigravity,
    _summary_payload_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp461GradualAuthorityAntigravity,
)
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import (
    MethodFoldMetricAntigravity,
    MethodQualificationAntigravity,
    normalize_qualification_antigravity,
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

    def reject_constant_antigravity(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates_antigravity,
            parse_constant=reject_constant_antigravity,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("artifact JSON is invalid") from error
    if payload != _json_bytes_antigravity(value):
        raise ValueError("artifact JSON is not canonical")
    return value


def _strict_jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if payload == b"":
        return ()
    if not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    rows: list[object] = []
    for line in payload.splitlines():
        rows.append(_strict_json_antigravity(line + b"\n"))
    return tuple(rows)


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
    expected = {
        name: sha256(leaves[name]).hexdigest()
        for name in _LEAVES_ANTIGRAVITY[:-1]
    }
    if artifact_map != expected:
        raise ValueError("artifact hash map differs")
    return leaves


def _context_antigravity(
    payload: bytes,
    authority: Exp461GradualAuthorityAntigravity,
) -> str:
    value = _strict_json_antigravity(payload)
    if not isinstance(value, dict):
        raise ValueError("context artifact must be an object")
    score_evidence_sha256 = value.pop("score_evidence_sha256", None)
    if type(score_evidence_sha256) is not str:
        raise ValueError("context score-evidence digest is missing")
    if value != _context_payload_antigravity(authority):
        raise ValueError("context artifact differs from the frozen Exp462 contract")
    if value.get("label_query_count") != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
        raise ValueError("context label query count drifted")
    return score_evidence_sha256


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
        "normal_scores",
        "primary_scores",
        "passed",
        "failure_reasons",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or type(value["failure_reasons"]) is not list
        or type(value["normal_scores"]) is not list
        or type(value["primary_scores"]) is not dict
    ):
        raise ValueError("method metric row is invalid")
    primary = value["primary_scores"]
    if set(primary) != set(FROZEN_CONFIG_ANTIGRAVITY.change_order):
        raise ValueError("method metric primary score order differs")
    if not isinstance(value["per_type_auc"], dict):
        raise ValueError("method metric AUC payload is invalid")
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
        normal_scores=tuple(value["normal_scores"]),
        primary_scores={
            change_type: tuple(primary[change_type])
            for change_type in FROZEN_CONFIG_ANTIGRAVITY.change_order
        },
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
    """Inspect a private root without loading source data or TEST labels."""

    leaves = _load_antigravity(root)
    from .predecessor_antigravity import capture_exp461_gradual_authority_antigravity

    authority = capture_exp461_gradual_authority_antigravity()
    context_score_evidence_sha256 = _context_antigravity(
        leaves["context_antigravity.json"],
        authority,
    )
    raw_rows = _strict_jsonl_antigravity(leaves["method_fold_metrics_antigravity.jsonl"])
    rows = tuple(_metric_row_antigravity(row) for row in raw_rows)
    qualification = MethodQualificationAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        method_order=FROZEN_CONFIG_ANTIGRAVITY.method_order,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        fold_metrics=rows,
        summaries=summarize_methods_antigravity(rows),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )
    normalized = normalize_qualification_antigravity(authority, qualification)
    if normalized != qualification:
        raise ValueError("stored metric rows differ from score-evidence replay")
    actual_score_evidence_sha256 = sha256(
        _score_evidence_bytes_antigravity(normalized.fold_metrics)
    ).hexdigest()
    if context_score_evidence_sha256 != actual_score_evidence_sha256:
        raise ValueError("score evidence digest differs from context")

    failed_rows = _strict_jsonl_antigravity(leaves["failed_folds_antigravity.jsonl"])
    expected_failed = tuple(
        _row_payload_antigravity(row)
        for row in normalized.fold_metrics
        if not row.passed
    )
    if failed_rows != expected_failed:
        raise ValueError("failed-fold artifact differs from replay")

    stored_summary = _strict_json_antigravity(leaves["method_summary_antigravity.json"])
    if stored_summary != _summary_payload_antigravity(normalized):
        raise ValueError("method summary artifact differs from replay")

    decision = evaluate_qualification_antigravity(authority, normalized)
    stored_decision = _decision_antigravity(_strict_json_antigravity(leaves["decision_antigravity.json"]))
    if stored_decision != decision:
        raise ValueError("decision artifact differs from replay")

    expected_payloads = _payloads_antigravity(authority, normalized, decision)
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if leaves[name] != expected_payloads[name]:
            raise ValueError("artifact payload differs from semantic replay")
    dashboard_value = _strict_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])
    if not isinstance(dashboard_value, dict):
        raise ValueError("dashboard artifact must be an object")
    return QualificationReplayAntigravity(
        qualification=normalized,
        decision=decision,
        dashboard=dashboard_value,
        leaf_sha256=tuple(
            (name, sha256(leaves[name]).hexdigest())
            for name in _LEAVES_ANTIGRAVITY
        ),
        artifact_map_sha256=sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest(),
    )


__all__ = ("QualificationReplayAntigravity", "inspect_replay_at_antigravity")
