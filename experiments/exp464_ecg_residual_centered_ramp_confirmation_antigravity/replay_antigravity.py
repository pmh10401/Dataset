"""Semantic replay for Exp464 Task 4 result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _context_payload_antigravity,
    _decision_payload_antigravity,
    _payloads_antigravity,
    _row_payload_antigravity,
    _summary_payload_antigravity,
    canonical_json_bytes_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import (
    Exp464MethodFoldMetricAntigravity,
    Exp464QualificationContractAntigravity,
    normalize_qualification_antigravity,
    summarize_methods_antigravity,
)

_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_CONFIRMATION_TYPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY
_PAIR_TYPES_ANTIGRAVITY = ("gradual_drift", "period_change")
_PAIR_STAT_KEYS_ANTIGRAVITY = ("total_pairs", "fixed_pairs", "broken_pairs")


@dataclass(frozen=True, slots=True)
class QualificationReplayAntigravity:
    qualification: Exp464QualificationContractAntigravity
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
    if payload != canonical_json_bytes_antigravity(value):
        raise ValueError("artifact JSON is not canonical")
    return value


def _strict_jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if payload == b"":
        return ()
    if not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    return tuple(_strict_json_antigravity(line + b"\n") for line in payload.splitlines())


def _load_antigravity(root: Path) -> dict[str, bytes]:
    if (
        not isinstance(root, Path)
        or root == Path(FROZEN_CONFIG_ANTIGRAVITY.official_result_root)
        or not root.is_dir()
        or root.is_symlink()
    ):
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


def _metric_row_antigravity(value: object) -> Exp464MethodFoldMetricAntigravity:
    keys = {
        "method",
        "seed",
        "fold_index",
        "normal_trials",
        "normal_alerts",
        "normal_wilson_upper",
        "per_type_auc",
        "mean_auc",
        "scores_finite",
        "normal_scores",
        "primary_scores",
        "confirmation_counts",
        "confirmation_rates",
        "pair_diagnostics",
        "passed",
        "failure_reasons",
        "feature_dimension",
        "provenance",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or type(value["failure_reasons"]) is not list
        or type(value["normal_scores"]) is not list
        or type(value["primary_scores"]) is not dict
        or type(value["per_type_auc"]) is not dict
        or type(value["confirmation_counts"]) is not dict
        or type(value["confirmation_rates"]) is not dict
        or type(value["pair_diagnostics"]) is not dict
    ):
        raise ValueError("method metric row is invalid")
    primary = value["primary_scores"]
    per_type = value["per_type_auc"]
    confirmation_counts = value["confirmation_counts"]
    confirmation_rates = value["confirmation_rates"]
    pair_diagnostics = value["pair_diagnostics"]
    assert isinstance(primary, dict)
    assert isinstance(per_type, dict)
    assert isinstance(confirmation_counts, dict)
    assert isinstance(confirmation_rates, dict)
    assert isinstance(pair_diagnostics, dict)
    return Exp464MethodFoldMetricAntigravity(
        method=value["method"],
        seed=value["seed"],
        fold_index=value["fold_index"],
        normal_trials=value["normal_trials"],
        normal_alerts=value["normal_alerts"],
        normal_wilson_upper=value["normal_wilson_upper"],
        per_type_auc={
            change_type: per_type[change_type]
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
        mean_auc=value["mean_auc"],
        scores_finite=value["scores_finite"],
        normal_scores=tuple(value["normal_scores"]),
        primary_scores={
            change_type: tuple(primary[change_type])
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
        confirmation_counts={
            name: confirmation_counts[name]
            for name in _CONFIRMATION_TYPES_ANTIGRAVITY
        },
        confirmation_rates={
            name: confirmation_rates[name]
            for name in _CONFIRMATION_TYPES_ANTIGRAVITY
        },
        pair_diagnostics={
            name: {
                key: pair_diagnostics[name][key]
                for key in _PAIR_STAT_KEYS_ANTIGRAVITY
            }
            for name in _PAIR_TYPES_ANTIGRAVITY
        },
        passed=value["passed"],
        failure_reasons=tuple(value["failure_reasons"]),
        feature_dimension=value["feature_dimension"],
        provenance=value["provenance"],
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
    from .predecessor_antigravity import capture_exp463_authority_antigravity

    leaves = _load_antigravity(root)
    authority = capture_exp463_authority_antigravity()
    context_value = _strict_json_antigravity(leaves["context_antigravity.json"])
    if type(context_value) is not dict:
        raise ValueError("context artifact must be an object")
    method_fold_metrics_sha256 = context_value.get("method_fold_metrics_sha256")
    if type(method_fold_metrics_sha256) is not str:
        raise ValueError("context method-fold digest is missing")
    if context_value != _context_payload_antigravity(
        authority,
        method_fold_metrics_sha256=method_fold_metrics_sha256,
    ):
        raise ValueError("context artifact differs from frozen Exp464 contract")
    if method_fold_metrics_sha256 != sha256(leaves["method_fold_metrics_antigravity.jsonl"]).hexdigest():
        raise ValueError("method-fold digest differs from context")

    rows = tuple(
        _metric_row_antigravity(row)
        for row in _strict_jsonl_antigravity(leaves["method_fold_metrics_antigravity.jsonl"])
    )
    qualification = Exp464QualificationContractAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        method_order=FROZEN_CONFIG_ANTIGRAVITY.method_order,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        fold_metrics=rows,
        summaries=summarize_methods_antigravity(rows),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )
    normalized = normalize_qualification_antigravity(qualification, authority)
    if normalized != qualification:
        raise ValueError("stored metric rows differ from score-evidence replay")

    expected_failed = tuple(
        _row_payload_antigravity(row) for row in normalized.fold_metrics if not row.passed
    )
    if _strict_jsonl_antigravity(leaves["failed_folds_antigravity.jsonl"]) != expected_failed:
        raise ValueError("failed-fold artifact differs from replay")
    expected_summaries = tuple(
        _summary_payload_antigravity(summary) for summary in normalized.summaries
    )
    if _strict_jsonl_antigravity(leaves["method_summary_antigravity.jsonl"]) != expected_summaries:
        raise ValueError("method summary artifact differs from replay")

    decision = evaluate_qualification_antigravity(normalized, authority)
    stored_decision = _decision_antigravity(_strict_json_antigravity(leaves["decision_antigravity.json"]))
    if stored_decision != decision:
        raise ValueError("decision artifact differs from replay")

    expected_payloads = _payloads_antigravity(normalized, decision, authority)
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if leaves[name] != expected_payloads[name]:
            raise ValueError("artifact payload differs from semantic replay")
    dashboard_value = _strict_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])
    if type(dashboard_value) is not dict:
        raise ValueError("dashboard artifact must be an object")
    return QualificationReplayAntigravity(
        qualification=normalized,
        decision=decision,
        dashboard=dashboard_value,
        leaf_sha256=tuple(
            (name, sha256(leaves[name]).hexdigest()) for name in _LEAVES_ANTIGRAVITY
        ),
        artifact_map_sha256=sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest(),
    )


__all__ = ("QualificationReplayAntigravity", "inspect_replay_at_antigravity")
