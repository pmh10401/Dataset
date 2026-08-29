"""Semantic replay for Exp465 Task 4 result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from .artifacts_antigravity import _LEAVES_ANTIGRAVITY, _payloads_antigravity, _row_payload_antigravity, _summary_payload_antigravity, canonical_json_bytes_antigravity
from .contracts_antigravity import Exp465FoldResultContractAntigravity
from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_qualification_antigravity
from .qualification_antigravity import Exp465MethodFoldMetricAntigravity, Exp465MethodSummaryAntigravity, Exp465QualificationResultAntigravity, _METHODS_ANTIGRAVITY, recompute_qualification_antigravity


@dataclass(frozen=True, slots=True)
class QualificationReplayAntigravity:
    qualification: Exp465QualificationResultAntigravity
    decision: QualificationDecisionAntigravity
    dashboard: dict[str, object]
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


def _strict_json_antigravity(payload: bytes) -> object:
    def duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=duplicates, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("artifact JSON is invalid") from error
    if payload != canonical_json_bytes_antigravity(value):
        raise ValueError("artifact JSON is not canonical")
    return value


def _jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if not payload:
        return ()
    if not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    return tuple(_strict_json_antigravity(line + b"\n") for line in payload.splitlines())


def _load_antigravity(root: Path) -> dict[str, bytes]:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink() or {item.name for item in root.iterdir()} != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("artifact inventory differs")
    leaves = {name: (root / name).read_bytes() for name in _LEAVES_ANTIGRAVITY}
    artifact_map = _strict_json_antigravity(leaves[_LEAVES_ANTIGRAVITY[-1]])
    expected = {name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    if artifact_map != expected:
        raise ValueError("artifact hash map differs")
    return leaves


def _metric_antigravity(value: object) -> Exp465MethodFoldMetricAntigravity:
    if type(value) is not dict:
        raise ValueError("metric row is invalid")
    try:
        return Exp465MethodFoldMetricAntigravity(
            method=value["method"], seed=value["seed"], fold_index=value["fold_index"], normal_scores=tuple(value["normal_scores"]),
            primary_scores={key: tuple(value["primary_scores"][key]) for key in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")},
            confirmation_counts={key: value["confirmation_counts"][key] for key in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")},
            query_cosines={key: tuple(value["query_cosines"][key]) for key in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")} if value["query_cosines"] else {}, per_type_auc=tuple(value["per_type_auc"]),
            fit_amplitudes=tuple(value["fit_amplitudes"]), query_amplitudes={key: tuple(value["query_amplitudes"][key]) for key in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")} if value["query_amplitudes"] else {}, raw_evidence_kind=value["raw_evidence_kind"], raw_fit_vector_blob=value["raw_fit_vector_blob"], raw_query_vector_blobs={key: value["raw_query_vector_blobs"][key] for key in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")} if value["raw_query_vector_blobs"] else {},
            mean_auc=value["mean_auc"], all_scores_finite=value["all_scores_finite"], normal_wilson_upper=value["normal_wilson_upper"],
            pair_diagnostics={name: {key: value["pair_diagnostics"][name][key] for key in ("total_pairs", "fixed_pairs", "broken_pairs")} for name in ("gradual_drift", "period_change")}, passed=value["passed"], failure_reasons=tuple(value["failure_reasons"]),
            feature_dimension=value["feature_dimension"], provenance=value["provenance"],
            input_train140_blob=value["input_train140_blob"],
            input_fit_indices=tuple(value["input_fit_indices"]),
            input_query_indices=tuple(value["input_query_indices"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("metric row is invalid") from error


def _qualification_antigravity(rows: tuple[Exp465MethodFoldMetricAntigravity, ...]) -> Exp465QualificationResultAntigravity:
    c8 = tuple(row for row in rows if row.method == "C8")
    result_rows = tuple(Exp465FoldResultContractAntigravity(seed=row.seed, fold_index=row.fold_index, per_type_auc=row.per_type_auc, mean_auc=row.mean_auc,
        c0_gradual_auc=next(item.per_type_auc[1] for item in rows if item.method == "C0" and item.seed == row.seed and item.fold_index == row.fold_index),
        c0_period_auc=next(item.per_type_auc[3] for item in rows if item.method == "C0" and item.seed == row.seed and item.fold_index == row.fold_index),
        c8_gradual_auc=row.per_type_auc[1], c8_period_auc=row.per_type_auc[3], all_scores_finite=row.all_scores_finite, passed=row.passed, failure_reasons=row.failure_reasons) for row in c8)
    summaries = tuple(_summary_from_rows_antigravity(rows, method) for method in _METHODS_ANTIGRAVITY)
    seeds = {seed: sum(row.confirmation_counts["normal"] for row in rows if row.method == "C8" and row.seed == seed) for seed in (20260721, 20261721)}
    return Exp465QualificationResultAntigravity(rows=result_rows, seed_normal_confirmations=seeds, label_query_count=0, official_evaluate_count=0, artifact_map_sha256=None, method_rows=rows, summaries=summaries,
        seed_normal_wilson_upper={seed: __import__("experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity", fromlist=["wilson_upper_antigravity"]).wilson_upper_antigravity(seeds[seed], 69) for seed in seeds})


def _summary_from_rows_antigravity(rows: tuple[Exp465MethodFoldMetricAntigravity, ...], method: str) -> Exp465MethodSummaryAntigravity:
    selected = tuple(row for row in rows if row.method == method)
    pair = {name: MappingProxyType({key: sum(row.pair_diagnostics[name][key] for row in selected) for key in ("total_pairs", "fixed_pairs", "broken_pairs")}) for name in ("gradual_drift", "period_change")}
    return Exp465MethodSummaryAntigravity(method=method, fold_count=len(selected), passed_count=sum(row.passed for row in selected), failed_count=sum(not row.passed for row in selected),
        min_per_type_auc=tuple(min(row.per_type_auc[index] for row in selected) for index in range(4)), min_mean_auc=min(row.mean_auc for row in selected),
        max_normal_wilson_upper=max(row.normal_wilson_upper for row in selected), normal_confirmation_count=sum(row.confirmation_counts["normal"] for row in selected),
        all_scores_finite=all(row.all_scores_finite for row in selected), pair_diagnostics=MappingProxyType(pair), passed=all(row.passed for row in selected))


def _decision_antigravity(value: object) -> QualificationDecisionAntigravity:
    if type(value) is not dict:
        raise ValueError("decision artifact is invalid")
    try:
        return QualificationDecisionAntigravity(status=value["status"], primary_candidate=value["primary_candidate"], eligible_for_routing_research=value["eligible_for_routing_research"], primary_passed_count=value["primary_passed_count"], primary_required_count=value["primary_required_count"], failure_reasons=tuple(value["failure_reasons"]), routing_applied=value["routing_applied"], test_labels_opened=value["test_labels_opened"], label_query_count=value["label_query_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("decision artifact is invalid") from error


def inspect_replay_at_antigravity(root: Path) -> QualificationReplayAntigravity:
    leaves = _load_antigravity(root)
    metrics = tuple(_metric_antigravity(value) for value in _jsonl_antigravity(leaves["method_fold_metrics_antigravity.jsonl"]))
    qualification = _qualification_antigravity(metrics)
    replayed = recompute_qualification_antigravity(qualification)
    decision = evaluate_qualification_antigravity(replayed)
    stored = _decision_antigravity(_strict_json_antigravity(leaves["decision_antigravity.json"]))
    if stored != decision or _payloads_antigravity(replayed, decision) != {name: leaves[name] for name in _LEAVES_ANTIGRAVITY[:-1]}:
        raise ValueError("artifact payload differs from semantic replay")
    dashboard = _strict_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])
    if type(dashboard) is not dict:
        raise ValueError("dashboard artifact is invalid")
    return QualificationReplayAntigravity(replayed, decision, dashboard, tuple((name, sha256(leaves[name]).hexdigest()) for name in _LEAVES_ANTIGRAVITY), sha256(leaves[_LEAVES_ANTIGRAVITY[-1]]).hexdigest())


__all__ = ("QualificationReplayAntigravity", "inspect_replay_at_antigravity")
