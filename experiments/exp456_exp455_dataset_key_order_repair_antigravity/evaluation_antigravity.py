"""Evaluate the four frozen Exp456 comparisons without changing routing state."""

from __future__ import annotations

from fractions import Fraction
import re
from typing import Sequence

from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.config_antigravity import EVALUATION_LEAVES_ANTIGRAVITY

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    ComparisonMetricAntigravity,
    ConfusionAntigravity,
    DatasetRoutingAuthorityAntigravity,
    DatasetRoutingFeaturesAntigravity,
    FoldDecisionAntigravity,
    FrozenRoutingAuthorityAntigravity,
    RoutingEvaluationAntigravity,
)
from .router_antigravity import leave_one_dataset_out_antigravity


_COMPARISON_NAMES_ANTIGRAVITY = (
    "always_baseline",
    "family_wide_local_gap",
    "oracle_per_dataset",
    "lodo_router",
)
_TARGET_METHODS_ANTIGRAVITY = {"ECG5000": "hydra_local_gap", "ChlorineConcentration": "multirocket_local_gap"}
_EVIDENCE_BY_EVALUATION_ID_ANTIGRAVITY: dict[int, tuple[tuple[DatasetRoutingAuthorityAntigravity, ...], tuple[DatasetRoutingFeaturesAntigravity, ...], tuple[tuple[str, str], ...]]] = {}


def _valid_confusion_antigravity(value: object) -> ConfusionAntigravity:
    if type(value) is not ConfusionAntigravity or any(type(item) is not int or item < 0 for item in (value.tp, value.fp, value.fn, value.tn)):
        raise ValueError("confusion must contain non-negative integers")
    return value


def _validate_exp454_inventory_antigravity(authority: FrozenRoutingAuthorityAntigravity) -> tuple[tuple[str, str], ...]:
    inventory = authority.exp454_leaf_sha256
    if (
        type(inventory) is not tuple
        or any(type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or type(item[1]) is not str or not re.fullmatch(r"[0-9a-f]{64}", item[1]) for item in inventory)
        or tuple(name for name, _digest in inventory) != tuple(sorted(EVALUATION_LEAVES_ANTIGRAVITY))
    ):
        raise ValueError("Exp454 frozen 12-leaf inventory differs")
    configured = dict(FROZEN_CONFIG_ANTIGRAVITY.exp454_sha256)
    expected = {
        "artifact_sha256_antigravity.json": configured["artifact_map"],
        "evaluation_antigravity.json": configured["evaluation"],
        "decision_antigravity.json": configured["decision"],
    }
    if any(dict(inventory)[name] != digest for name, digest in expected.items()):
        raise ValueError("Exp454 frozen inventory differs from Exp456 config")
    return inventory


def _sum_antigravity(rows: Sequence[ConfusionAntigravity]) -> ConfusionAntigravity:
    return ConfusionAntigravity(
        sum(row.tp for row in rows), sum(row.fp for row in rows),
        sum(row.fn for row in rows), sum(row.tn for row in rows),
    )


def _f1_antigravity(confusion: ConfusionAntigravity) -> Fraction:
    denominator = 2 * confusion.tp + confusion.fp + confusion.fn
    return Fraction(0, 1) if denominator == 0 else Fraction(2 * confusion.tp, denominator)


def _prefer_candidate_antigravity(baseline: ConfusionAntigravity, candidate: ConfusionAntigravity) -> bool:
    """Use the preregistered FN, FP, exact-F1, baseline ordering."""
    baseline_key = (baseline.fn, baseline.fp, -_f1_antigravity(baseline), 0)
    candidate_key = (candidate.fn, candidate.fp, -_f1_antigravity(candidate), 1)
    return candidate_key < baseline_key


def _unique_dataset_map_antigravity(rows: tuple[object, ...], kind: str) -> dict[str, object]:
    """Bind an eight-row frozen input by its exact, unique string dataset key."""
    result: dict[str, object] = {}
    for row in rows:
        dataset = getattr(row, "dataset", None)
        if type(dataset) is not str or not dataset or dataset in result:
            raise ValueError(f"{kind} dataset keys are invalid")
        result[dataset] = row
    return result


def align_targets_by_dataset_antigravity(
    targets: tuple[DatasetRoutingAuthorityAntigravity, ...],
    features: tuple[DatasetRoutingFeaturesAntigravity, ...],
    decisions: tuple[FoldDecisionAntigravity, ...],
) -> tuple[tuple[DatasetRoutingAuthorityAntigravity, DatasetRoutingFeaturesAntigravity, FoldDecisionAntigravity], ...]:
    """Join frozen target evidence by dataset, retaining the public feature order."""
    if len(targets) != 8 or len(features) != 8 or len(decisions) != 8:
        raise ValueError("dataset-key inputs must contain exactly eight rows")
    target_by_dataset = _unique_dataset_map_antigravity(targets, "target")
    feature_by_dataset = _unique_dataset_map_antigravity(features, "feature")
    decision_by_dataset = _unique_dataset_map_antigravity(decisions, "decision")
    if set(target_by_dataset) != set(feature_by_dataset) or set(target_by_dataset) != set(decision_by_dataset):
        raise ValueError("dataset-key sets differ")
    approved = _TARGET_METHODS_ANTIGRAVITY
    aligned = []
    for feature in features:
        target = target_by_dataset[feature.dataset]
        decision = decision_by_dataset[feature.dataset]
        if (
            target.dataset != feature.dataset or feature.dataset != decision.dataset
            or type(target.family) is not str or type(feature.family) is not str or type(decision.family) is not str
            or type(target.method) is not str or type(decision.method) is not str or type(decision.selected_route) is not str
            or target.family != feature.family or feature.family != decision.family
            or target.method != approved.get(target.family) or decision.method != target.method
            or type(target.target_length) is not int or target.target_length < 0
            or type(feature.target_length) is not int or feature.target_length < 0
            or feature.target_length != target.target_length
            or decision.selected_route not in {"baseline", "candidate"}
        ):
            raise ValueError("dataset-key triple is invalid")
        aligned.append((target, feature, decision))
    return tuple(aligned)


def _validate_inputs_antigravity(
    authority: object, features: object, decisions: object,
) -> tuple[tuple[DatasetRoutingAuthorityAntigravity, ...], tuple[DatasetRoutingFeaturesAntigravity, ...], dict[str, FoldDecisionAntigravity], tuple[FoldDecisionAntigravity, ...], tuple[tuple[str, str], ...]]:
    if type(authority) is not FrozenRoutingAuthorityAntigravity or authority.label_query_count != 0:
        raise ValueError("routing authority is invalid")
    inventory = _validate_exp454_inventory_antigravity(authority)
    rows = authority.rows
    if type(rows) is not tuple or len(rows) != 48 or any(type(row) is not DatasetRoutingAuthorityAntigravity for row in rows):
        raise ValueError("authority must contain exactly 48 rows")
    datasets = tuple(row.dataset for row in rows)
    if any(type(dataset) is not str or not dataset for dataset in datasets) or len(set(datasets)) != 48:
        raise ValueError("authority datasets are invalid")
    targets = []
    for row in rows:
        _valid_confusion_antigravity(row.baseline_confusion)
        if type(row.family) is not str:
            raise ValueError("authority family is invalid")
        expected = _TARGET_METHODS_ANTIGRAVITY.get(row.family)
        if expected is None:
            if row.method is not None or row.candidate_confusion is not None:
                raise ValueError("baseline authority row is invalid")
            continue
        if type(row.method) is not str or row.method != expected or row.candidate_confusion is None or type(row.target_length) is not int or row.target_length < 0:
            raise ValueError("target authority row is invalid")
        _valid_confusion_antigravity(row.candidate_confusion)
        targets.append(row)
    if len(targets) != 8 or len(rows) - len(targets) != 40 or {family: sum(row.family == family for row in targets) for family in _TARGET_METHODS_ANTIGRAVITY} != {"ECG5000": 5, "ChlorineConcentration": 3}:
        raise ValueError("frozen target population differs")
    if type(features) is not tuple or len(features) != 8 or any(type(row) is not DatasetRoutingFeaturesAntigravity for row in features):
        raise ValueError("features must contain exactly eight rows")
    if type(decisions) is not tuple or len(decisions) != 8 or any(type(row) is not FoldDecisionAntigravity for row in decisions):
        raise ValueError("decisions must contain exactly eight rows")
    aligned = align_targets_by_dataset_antigravity(tuple(targets), features, decisions)
    aligned_features = tuple(feature for _target, feature, _decision in aligned)
    aligned_decisions = tuple(decision for _target, _feature, decision in aligned)
    decision_by_dataset = {decision.dataset: decision for decision in aligned_decisions}
    outcomes = {
        row.dataset: (row.baseline_confusion, row.candidate_confusion)
        for row in targets
    }
    expected_decisions = leave_one_dataset_out_antigravity(aligned_features, outcomes)
    if aligned_decisions != expected_decisions:
        raise ValueError("decisions differ from frozen Task 3 LODO replay")
    return rows, aligned_features, decision_by_dataset, aligned_decisions, inventory


def _comparison_antigravity(name: str, rows: Sequence[ConfusionAntigravity]) -> ComparisonMetricAntigravity:
    confusion = _sum_antigravity(rows)
    return ComparisonMetricAntigravity(name, confusion, float(_f1_antigravity(confusion)))


def evidence_for_evaluation_antigravity(
    evaluation: RoutingEvaluationAntigravity,
) -> tuple[tuple[DatasetRoutingAuthorityAntigravity, ...], tuple[DatasetRoutingFeaturesAntigravity, ...], tuple[tuple[str, str], ...]]:
    """Return the in-memory immutable evidence bound to this freshly evaluated result."""
    evidence = _EVIDENCE_BY_EVALUATION_ID_ANTIGRAVITY.get(id(evaluation))
    if evidence is None:
        raise ValueError("evaluation lacks replay evidence")
    return evidence


def evaluate_routing_antigravity(
    authority: FrozenRoutingAuthorityAntigravity,
    features: tuple[DatasetRoutingFeaturesAntigravity, ...],
    decisions: tuple[FoldDecisionAntigravity, ...],
) -> RoutingEvaluationAntigravity:
    """Compute all four preregistered comparisons over the exact frozen 48 rows."""
    rows, _features, decision_by_dataset, aligned_decisions, inventory = _validate_inputs_antigravity(authority, features, decisions)
    baseline = tuple(row.baseline_confusion for row in rows)
    family_wide = tuple(
        row.candidate_confusion if row.candidate_confusion is not None else row.baseline_confusion
        for row in rows
    )
    oracle = tuple(
        row.candidate_confusion if row.candidate_confusion is not None and _prefer_candidate_antigravity(row.baseline_confusion, row.candidate_confusion) else row.baseline_confusion
        for row in rows
    )
    routed = tuple(
        row.candidate_confusion if row.candidate_confusion is not None and decision_by_dataset[row.dataset].selected_route == "candidate" else row.baseline_confusion
        for row in rows
    )
    comparisons = (
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[0], baseline),
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[1], family_wide),
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[2], oracle),
        _comparison_antigravity(_COMPARISON_NAMES_ANTIGRAVITY[3], routed),
    )
    routed_by_dataset = dict(zip((row.dataset for row in rows), routed, strict=True))
    family_delta = tuple(
        (family, sum(routed_by_dataset[row.dataset].fn - row.baseline_confusion.fn for row in rows if row.family == family))
        for family in _TARGET_METHODS_ANTIGRAVITY
    )
    final = comparisons[-1]
    passed = (
        final.confusion.fn < FROZEN_CONFIG_ANTIGRAVITY.global_fn_must_be_below
        and final.confusion.fp <= FROZEN_CONFIG_ANTIGRAVITY.global_fp_max
        and final.f1 >= FROZEN_CONFIG_ANTIGRAVITY.global_f1_min
        and all(delta <= FROZEN_CONFIG_ANTIGRAVITY.target_family_fn_delta_max for _family, delta in family_delta)
    )
    status = "routing_candidate_qualified_for_external_validation" if passed else "routing_candidate_not_qualified_for_external_validation"
    result = RoutingEvaluationAntigravity(comparisons, family_delta, aligned_decisions, passed, status)
    _EVIDENCE_BY_EVALUATION_ID_ANTIGRAVITY[id(result)] = (rows, _features, inventory)
    return result


__all__ = ["align_targets_by_dataset_antigravity", "evaluate_routing_antigravity", "evidence_for_evaluation_antigravity"]
