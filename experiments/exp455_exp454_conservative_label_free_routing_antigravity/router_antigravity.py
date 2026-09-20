"""Select deterministic family rules from frozen label-free Exp455 features."""

from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

from .contracts_antigravity import (
    ConfusionAntigravity,
    DatasetRoutingFeaturesAntigravity,
    FoldDecisionAntigravity,
    RouteRuleAntigravity,
)


_FEATURE_NAMES_ANTIGRAVITY = (
    "target_length",
    "baseline_count",
    "candidate_count",
    "overlap_count",
    "union_count",
    "added_count",
    "removed_count",
    "overlap_ppm",
)
_FAMILY_METHODS_ANTIGRAVITY = {
    "ECG5000": "hydra_local_gap",
    "ChlorineConcentration": "multirocket_local_gap",
}
_FAMILY_COUNTS_ANTIGRAVITY = {"ECG5000": 5, "ChlorineConcentration": 3}
_OutcomePairAntigravity = tuple[ConfusionAntigravity, ConfusionAntigravity]


def threshold_rules_antigravity(feature_name: str, values: Sequence[int]) -> tuple[RouteRuleAntigravity, ...]:
    """Return the preregistered depth-one integer rules for one allowed feature."""
    if feature_name not in _FEATURE_NAMES_ANTIGRAVITY or any(type(value) is not int for value in values):
        raise ValueError("routing feature is invalid")
    ordered = sorted(set(values))
    rules = [RouteRuleAntigravity(None, None, "never"), RouteRuleAntigravity(None, None, "always")]
    for left, right in zip(ordered, ordered[1:]):
        threshold_twice = left + right
        rules.extend((
            RouteRuleAntigravity(feature_name, threshold_twice, "lte"),
            RouteRuleAntigravity(feature_name, threshold_twice, "gt"),
        ))
    return tuple(rules)


def applies_antigravity(rule: RouteRuleAntigravity, value: int) -> bool:
    """Return whether a preregistered route selects the candidate prediction."""
    if type(rule) is not RouteRuleAntigravity or type(value) is not int:
        raise ValueError("routing rule or value is invalid")
    if rule.candidate_when == "never":
        return False
    if rule.candidate_when == "always":
        return True
    if rule.feature_name not in _FEATURE_NAMES_ANTIGRAVITY or type(rule.threshold_twice) is not int:
        raise ValueError("routing rule is invalid")
    if rule.candidate_when == "lte":
        return 2 * value <= rule.threshold_twice
    if rule.candidate_when == "gt":
        return 2 * value > rule.threshold_twice
    raise ValueError("routing rule is invalid")


def _validated_outcomes_antigravity(
    features: Sequence[DatasetRoutingFeaturesAntigravity], outcomes: Mapping[str, _OutcomePairAntigravity],
) -> dict[str, _OutcomePairAntigravity]:
    if not isinstance(outcomes, Mapping) or any(type(row) is not DatasetRoutingFeaturesAntigravity for row in features):
        raise ValueError("routing inputs are invalid")
    datasets = tuple(row.dataset for row in features)
    if len(set(datasets)) != len(datasets) or set(outcomes) != set(datasets):
        raise ValueError("routing outcomes do not match features")
    validated: dict[str, _OutcomePairAntigravity] = {}
    for dataset in datasets:
        pair = outcomes[dataset]
        if type(pair) is not tuple or len(pair) != 2 or any(type(confusion) is not ConfusionAntigravity for confusion in pair):
            raise ValueError("routing outcome is invalid")
        if any(type(value) is not int or value < 0 for confusion in pair for value in (confusion.tp, confusion.fp, confusion.fn, confusion.tn)):
            raise ValueError("routing outcome is invalid")
        validated[dataset] = pair
    return validated


def _sum_confusion_antigravity(rows: Sequence[ConfusionAntigravity]) -> ConfusionAntigravity:
    return ConfusionAntigravity(
        tp=sum(row.tp for row in rows), fp=sum(row.fp for row in rows),
        fn=sum(row.fn for row in rows), tn=sum(row.tn for row in rows),
    )


def _f1_antigravity(confusion: ConfusionAntigravity) -> Fraction:
    denominator = 2 * confusion.tp + confusion.fp + confusion.fn
    return Fraction(0, 1) if denominator == 0 else Fraction(2 * confusion.tp, denominator)


def _rule_lexical_key_antigravity(rule: RouteRuleAntigravity) -> tuple[int, str, int, int, str]:
    """Sort `None` fields explicitly, rather than relying on mixed-type ordering."""
    return (
        0 if rule.feature_name is None else 1,
        "" if rule.feature_name is None else rule.feature_name,
        0 if rule.threshold_twice is None else 1,
        0 if rule.threshold_twice is None else rule.threshold_twice,
        rule.candidate_when,
    )


def _candidate_count_antigravity(rule: RouteRuleAntigravity, features: Sequence[DatasetRoutingFeaturesAntigravity]) -> int:
    return sum(_selects_candidate_antigravity(rule, row) for row in features)


def _selects_candidate_antigravity(
    rule: RouteRuleAntigravity, row: DatasetRoutingFeaturesAntigravity,
) -> bool:
    value = 0 if rule.feature_name is None else getattr(row, rule.feature_name)
    return applies_antigravity(rule, value)


def fit_family_rule_antigravity(
    training_features: Sequence[DatasetRoutingFeaturesAntigravity], outcomes: Mapping[str, _OutcomePairAntigravity],
) -> RouteRuleAntigravity:
    """Fit one family using only its training rows' frozen confusion dependencies."""
    features = tuple(training_features)
    if not features or len({row.family for row in features}) != 1:
        raise ValueError("training features must contain one family")
    validated_outcomes = _validated_outcomes_antigravity(features, outcomes)
    rules = tuple(
        rule
        for feature_name in _FEATURE_NAMES_ANTIGRAVITY
        for rule in threshold_rules_antigravity(feature_name, tuple(getattr(row, feature_name) for row in features))
    )
    scored = []
    for rule in rules:
        selected = tuple(
            validated_outcomes[row.dataset][1 if _selects_candidate_antigravity(rule, row) else 0]
            for row in features
        )
        confusion = _sum_confusion_antigravity(selected)
        scored.append((
            confusion.fn,
            confusion.fp,
            -_f1_antigravity(confusion),
            _candidate_count_antigravity(rule, features),
            _rule_lexical_key_antigravity(rule),
            rule,
        ))
    return min(scored)[-1]


def leave_one_dataset_out_antigravity(
    features: Sequence[DatasetRoutingFeaturesAntigravity], outcomes: Mapping[str, _OutcomePairAntigravity],
) -> tuple[FoldDecisionAntigravity, ...]:
    """Choose every fold rule from the other datasets in its own frozen family."""
    ordered_features = tuple(features)
    validated_outcomes = _validated_outcomes_antigravity(ordered_features, outcomes)
    counts = {family: sum(row.family == family for row in ordered_features) for family in _FAMILY_COUNTS_ANTIGRAVITY}
    if len(ordered_features) != 8 or counts != _FAMILY_COUNTS_ANTIGRAVITY or any(row.family not in _FAMILY_METHODS_ANTIGRAVITY for row in ordered_features):
        raise ValueError("frozen routing population differs")
    decisions = []
    for held_out in ordered_features:
        training = tuple(row for row in ordered_features if row.family == held_out.family and row.dataset != held_out.dataset)
        training_outcomes = {row.dataset: validated_outcomes[row.dataset] for row in training}
        rule = fit_family_rule_antigravity(training, training_outcomes)
        selected_route = "candidate" if _selects_candidate_antigravity(rule, held_out) else "baseline"
        decisions.append(FoldDecisionAntigravity(
            dataset=held_out.dataset,
            family=held_out.family,
            method=_FAMILY_METHODS_ANTIGRAVITY[held_out.family],
            selected_route=selected_route,
            rule=rule,
        ))
    return tuple(decisions)


__all__ = [
    "applies_antigravity",
    "fit_family_rule_antigravity",
    "leave_one_dataset_out_antigravity",
    "threshold_rules_antigravity",
]
