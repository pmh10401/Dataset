"""Family-blocked qualification for the Exp476 minimum-ZF guard."""

from __future__ import annotations

from typing import Mapping, Sequence

from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    _fit_row_antigravity,
    _summarise_rows_antigravity,
    family_from_dataset_antigravity,
    iter_family_splits_antigravity,
)
from experiments.exp474_conservative_fallback_routing_antigravity.policy_antigravity import (
    BASELINE_METHOD_ANTIGRAVITY,
)

from .policy_antigravity import MinimumZFGuardKNNPolicyAntigravity


def evaluate_minimum_zf_family_blocked_antigravity(
    records: Sequence[Mapping[str, object]],
    *,
    policy_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("records must not be empty")
    by_dataset = {str(record["dataset"]): record for record in records}
    if len(by_dataset) != len(records):
        raise ValueError("dataset names must be unique")
    kwargs = dict(policy_kwargs or {})
    decisions: list[dict[str, object]] = []
    family_overlap_count = 0
    for held_family, train_names, test_names in iter_family_splits_antigravity(list(by_dataset)):
        train_families = {family_from_dataset_antigravity(name) for name in train_names}
        test_families = {family_from_dataset_antigravity(name) for name in test_names}
        family_overlap_count += len(train_families & test_families)
        policy = MinimumZFGuardKNNPolicyAntigravity(**kwargs).fit(
            _fit_row_antigravity(by_dataset[name]) for name in train_names
        )
        for dataset in test_names:
            record = by_dataset[dataset]
            features = record["features"]
            scores = record["scores"]
            proposal = str(record["proposal"])
            assert isinstance(features, Mapping) and isinstance(scores, Mapping)
            decision = policy.route(features, proposal)  # type: ignore[arg-type]
            selected = scores[decision.method]
            baseline = scores[BASELINE_METHOD_ANTIGRAVITY]
            assert isinstance(selected, Mapping) and isinstance(baseline, Mapping)
            oracle_auc = max(float(value["auc"]) for value in scores.values() if isinstance(value, Mapping))
            decisions.append(
                {
                    "dataset": dataset,
                    "held_family": held_family,
                    "proposal": proposal,
                    "method": decision.method,
                    "accepted": decision.accepted,
                    "confidence": decision.confidence,
                    "support": decision.support,
                    "lower_auc_gain": decision.lower_auc_gain,
                    "median_zf_gain": decision.median_zf_gain,
                    "minimum_zf_gain": decision.minimum_zf_gain,
                    "reason": decision.reason,
                    "auc": float(selected["auc"]),
                    "zf": int(selected["zf"]),
                    "baseline_auc": float(baseline["auc"]),
                    "baseline_zf": int(baseline["zf"]),
                    "oracle_auc": oracle_auc,
                }
            )
    baseline_rows = [
        {"auc": row["baseline_auc"], "zf": row["baseline_zf"], "oracle_auc": row["oracle_auc"]}
        for row in decisions
    ]
    baseline_summary = _summarise_rows_antigravity(baseline_rows)
    policy_summary = _summarise_rows_antigravity(decisions)
    policy_summary["accepted_non_baseline"] = int(sum(bool(row["accepted"]) for row in decisions))
    policy_summary["fallback_count"] = int(sum(row["method"] == BASELINE_METHOD_ANTIGRAVITY for row in decisions))
    return {
        "dataset_count": len(decisions),
        "family_count": len({str(row["held_family"]) for row in decisions}),
        "policy_config": {
            "k": kwargs.get("k", 31),
            "min_support": kwargs.get("min_support", 20),
            "positive_rate_threshold": kwargs.get("positive_rate_threshold", 0.90),
            "lower_quantile": kwargs.get("lower_quantile", 0.20),
            "zf_guard": "minimum_neighbour_gain_gte_zero",
        },
        "baseline": baseline_summary,
        "policy": policy_summary,
        "minimum_zf_rejection_count": int(
            sum(row["reason"] == "minimum_zf_not_positive" for row in decisions)
        ),
        "leakage_checks": {
            "family_overlap_count": family_overlap_count,
            "dataset_name_feature_count": 0,
            "held_out_score_feature_count": 0,
        },
        "decisions": decisions,
    }


__all__ = ["evaluate_minimum_zf_family_blocked_antigravity"]
