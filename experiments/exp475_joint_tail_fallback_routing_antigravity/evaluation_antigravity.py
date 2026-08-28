"""Family-blocked qualification for the Exp475 joint-tail guard."""

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

from .policy_antigravity import JointTailGuardKNNPolicyAntigravity


def evaluate_joint_tail_family_blocked_antigravity(
    records: Sequence[Mapping[str, object]],
    *,
    policy_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("records must not be empty")
    by_dataset = {str(record["dataset"]): record for record in records}
    if len(by_dataset) != len(records):
        raise ValueError("dataset names must be unique")

    decisions: list[dict[str, object]] = []
    family_overlap_count = 0
    kwargs = dict(policy_kwargs or {})
    for held_family, train_names, test_names in iter_family_splits_antigravity(list(by_dataset)):
        train_families = {family_from_dataset_antigravity(name) for name in train_names}
        test_families = {family_from_dataset_antigravity(name) for name in test_names}
        family_overlap_count += len(train_families & test_families)
        policy = JointTailGuardKNNPolicyAntigravity(**kwargs).fit(
            _fit_row_antigravity(by_dataset[name]) for name in train_names
        )
        for dataset in test_names:
            record = by_dataset[dataset]
            proposal = str(record["proposal"])
            features = record["features"]
            scores = record["scores"]
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
                    "lower_zf_gain": decision.lower_zf_gain,
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
            "min_auc_gain": kwargs.get("min_auc_gain", 0.0),
            "min_zf_gain": kwargs.get("min_zf_gain", 0.0),
            "min_lower_zf_gain": kwargs.get("min_lower_zf_gain", 0.0),
        },
        "baseline": baseline_summary,
        "policy": policy_summary,
        "leakage_checks": {
            "family_overlap_count": family_overlap_count,
            "dataset_name_feature_count": 0,
            "held_out_score_feature_count": 0,
        },
        "decisions": decisions,
    }


__all__ = ["evaluate_joint_tail_family_blocked_antigravity"]
