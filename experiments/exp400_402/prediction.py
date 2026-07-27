from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.exp400_402.contracts import (
    ALPHAS,
    CONTROL_VARIANTS,
    VISIBLE_VARIANTS,
    COLLECTION_MODE,
    POLICY_VERSION,
    SCHEMA_VERSION,
    prediction_sha256,
    validate_shadow_row,
)


ALL_VARIANTS = VISIBLE_VARIANTS + CONTROL_VARIANTS


def _indices(row: Mapping[str, object], field: str) -> set[int]:
    value = row.get(field, [])
    if type(value) is not list or any(type(index) is not int for index in value):
        raise ValueError(f"{field} must be an integer list")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} contains duplicate index")
    return set(value)


def _tier_by_index(row: Mapping[str, object]) -> dict[int, int]:
    tiers: dict[int, int] = {}
    for tier in (1, 2, 3):
        for index in _indices(row, f"review_priority_{tier}_indices"):
            if index in tiers:
                raise ValueError("Exp292 review tier overlap")
            tiers[index] = tier
    standard = _indices(row, "standard_review_indices")
    if set(tiers) != standard:
        raise ValueError("Exp292 review tiers do not cover Standard candidates")
    return tiers


def _ranking_index(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int, str, float], Mapping[str, object]]:
    result: dict[tuple[str, int, str, float], Mapping[str, object]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["candidate_index"]),
            str(row["variant"]),
            float(row["alpha"]),
        )
        if key in result:
            raise ValueError(f"duplicate Exp397 ranking row: {key}")
        result[key] = row
    return result


def _validate_exp397_coverage(
    index: Mapping[tuple[str, int, str, float], Mapping[str, object]],
    candidate_keys: set[tuple[str, int]],
) -> None:
    actual_keys = {(dataset, candidate) for dataset, candidate, _, _ in index}
    if actual_keys != candidate_keys:
        raise ValueError("Exp397 candidate universe differs from Exp292")
    expected = {
        (dataset, candidate, variant, float(alpha))
        for dataset, candidate in candidate_keys
        for variant in ALL_VARIANTS
        for alpha in ALPHAS
    }
    if set(index) != expected:
        missing = len(expected - set(index))
        extra = len(set(index) - expected)
        raise ValueError(
            f"Exp397 variant/alpha coverage mismatch: missing={missing} extra={extra}"
        )


def _run_id(dataset: str) -> str:
    return "replay-exp397-" + hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:20]


def build_shadow_predictions(
    exp292_rows: Sequence[Mapping[str, object]],
    exp397_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    exp292 = {str(row["dataset"]): row for row in exp292_rows}
    if len(exp292) != len(exp292_rows):
        raise ValueError("Exp292 contains duplicate datasets")

    tiers_by_dataset = {
        dataset: _tier_by_index(row)
        for dataset, row in exp292.items()
    }
    candidate_keys = {
        (dataset, index)
        for dataset, tiers in tiers_by_dataset.items()
        for index in tiers
    }
    ranking = _ranking_index(exp397_rows)
    _validate_exp397_coverage(ranking, candidate_keys)

    output: list[dict[str, object]] = []
    for dataset in sorted(exp292):
        baseline = exp292[dataset]
        n_test = int(baseline["n_test"])
        hard = _indices(baseline, "automatic_hard_indices")
        priority = _indices(baseline, "priority_review_indices")
        standard = _indices(baseline, "standard_review_indices")
        frozen_lanes = (hard, priority, standard)
        if any(
            frozen_lanes[left] & frozen_lanes[right]
            for left in range(len(frozen_lanes))
            for right in range(left + 1, len(frozen_lanes))
        ):
            raise ValueError(f"Exp292 lane overlap: {dataset}")
        if any(index < 0 or index >= n_test for lane in frozen_lanes for index in lane):
            raise ValueError(f"Exp292 candidate outside range: {dataset}")

        run_id = _run_id(dataset)
        candidates: list[dict[str, object]] = []
        for index in sorted(standard):
            baseline_rows = [
                ranking[(dataset, index, "exp292_baseline", float(alpha))]
                for alpha in ALPHAS
            ]
            baseline_ranks = {int(row["rank"]) for row in baseline_rows}
            if len(baseline_ranks) != 1:
                raise ValueError("Exp292 baseline rank changes across alpha")
            tier_rows = {
                str(alpha): ranking[
                    (dataset, index, "tier_preserving_grammar", float(alpha))
                ]
                for alpha in ALPHAS
            }
            tier = tiers_by_dataset[dataset][index]
            if any(int(row["exp292_tier"]) != tier for row in tier_rows.values()):
                raise ValueError("Exp397 tier differs from Exp292")
            if any(int(row["n_test"]) != n_test for row in tier_rows.values()):
                raise ValueError("Exp397 n_test differs from Exp292")
            p_value_rows = [dict(row["p_values"]) for row in tier_rows.values()]
            if any(values != p_value_rows[0] for values in p_value_rows[1:]):
                raise ValueError("Exp397 grammar p-values change across alpha")
            candidate = {
                "candidate_id": f"{run_id}:{index}",
                "candidate_index": index,
                "exp292_tier": tier,
                "exp292_rank": next(iter(baseline_ranks)),
                "tier_preserving_rank": {
                    alpha: int(row["rank"])
                    for alpha, row in tier_rows.items()
                },
                "control_rank": {
                    variant: {
                        str(alpha): int(
                            ranking[(dataset, index, variant, float(alpha))]["rank"]
                        )
                        for alpha in ALPHAS
                    }
                    for variant in CONTROL_VARIANTS
                },
                "grammar_support_count": {
                    alpha: int(row["grammar_support_count"])
                    for alpha, row in tier_rows.items()
                },
                "p_values": p_value_rows[0],
            }
            candidates.append(candidate)

        unsigned: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "collection_mode": COLLECTION_MODE,
            "policy_version": POLICY_VERSION,
            "run_id": run_id,
            "dataset": dataset,
            "n_test": n_test,
            "automatic_hard_count": len(hard),
            "priority_review_count": len(priority),
            "standard_review_count": len(standard),
            "candidates": candidates,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "automatic_policy_changed": False,
            "retrospective": True,
            "prospective": False,
        }
        row = {**unsigned, "prediction_sha256": prediction_sha256(unsigned)}
        output.append(validate_shadow_row(row))
    return output
