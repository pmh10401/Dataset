from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.exp397_399.artifact_io import (
    load_verified_ranking,
    write_ranking_artifact,
)
from experiments.exp397_399.contracts import ALPHAS, VARIANTS
from experiments.exp397_399.ranking import rank_candidates


def _tier_by_index(row: dict[str, Any]) -> dict[int, int]:
    tiers: dict[int, int] = {}
    for tier in (1, 2, 3):
        for raw_index in row.get(f"review_priority_{tier}_indices", []):
            index = int(raw_index)
            if index in tiers:
                raise ValueError("Exp292 review tiers overlap")
            tiers[index] = tier
    standard = {int(index) for index in row.get("standard_review_indices", [])}
    if set(tiers) != standard:
        raise ValueError("Exp292 review tiers do not partition standard review")
    return tiers


def build_prediction(
    exp292_rows: Sequence[dict[str, Any]],
    exp393_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    exp292 = {str(row["dataset"]): row for row in exp292_rows}
    exp393 = {str(row["dataset"]): row for row in exp393_rows}
    if len(exp292) != len(exp292_rows) or len(exp393) != len(exp393_rows):
        raise ValueError("prediction inputs contain duplicate datasets")
    if set(exp292) != set(exp393):
        raise ValueError("Exp292 and Exp393 dataset coverage differs")

    features: list[dict[str, Any]] = []
    for dataset in sorted(exp292):
        review_row = exp292[dataset]
        grammar_row = exp393[dataset]
        n_test = int(review_row["n_test"])
        if int(grammar_row["n_test"]) != n_test:
            raise ValueError(f"n_test mismatch for {dataset}")
        p_value_arrays = dict(grammar_row.get("p_values", {}))
        tiers = _tier_by_index(review_row)
        for index, tier in sorted(tiers.items()):
            if index < 0 or index >= n_test:
                raise ValueError(f"review index out of range for {dataset}")
            p_values = {}
            for score in ("transition", "dwell", "recovery", "composite"):
                values = list(p_value_arrays.get(score, []))
                p_values[score] = float(values[index]) if len(values) == n_test else 1.0
            features.append(
                {
                    "dataset": dataset,
                    "candidate_index": index,
                    "n_test": n_test,
                    "exp292_tier": tier,
                    "p_values": p_values,
                }
            )

    output: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for variant in VARIANTS:
            ranked = rank_candidates(features, variant, alpha)
            for row in ranked:
                output.append(
                    {
                        **row,
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                        "automatic_hard_changed": False,
                        "priority_review_changed": False,
                        "standard_review_changed": False,
                    }
                )
    return output
