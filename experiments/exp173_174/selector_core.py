from __future__ import annotations

import hashlib
from collections import defaultdict
from statistics import pstdev

from experiments.exp171_172.policy_core import (
    expected_policy_keys,
    simultaneous_wilson_upper,
)


POOL_PRIORITY = (
    "similarity_pool",
    "coarse_strata_pool",
    "global_balanced_pool",
)
SOURCE_AGREEMENT = "2of3"
FOLD_COUNT = 5


def deterministic_fold(dataset: str, folds: int = FOLD_COUNT) -> int:
    if folds < 2:
        raise ValueError("folds must be at least two")
    digest = hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % folds


def _row_key(row: dict) -> tuple[str, float, str]:
    return (
        str(row["pool_policy"]),
        float(row["alpha"]),
        str(row["source_config"]),
    )


def _validate_grid(
    global_rows: list[dict],
    dataset_rows: list[dict],
    expected_dataset_count: int,
) -> tuple[dict[tuple[str, float, str], dict], dict[tuple[str, float, str], list[dict]]]:
    expected = set(expected_policy_keys())
    actual_global = {_row_key(row) for row in global_rows}
    if len(global_rows) != 60 or actual_global != expected:
        raise ValueError("Exp169 summary must contain the exact 60-row policy grid")

    summaries = {_row_key(row): row for row in global_rows}
    grouped: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    for row in dataset_rows:
        grouped[_row_key(row)].append(row)
    if set(grouped) != expected:
        raise ValueError("Exp169 dataset audit does not cover the exact policy grid")

    for key in expected_policy_keys():
        rows = grouped[key]
        datasets = [str(row["dataset"]) for row in rows]
        if len(rows) != expected_dataset_count or len(set(datasets)) != expected_dataset_count:
            raise ValueError(f"dataset coverage mismatch for {key}")
        summary = summaries[key]
        normal_instances = sum(int(row["n_train"]) for row in rows)
        normal_reviews = sum(int(row["normal_review_count"]) for row in rows)
        blocked = sum(int(row["resolution_blocked"]) for row in rows)
        if int(summary["datasets"]) != expected_dataset_count:
            raise ValueError(f"summary dataset count mismatch for {key}")
        if normal_instances != int(summary["total_normal_instances"]):
            raise ValueError(f"normal instance total mismatch for {key}")
        if normal_reviews != int(summary["total_normal_review_count"]):
            raise ValueError(f"normal review total mismatch for {key}")
        if blocked != int(summary["resolution_blocked_datasets"]):
            raise ValueError(f"resolution blocker total mismatch for {key}")
    return summaries, grouped


def _fold_statistics(rows: list[dict], fold_count: int) -> dict:
    review_counts = [0] * fold_count
    normal_counts = [0] * fold_count
    dataset_counts = [0] * fold_count
    for row in sorted(rows, key=lambda item: str(item["dataset"])):
        fold = deterministic_fold(str(row["dataset"]), folds=fold_count)
        review_counts[fold] += int(row["normal_review_count"])
        normal_counts[fold] += int(row["n_train"])
        dataset_counts[fold] += 1
    if any(count < 1 for count in normal_counts):
        raise ValueError("deterministic folds must all contain TRAIN-normal instances")
    rates = [reviews / normals for reviews, normals in zip(review_counts, normal_counts)]
    return {
        "fold_count": fold_count,
        "fold_dataset_counts": dataset_counts,
        "fold_normal_counts": normal_counts,
        "fold_review_counts": review_counts,
        "fold_review_rates": rates,
        "fold_rate_pstdev": float(pstdev(rates)),
        "fold_rate_max": float(max(rates)),
    }


def select_balanced_train_policy(
    global_rows: list[dict],
    dataset_rows: list[dict],
    *,
    risk_target: float = 0.01,
    confidence: float = 0.95,
    expected_dataset_count: int = 316,
    fold_count: int = FOLD_COUNT,
) -> dict:
    if not 0.0 < risk_target < 1.0:
        raise ValueError("risk_target must be between zero and one")
    if fold_count != FOLD_COUNT:
        raise ValueError(f"fold_count must remain frozen at {FOLD_COUNT}")

    summaries, grouped = _validate_grid(
        global_rows,
        dataset_rows,
        expected_dataset_count,
    )
    comparisons = len(expected_policy_keys())
    candidates = []
    for key in expected_policy_keys():
        summary = summaries[key]
        successes = int(summary["total_normal_review_count"])
        trials = int(summary["total_normal_instances"])
        blocked = int(summary["resolution_blocked_datasets"])
        upper = simultaneous_wilson_upper(
            successes,
            trials,
            comparisons=comparisons,
            confidence=confidence,
        )
        risk_eligible = blocked == 0 and upper <= risk_target
        candidates.append(
            {
                "pool_policy": key[0],
                "alpha": key[1],
                "source_config": key[2],
                "datasets": int(summary["datasets"]),
                "total_normal_instances": trials,
                "total_normal_review_count": successes,
                "normal_review_rate_micro": successes / trials,
                "simultaneous_wilson_upper": upper,
                "risk_target": risk_target,
                "confidence": confidence,
                "comparisons": comparisons,
                "resolution_blocked_datasets": blocked,
                "mean_reference_size": float(summary["mean_reference_size"]),
                "risk_eligible": int(risk_eligible),
                "source_eligible": int(key[2] == SOURCE_AGREEMENT),
                "eligibility_reason": (
                    "eligible"
                    if risk_eligible
                    else "resolution_blocked"
                    if blocked
                    else "risk_upper_exceeds_target"
                ),
                **_fold_statistics(grouped[key], fold_count),
                "selection_input_scope": "train_oob_only",
                "test_metrics_used": 0,
                "selection_rank": None,
                "selected": 0,
            }
        )

    eligible_two_of_three = [
        row
        for row in candidates
        if row["risk_eligible"] and row["source_eligible"]
    ]
    largest_alpha = (
        max(float(row["alpha"]) for row in eligible_two_of_three)
        if eligible_two_of_three
        else None
    )
    ranked = []
    if largest_alpha is not None:
        ranked = sorted(
            [row for row in eligible_two_of_three if float(row["alpha"]) == largest_alpha],
            key=lambda row: (
                row["fold_rate_pstdev"],
                row["fold_rate_max"],
                -row["mean_reference_size"],
                POOL_PRIORITY.index(row["pool_policy"]),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["selection_rank"] = rank
            row["selected"] = int(rank == 1)

    selected = None
    if ranked:
        winner = ranked[0]
        selected = {
            **winner,
            "policy_key": [
                winner["pool_policy"],
                winner["alpha"],
                winner["source_config"],
            ],
            "selection_rule": "largest_eligible_alpha_then_min_five_fold_normal_rate_pstdev",
            "candidate_lane": "research_review_candidate",
            "autonomous_hard_alert": False,
        }

    return {
        "status": "policy_selected" if selected else "policy_abstention",
        "risk_target": risk_target,
        "confidence": confidence,
        "comparisons": comparisons,
        "multiple_comparison_correction": "bonferroni",
        "source_agreement": SOURCE_AGREEMENT,
        "fold_count": fold_count,
        "largest_eligible_alpha": largest_alpha,
        "selection_input_scope": "train_oob_only",
        "test_metrics_used": 0,
        "selected": selected,
        "candidates": candidates,
    }
