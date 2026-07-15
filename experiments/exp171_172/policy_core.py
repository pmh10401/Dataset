from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

from experiments.exp169_170.pool_core import ALPHAS, POLICIES, SOURCE_CONFIGS

SOURCE_PRIORITY = (
    "2of3",
    "spectral+shapelet",
    "spectral+interval",
    "shapelet+interval",
    "3of3",
)
POOL_PRIORITY = (
    "similarity_pool",
    "coarse_strata_pool",
    "global_balanced_pool",
)


def expected_policy_keys() -> list[tuple[str, float, str]]:
    return [
        (policy, float(alpha), source_config)
        for policy in POLICIES
        for alpha in ALPHAS
        for source_config in SOURCE_CONFIGS
    ]


def simultaneous_wilson_upper(
    successes: int,
    trials: int,
    comparisons: int = 60,
    confidence: float = 0.95,
) -> float:
    if trials < 1 or comparisons < 1:
        raise ValueError("trials and comparisons must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / comparisons)
    p_hat = successes / trials
    denominator = 1.0 + z * z / trials
    numerator = (
        p_hat
        + z * z / (2.0 * trials)
        + z
        * math.sqrt(
            (p_hat * (1.0 - p_hat) + z * z / (4.0 * trials)) / trials
        )
    )
    return numerator / denominator


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
) -> dict[tuple[str, float, str], dict]:
    expected = set(expected_policy_keys())
    if len(global_rows) != 60 or {_row_key(row) for row in global_rows} != expected:
        raise ValueError("Exp169 summary must contain the exact 60-row policy grid")
    by_key = {_row_key(row): row for row in global_rows}
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
        summary = by_key[key]
        if int(summary["datasets"]) != expected_dataset_count:
            raise ValueError(f"summary dataset count mismatch for {key}")
        normal_instances = sum(int(row["n_train"]) for row in rows)
        normal_reviews = sum(int(row["normal_review_count"]) for row in rows)
        blocked = sum(int(row["resolution_blocked"]) for row in rows)
        if normal_instances != int(summary["total_normal_instances"]):
            raise ValueError(f"normal instance total mismatch for {key}")
        if normal_reviews != int(summary["total_normal_review_count"]):
            raise ValueError(f"normal review total mismatch for {key}")
        if blocked != int(summary["resolution_blocked_datasets"]):
            raise ValueError(f"resolution blocker total mismatch for {key}")
    return by_key


def select_train_risk_policy(
    global_rows: list[dict],
    dataset_rows: list[dict],
    *,
    risk_target: float = 0.01,
    confidence: float = 0.95,
    expected_dataset_count: int = 316,
) -> dict:
    if not 0.0 < risk_target < 1.0:
        raise ValueError("risk_target must be between zero and one")
    by_key = _validate_grid(global_rows, dataset_rows, expected_dataset_count)
    comparisons = len(expected_policy_keys())
    candidates = []
    for key in expected_policy_keys():
        row = by_key[key]
        successes = int(row["total_normal_review_count"])
        trials = int(row["total_normal_instances"])
        blocked = int(row["resolution_blocked_datasets"])
        upper = simultaneous_wilson_upper(
            successes,
            trials,
            comparisons=comparisons,
            confidence=confidence,
        )
        eligible = blocked == 0 and upper <= risk_target
        candidates.append(
            {
                "pool_policy": key[0],
                "alpha": key[1],
                "source_config": key[2],
                "datasets": int(row["datasets"]),
                "total_normal_instances": trials,
                "total_normal_review_count": successes,
                "normal_review_rate_micro": successes / trials,
                "simultaneous_wilson_upper": upper,
                "risk_target": risk_target,
                "confidence": confidence,
                "comparisons": comparisons,
                "resolution_blocked_datasets": blocked,
                "eligible": int(eligible),
                "eligibility_reason": (
                    "eligible"
                    if eligible
                    else "resolution_blocked"
                    if blocked
                    else "risk_upper_exceeds_target"
                ),
                "selection_input_scope": "train_oob_only",
                "test_metrics_used": 0,
            }
        )

    eligible_rows = [row for row in candidates if row["eligible"]]
    ranked = sorted(
        eligible_rows,
        key=lambda row: (
            -row["total_normal_review_count"],
            -row["alpha"],
            SOURCE_PRIORITY.index(row["source_config"]),
            POOL_PRIORITY.index(row["pool_policy"]),
        ),
    )
    ranks = {
        (row["pool_policy"], row["alpha"], row["source_config"]): rank
        for rank, row in enumerate(ranked, start=1)
    }
    for row in candidates:
        row["selection_rank"] = ranks.get(
            (row["pool_policy"], row["alpha"], row["source_config"])
        )
        row["selected"] = int(row["selection_rank"] == 1)

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
            "selection_rule": "max_train_oob_review_count_under_simultaneous_risk_cap",
            "candidate_lane": "research_review_candidate",
            "autonomous_hard_alert": False,
        }
    return {
        "status": "policy_selected" if selected else "policy_abstention",
        "risk_target": risk_target,
        "confidence": confidence,
        "comparisons": comparisons,
        "multiple_comparison_correction": "bonferroni",
        "selection_input_scope": "train_oob_only",
        "test_metrics_used": 0,
        "selected": selected,
        "candidates": candidates,
    }
