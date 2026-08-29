from __future__ import annotations

from experiments.exp476_minimum_zf_fallback_routing_antigravity.evaluation_antigravity import (
    evaluate_minimum_zf_family_blocked_antigravity,
)


def _record_antigravity(dataset: str, value: float, hydra_zf: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "proposal": "hydra_local_gap",
        "features": {
            "width": 100.0 + value,
            "n_train": 50.0,
            "spectral_entropy": 0.5,
            "skewness": value / 100.0,
            "kurtosis": 0.0,
            "diff_std_ratio": 0.2,
            "zero_crossing_rate": 0.1,
        },
        "scores": {
            "minirocket_local_gap": {"auc": 0.90, "zf": 1},
            "hydra_local_gap": {"auc": 0.95, "zf": hydra_zf},
        },
    }


def test_family_blocked_evaluation_rejects_one_negative_neighbour_antigravity() -> None:
    zfs = [0, 1, 2, 2, 3]
    records = [
        _record_antigravity(f"Alpha_normal_{index}", float(index), zf)
        for index, zf in enumerate(zfs)
    ] + [
        _record_antigravity(f"Beta_normal_{index}", float(index + 10), zf)
        for index, zf in enumerate(zfs)
    ]

    result = evaluate_minimum_zf_family_blocked_antigravity(
        records,
        policy_kwargs={
            "k": 5,
            "min_support": 5,
            "positive_rate_threshold": 0.8,
            "lower_quantile": 0.20,
        },
    )

    assert result["dataset_count"] == 10
    assert result["policy"]["accepted_non_baseline"] == 0
    assert result["minimum_zf_rejection_count"] == 10
    assert result["policy"]["average_auc"] == result["baseline"]["average_auc"]
    assert result["leakage_checks"]["family_overlap_count"] == 0
