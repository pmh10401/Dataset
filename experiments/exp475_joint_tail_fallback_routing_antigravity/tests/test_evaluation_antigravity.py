from __future__ import annotations

from experiments.exp475_joint_tail_fallback_routing_antigravity.evaluation_antigravity import (
    evaluate_joint_tail_family_blocked_antigravity,
)


def _record_antigravity(dataset: str, value: float, zf: int) -> dict[str, object]:
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
            "hydra_local_gap": {"auc": 0.95, "zf": zf},
        },
    }


def test_family_blocked_evaluation_applies_zf_tail_guard_antigravity() -> None:
    records = [
        _record_antigravity("Alpha_normal_1", 0.0, 0),
        _record_antigravity("Alpha_normal_2", 1.0, 2),
        _record_antigravity("Beta_normal_1", 2.0, 0),
        _record_antigravity("Beta_normal_2", 3.0, 2),
    ]

    result = evaluate_joint_tail_family_blocked_antigravity(
        records,
        policy_kwargs={
            "k": 2,
            "min_support": 2,
            "positive_rate_threshold": 0.5,
            "lower_quantile": 0.25,
            "min_lower_zf_gain": 0.0,
        },
    )

    assert result["dataset_count"] == 4
    assert result["family_count"] == 2
    assert result["policy"]["accepted_non_baseline"] == 0
    assert result["policy"]["average_auc"] == result["baseline"]["average_auc"]
    assert result["policy"]["average_zf_tp"] == result["baseline"]["average_zf_tp"]
    assert result["leakage_checks"]["family_overlap_count"] == 0
