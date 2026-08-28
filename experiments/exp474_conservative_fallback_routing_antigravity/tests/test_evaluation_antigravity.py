from __future__ import annotations

from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    evaluate_family_blocked_antigravity,
    family_from_dataset_antigravity,
    iter_family_splits_antigravity,
)


def _features_antigravity(value: float) -> dict[str, float]:
    return {
        "width": 100.0 + value,
        "n_train": 50.0,
        "spectral_entropy": 0.5,
        "skewness": value / 100.0,
        "kurtosis": 0.0,
        "diff_std_ratio": 0.2,
        "zero_crossing_rate": 0.1,
    }


def _record_antigravity(dataset: str, value: float) -> dict[str, object]:
    return {
        "dataset": dataset,
        "proposal": "hydra_local_gap",
        "features": _features_antigravity(value),
        "scores": {
            "minirocket_local_gap": {"auc": 0.90, "zf": 1},
            "hydra_local_gap": {"auc": 0.95, "zf": 2},
        },
    }


def test_family_name_uses_only_evaluation_group_suffix_antigravity() -> None:
    assert family_from_dataset_antigravity("ECG5000_normal_3") == "ECG5000"
    assert family_from_dataset_antigravity("Wafer_normal_-1") == "Wafer"


def test_family_splits_never_mix_held_family_antigravity() -> None:
    datasets = ["Alpha_normal_1", "Alpha_normal_2", "Beta_normal_1"]

    splits = list(iter_family_splits_antigravity(datasets))

    assert len(splits) == 2
    for held_family, train_names, test_names in splits:
        assert test_names
        assert all(family_from_dataset_antigravity(name) == held_family for name in test_names)
        assert all(family_from_dataset_antigravity(name) != held_family for name in train_names)


def test_family_blocked_evaluation_reports_baseline_and_policy_antigravity() -> None:
    records = [
        _record_antigravity("Alpha_normal_1", 0.0),
        _record_antigravity("Alpha_normal_2", 1.0),
        _record_antigravity("Beta_normal_1", 2.0),
        _record_antigravity("Beta_normal_2", 3.0),
    ]

    result = evaluate_family_blocked_antigravity(
        records,
        policy_kwargs={
            "k": 2,
            "min_support": 2,
            "positive_rate_threshold": 1.0,
            "lower_quantile": 0.25,
        },
    )

    assert result["dataset_count"] == 4
    assert result["family_count"] == 2
    assert result["baseline"]["average_auc"] == 0.90
    assert result["policy"]["average_auc"] == 0.95
    assert result["policy"]["accepted_non_baseline"] == 4
    assert result["leakage_checks"]["family_overlap_count"] == 0
