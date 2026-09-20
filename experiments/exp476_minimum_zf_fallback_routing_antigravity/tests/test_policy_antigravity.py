from __future__ import annotations

from experiments.exp476_minimum_zf_fallback_routing_antigravity.policy_antigravity import (
    MinimumZFGuardKNNPolicyAntigravity,
)


def _features_antigravity(value: float) -> dict[str, float]:
    return {
        "width": 100.0 + value,
        "n_train": 80.0,
        "spectral_entropy": 0.5,
        "skewness": value / 100.0,
        "kurtosis": 0.0,
        "diff_std_ratio": 0.2,
        "zero_crossing_rate": 0.1,
    }


def _rows_antigravity(zf_gains: list[float]) -> list[dict[str, object]]:
    return [
        {
            "features": _features_antigravity(float(index)),
            "proposal": "hydra_local_gap",
            "auc_gain": 0.03,
            "zf_gain": gain,
        }
        for index, gain in enumerate(zf_gains)
    ]


def test_one_negative_zf_neighbour_forces_fallback_antigravity() -> None:
    policy = MinimumZFGuardKNNPolicyAntigravity(
        k=5,
        min_support=5,
        positive_rate_threshold=0.8,
        lower_quantile=0.20,
    ).fit(_rows_antigravity([-1.0, 0.0, 1.0, 1.0, 2.0]))

    decision = policy.route(_features_antigravity(2.0), "hydra_local_gap")

    assert decision.method == "minirocket_local_gap"
    assert not decision.accepted
    assert decision.reason == "minimum_zf_not_positive"
    assert decision.minimum_zf_gain == -1.0


def test_all_nonnegative_zf_neighbours_keep_proposal_antigravity() -> None:
    policy = MinimumZFGuardKNNPolicyAntigravity(
        k=5,
        min_support=5,
        positive_rate_threshold=0.8,
        lower_quantile=0.20,
    ).fit(_rows_antigravity([0.0, 0.0, 1.0, 1.0, 2.0]))

    decision = policy.route(_features_antigravity(2.0), "hydra_local_gap")

    assert decision.method == "hydra_local_gap"
    assert decision.accepted
    assert decision.minimum_zf_gain == 0.0
