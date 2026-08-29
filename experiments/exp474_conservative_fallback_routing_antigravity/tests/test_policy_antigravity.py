from __future__ import annotations

from experiments.exp474_conservative_fallback_routing_antigravity.policy_antigravity import (
    ConservativeKNNPolicyAntigravity,
)


def _features_antigravity(value: float) -> dict[str, float]:
    return {
        "width": 100.0 + value,
        "n_train": 80.0 + value,
        "spectral_entropy": 0.5 + value / 1000.0,
        "skewness": value / 100.0,
        "kurtosis": value / 50.0,
        "diff_std_ratio": 0.2 + value / 1000.0,
        "zero_crossing_rate": 0.1 + value / 2000.0,
    }


def _fit_rows_antigravity(*, auc_gain: float, zf_gain: float, count: int = 8) -> list[dict[str, object]]:
    return [
        {
            "features": _features_antigravity(float(index)),
            "proposal": "hydra_local_gap",
            "auc_gain": auc_gain,
            "zf_gain": zf_gain,
        }
        for index in range(count)
    ]


def test_minirocket_proposal_is_always_kept_antigravity() -> None:
    policy = ConservativeKNNPolicyAntigravity(k=5, min_support=5).fit(
        _fit_rows_antigravity(auc_gain=0.1, zf_gain=2.0)
    )

    decision = policy.route(_features_antigravity(2.0), "minirocket_local_gap")

    assert decision.method == "minirocket_local_gap"
    assert not decision.accepted
    assert decision.reason == "baseline_proposal"


def test_strong_neighbour_evidence_accepts_non_baseline_antigravity() -> None:
    policy = ConservativeKNNPolicyAntigravity(
        k=5,
        min_support=5,
        positive_rate_threshold=0.8,
        lower_quantile=0.25,
    ).fit(_fit_rows_antigravity(auc_gain=0.03, zf_gain=1.0))

    decision = policy.route(_features_antigravity(2.5), "hydra_local_gap")

    assert decision.method == "hydra_local_gap"
    assert decision.accepted
    assert decision.support == 5
    assert decision.lower_auc_gain >= 0.03 - 1e-12


def test_low_support_falls_back_to_minirocket_antigravity() -> None:
    policy = ConservativeKNNPolicyAntigravity(k=6, min_support=6).fit(
        _fit_rows_antigravity(auc_gain=0.1, zf_gain=2.0, count=5)
    )

    decision = policy.route(_features_antigravity(2.0), "hydra_local_gap")

    assert decision.method == "minirocket_local_gap"
    assert not decision.accepted
    assert decision.reason == "insufficient_support"


def test_negative_auc_tail_forces_fallback_antigravity() -> None:
    rows = _fit_rows_antigravity(auc_gain=0.03, zf_gain=1.0, count=3)
    rows.extend(
        {
            "features": _features_antigravity(value),
            "proposal": "hydra_local_gap",
            "auc_gain": -0.2,
            "zf_gain": 1.0,
        }
        for value in (2.1, 2.2)
    )
    policy = ConservativeKNNPolicyAntigravity(
        k=5,
        min_support=5,
        positive_rate_threshold=0.5,
        lower_quantile=0.25,
    ).fit(rows)

    decision = policy.route(_features_antigravity(2.0), "hydra_local_gap")

    assert decision.method == "minirocket_local_gap"
    assert not decision.accepted
    assert decision.reason == "auc_tail_not_positive"
