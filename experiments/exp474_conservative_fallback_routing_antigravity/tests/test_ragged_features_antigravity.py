from __future__ import annotations

import numpy as np

from experiments.exp474_conservative_fallback_routing_antigravity.ragged_features_antigravity import (
    extract_ragged_routing_features_antigravity,
)
from experiments.exp_routing_label_free_antigravity.feature_extractor_antigravity import (
    extract_routing_features_antigravity,
)


def test_fixed_width_rows_match_frozen_extractor_antigravity() -> None:
    rng = np.random.default_rng(20260827)
    matrix = rng.normal(size=(7, 64))

    expected = extract_routing_features_antigravity(matrix)
    actual = extract_ragged_routing_features_antigravity([row.copy() for row in matrix])

    for key in (
        "width",
        "n_train",
        "spectral_entropy",
        "skewness",
        "kurtosis",
        "diff_std_ratio",
        "zero_crossing_rate",
    ):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1e-12)


def test_variable_width_rows_use_median_width_without_resampling_antigravity() -> None:
    rows = [
        np.linspace(0.0, 1.0, 5),
        np.linspace(0.0, 1.0, 9),
        np.linspace(0.0, 1.0, 7),
    ]

    features = extract_ragged_routing_features_antigravity(rows)

    assert features["width"] == 7.0
    assert features["n_train"] == 3.0
    assert all(np.isfinite(value) for value in features.values())


def test_empty_or_too_short_rows_are_rejected_antigravity() -> None:
    for rows in ([], [np.arange(5.0)], [np.arange(5.0), np.arange(1.0)]):
        try:
            extract_ragged_routing_features_antigravity(rows)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rows must be rejected")
