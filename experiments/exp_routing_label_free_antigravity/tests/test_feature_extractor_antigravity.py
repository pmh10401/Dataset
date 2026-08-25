"""Tests for label-free routing feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp_routing_label_free_antigravity.feature_extractor_antigravity import (
    extract_routing_features_antigravity,
)


def test_basic_features_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 140))
    f = extract_routing_features_antigravity(X)
    assert set(f) == {"width", "n_train", "spectral_entropy", "skewness", "kurtosis", "ref_score_var"}
    assert f["width"] == 140 and f["n_train"] == 100
    assert 0.0 <= f["spectral_entropy"] <= 1.0
    assert abs(f["skewness"]) < 1.0
    assert abs(f["kurtosis"]) < 2.0


def test_periodic_signal_has_lower_entropy_than_noise() -> None:
    rng = np.random.default_rng(1)
    t = np.linspace(0, 8 * np.pi, 140)
    periodic = np.tile(np.sin(t), (50, 1))
    noise = rng.standard_normal((50, 140))
    fp = extract_routing_features_antigravity(periodic)["spectral_entropy"]
    fn = extract_routing_features_antigravity(noise)["spectral_entropy"]
    assert fp < fn - 0.3


def test_reference_score_var_is_returned_when_given() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 100))
    scores = rng.standard_normal(30)
    f = extract_routing_features_antigravity(X, reference_scores=scores)
    assert np.isfinite(f["ref_score_var"])
    assert f["ref_score_var"] > 0


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        extract_routing_features_antigravity(np.zeros((1, 10)))
