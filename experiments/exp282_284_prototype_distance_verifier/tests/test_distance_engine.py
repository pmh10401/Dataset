import unittest

import numpy as np

from experiments.exp282_284_prototype_distance_verifier.distance_engine import (
    DISTANCES, cross_fitted_references, fit_distance_model, fixed_features, score_distances,
)


class DistanceEngineTest(unittest.TestCase):
    def test_features_and_distances_are_finite_and_deterministic(self) -> None:
        rng = np.random.default_rng(7); values = rng.normal(size=(25, 64))
        self.assertEqual(fixed_features(values).shape, (25, 16))
        model = fit_distance_model(values); scores = score_distances(model, values[:3])
        self.assertEqual(set(scores), set(DISTANCES))
        self.assertTrue(all(np.isfinite(value).all() for value in scores.values()))
        first, _ = cross_fitted_references(values); second, _ = cross_fitted_references(values)
        for name in DISTANCES: np.testing.assert_allclose(first[name], second[name])

    def test_constant_series_matches_constant_normal_medoid(self) -> None:
        train = np.vstack([np.ones(32), np.ones(32), np.linspace(0, 1, 32), np.linspace(0, 1, 32)])
        model = fit_distance_model(train)
        scores = score_distances(model, np.ones((1, 32)))
        self.assertEqual(float(scores["correlation"][0]), 0.0)
        self.assertEqual(float(scores["spectral_wasserstein"][0]), 0.0)


if __name__ == "__main__": unittest.main()
