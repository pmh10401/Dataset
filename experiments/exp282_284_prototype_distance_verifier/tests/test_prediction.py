import unittest

from experiments.exp282_284_prototype_distance_verifier.prediction import _passes


class PredictionPolicyTest(unittest.TestCase):
    def test_mahalanobis_distances_cast_one_geometry_vote(self) -> None:
        result = _passes({
            "diagonal_mahalanobis": 0.01, "shrinkage_mahalanobis": 0.01,
            "correlation": 0.5, "spectral_wasserstein": 0.5,
        })
        self.assertFalse(result["balanced_consensus_2of3"])
        self.assertFalse(result["sensitivity_any_mahal_2of3"])


if __name__ == "__main__": unittest.main()
