from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.fourier_residual import (
    SmoothFourierNormalResidual,
)
from experiments.exp239_248_diverse_score_sources.state_space import (
    RobustStateSpaceInnovation,
)


def bundle() -> DatasetBundle:
    rng = np.random.default_rng(8)
    time = np.linspace(0, 6 * np.pi, 128)
    train = np.vstack([np.sin(time + shift) + rng.normal(0, 0.02, len(time)) for shift in np.linspace(0, 0.2, 12)])
    shifted = np.sin(time); shifted[64:] += 3.0
    test = np.vstack([np.sin(time), shifted])
    return DatasetBundle("statistical", train, test, len(time))


class StatisticalSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = bundle()
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_state_space_detects_level_innovation(self) -> None:
        result = RobustStateSpaceInnovation().score(self.bundle, 20260720, self.path)
        self.assertGreater(result.test_scores[1], result.test_scores[0])
        self.assertTrue(result.provenance["cross_fitted"])

    def test_fourier_harmonics_are_train_derived(self) -> None:
        result = SmoothFourierNormalResidual().score(self.bundle, 20260720, self.path)
        self.assertFalse(result.provenance["uses_test_frequency_selection"])
        self.assertEqual(result.provenance["retained_harmonics"], 8)
        self.assertEqual(result.train_oof.shape, (12,))
        self.assertEqual(result.test_scores.shape, (2,))

    def test_statistical_sources_abstain_below_ten(self) -> None:
        short = DatasetBundle("short", self.bundle.train_normal[:9], self.bundle.test_values, 128)
        for source in (RobustStateSpaceInnovation(), SmoothFourierNormalResidual()):
            with self.subTest(source=source.name):
                self.assertEqual(source.score(short, 20260720, self.path).abstention_reason, "n_train_below_10")


if __name__ == "__main__":
    unittest.main()
