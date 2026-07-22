from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.complexity_signature import (
    ComplexitySignatureDeviation,
)
from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.ordinal_mmd import (
    OrdinalPatternDistributionShift,
)
from experiments.exp239_248_diverse_score_sources.recurrence_dynamics import (
    RecurrenceDynamicsDeviation,
)
from experiments.exp239_248_diverse_score_sources.symbolic_motif import (
    SymbolicMotifRarity,
)
from experiments.exp239_248_diverse_score_sources.wavelet_energy import (
    MultiscaleWaveletEnergyDeviation,
)


LIGHTWEIGHT_SOURCES = (
    MultiscaleWaveletEnergyDeviation(),
    OrdinalPatternDistributionShift(),
    SymbolicMotifRarity(),
    ComplexitySignatureDeviation(),
    RecurrenceDynamicsDeviation(),
)


def synthetic_bundle() -> DatasetBundle:
    rng = np.random.default_rng(4)
    time = np.linspace(0, 4 * np.pi, 128)
    train = np.vstack([np.sin(time + shift) + rng.normal(0, 0.03, len(time)) for shift in np.linspace(0, 0.4, 10)])
    test = np.vstack([np.sin(time), np.sign(np.sin(3 * time)), np.zeros_like(time)])
    return DatasetBundle("synthetic", train, test, len(time))


class LightweightSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = synthetic_bundle()
        self.temp = tempfile.TemporaryDirectory()
        self.artifact_dir = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_all_sources_return_aligned_finite_scores(self) -> None:
        for source in LIGHTWEIGHT_SOURCES:
            with self.subTest(source=source.name):
                result = source.score(self.bundle, seed=20260720, artifact_dir=self.artifact_dir)
                self.assertEqual(result.train_oof.shape, (len(self.bundle.train_normal),))
                self.assertEqual(result.test_scores.shape, (len(self.bundle.test_values),))
                self.assertTrue(np.isfinite(result.train_oof).all())
                self.assertTrue(np.isfinite(result.test_scores).all())
                self.assertTrue(result.provenance["cross_fitted"])
                self.assertEqual(result.provenance["self_reference_violations"], 0)

    def test_sources_abstain_below_five_train_rows(self) -> None:
        short = DatasetBundle("short", self.bundle.train_normal[:4], self.bundle.test_values, 128)
        for source in LIGHTWEIGHT_SOURCES:
            with self.subTest(source=source.name):
                result = source.score(short, 20260720, self.artifact_dir)
                self.assertTrue(result.abstained)
                self.assertEqual(result.abstention_reason, "n_train_below_5")


if __name__ == "__main__":
    unittest.main()
