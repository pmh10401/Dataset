from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.masked_patch import MaskedPatchContextError
from experiments.exp239_248_diverse_score_sources.normal_density import NonlinearNormalDensity
from experiments.exp239_248_diverse_score_sources.transformation_consistency import (
    TransformationConsistency,
)


def bundle(n_train: int = 20) -> DatasetBundle:
    rng = np.random.default_rng(3)
    time = np.linspace(0, 4 * np.pi, 64)
    train = np.vstack([np.sin(time + shift) + rng.normal(0, 0.02, len(time)) for shift in np.linspace(0, 0.3, n_train)])
    test = np.vstack([np.sin(time), np.sign(np.sin(3 * time))])
    return DatasetBundle("model", train.astype(np.float32), test.astype(np.float32), len(time))


class ModelSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_model_source_abstains_below_minimum_train(self) -> None:
        short = bundle(19)
        result = TransformationConsistency(epochs=1, seeds=(7,)).score(short, 7, self.path)
        self.assertTrue(result.abstained)
        self.assertEqual(result.abstention_reason, "n_train_below_20")

    def test_masked_patch_seed_is_deterministic(self) -> None:
        first = MaskedPatchContextError(epochs=1, seeds=(7,)).score(bundle(), 7, self.path / "a")
        second = MaskedPatchContextError(epochs=1, seeds=(7,)).score(bundle(), 7, self.path / "b")
        np.testing.assert_allclose(first.test_scores, second.test_scores, rtol=1e-5, atol=1e-6)
        self.assertEqual(first.artifacts["test_embeddings"].shape, (2, 32))

    def test_density_never_fits_test_embedding(self) -> None:
        rng = np.random.default_rng(9)
        train = rng.normal(size=(50, 32)).astype(np.float32)
        test = rng.normal(size=(3, 32)).astype(np.float32)
        source = NonlinearNormalDensity(epochs=1, seeds=(7,))
        result = source.score_from_embeddings(train, test, seed=7)
        self.assertFalse(result.provenance["test_used_for_fit"])
        self.assertEqual(result.test_scores.shape, (3,))
        self.assertTrue(np.isfinite(result.test_scores).all())


if __name__ == "__main__":
    unittest.main()
