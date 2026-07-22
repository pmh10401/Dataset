from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp249_250_official_mini_multirocket.config import MODEL_CONFIGS
from experiments.exp249_250_official_mini_multirocket.scoring import score_configuration


class _FakeTransformer:
    fit_sizes: list[int] = []
    transform_sizes: list[int] = []

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit_sizes.append(len(values))
        return values[:, 0, :2].astype(float)

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.transform_sizes.append(len(values))
        return values[:, 0, :2].astype(float)


class ScoringContractTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeTransformer.fit_sizes.clear()
        _FakeTransformer.transform_sizes.clear()

    def test_transformer_is_refit_inside_every_crossfit_fold(self) -> None:
        train = np.arange(60, dtype=np.float32).reshape(10, 6)
        test = np.arange(18, dtype=np.float32).reshape(3, 6)
        with patch(
            "experiments.exp249_250_official_mini_multirocket.scoring.make_transformer",
            side_effect=lambda *args, **kwargs: _FakeTransformer(),
        ):
            result = score_configuration(
                train,
                test,
                config=MODEL_CONFIGS[249],
                requested_kernels=1024,
                seed=17,
                expected_dimension=None,
            )
        self.assertEqual(_FakeTransformer.fit_sizes.count(8), 5)
        self.assertEqual(_FakeTransformer.fit_sizes.count(10), 1)
        self.assertEqual(result.train_oof.shape, (10,))
        self.assertEqual(result.test_scores.shape, (3,))
        self.assertTrue(np.isfinite(result.train_oof).all())
        self.assertTrue(np.isfinite(result.test_scores).all())

    def test_expected_dimension_mismatch_is_rejected(self) -> None:
        train = np.arange(48, dtype=np.float32).reshape(8, 6)
        test = np.arange(12, dtype=np.float32).reshape(2, 6)
        with patch(
            "experiments.exp249_250_official_mini_multirocket.scoring.make_transformer",
            side_effect=lambda *args, **kwargs: _FakeTransformer(),
        ):
            with self.assertRaisesRegex(ValueError, "feature dimension mismatch"):
                score_configuration(
                    train,
                    test,
                    config=MODEL_CONFIGS[249],
                    requested_kernels=1024,
                    seed=17,
                    expected_dimension=1008,
                )

    def test_reference_cap_and_query_chunk_bound_memory(self) -> None:
        train = np.arange(120, dtype=np.float32).reshape(20, 6)
        test = np.arange(42, dtype=np.float32).reshape(7, 6)
        with patch(
            "experiments.exp249_250_official_mini_multirocket.scoring.make_transformer",
            side_effect=lambda *args, **kwargs: _FakeTransformer(),
        ):
            result = score_configuration(
                train,
                test,
                config=MODEL_CONFIGS[249],
                requested_kernels=1024,
                seed=17,
                expected_dimension=None,
                max_train_reference=5,
                query_chunk_size=3,
            )
        self.assertTrue(all(size <= 5 for size in _FakeTransformer.fit_sizes))
        self.assertTrue(all(size <= 3 for size in _FakeTransformer.transform_sizes))
        self.assertEqual(result.train_oof.shape, (20,))
        self.assertEqual(result.test_scores.shape, (7,))


if __name__ == "__main__":
    unittest.main()
