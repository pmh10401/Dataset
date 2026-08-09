"""RED/GREEN tests for the frozen Exp443 metric and bootstrap contract."""
from __future__ import annotations

from hashlib import sha256
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity import bootstrap_antigravity


class BootstrapAntigravityTests(unittest.TestCase):
    def test_project_metrics_use_zero_for_every_zero_denominator(self) -> None:
        metrics = bootstrap_antigravity.project_metrics_antigravity((0, 0, 0, 0))
        self.assertEqual(metrics.precision, 0.0)
        self.assertEqual(metrics.recall, 0.0)
        self.assertEqual(metrics.f1, 0.0)
        self.assertEqual(metrics.false_positive_rate, 0.0)

        balanced = bootstrap_antigravity.project_metrics_antigravity((1, 1, 1, 1))
        self.assertEqual(
            (balanced.precision, balanced.recall, balanced.f1, balanced.false_positive_rate),
            (0.5, 0.5, 0.5, 0.5),
        )

    def test_project_metrics_reject_non_integer_negative_and_boolean_counts(self) -> None:
        for confusion in ((1.0, 0, 0, 0), (-1, 0, 0, 0), (True, 0, 0, 0), (1, 2, 3)):
            with self.subTest(confusion=confusion), self.assertRaises(ValueError):
                bootstrap_antigravity.project_metrics_antigravity(confusion)

    def test_tiny_paired_bootstrap_is_deterministic_and_hashes_exact_int64_indices(self) -> None:
        rows = np.asarray(
            [
                [1, 0, 1, 2, 2, 0, 0, 2, 0.4],
                [0, 1, 2, 3, 1, 1, 1, 3, 0.2],
                [2, 1, 0, 1, 2, 0, 0, 2, 0.1],
            ],
            dtype=np.float64,
        )
        first = bootstrap_antigravity._paired_dataset_bootstrap_for_test_antigravity(
            rows, population=3, samples=7, seed=443,
        )
        second = bootstrap_antigravity._paired_dataset_bootstrap_for_test_antigravity(
            rows, population=3, samples=7, seed=443,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.population, 3)
        self.assertEqual(first.samples, 7)
        self.assertEqual(first.sample_dtype, "int64")
        self.assertFalse(first.endpoint)
        self.assertEqual(first.quantile_method, "linear")

        rng = np.random.Generator(np.random.PCG64(443))
        sampled = np.empty((7, 3), dtype=np.int64)
        for repetition in range(7):
            sampled[repetition] = rng.integers(
                low=0, high=3, size=3, dtype=np.int64, endpoint=False,
            )
        self.assertEqual(first.sample_indices_sha256, sha256(sampled.tobytes(order="C")).hexdigest())

    def test_production_bootstrap_rejects_every_contract_override_and_wrong_numpy(self) -> None:
        rows = np.zeros((1117, 9), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "population"):
            bootstrap_antigravity.paired_dataset_bootstrap_antigravity(rows[:1116])
        with self.assertRaisesRegex(ValueError, "samples"):
            bootstrap_antigravity.paired_dataset_bootstrap_antigravity(rows, samples=2)
        with self.assertRaisesRegex(ValueError, "population"):
            bootstrap_antigravity.paired_dataset_bootstrap_antigravity(rows, population=1116)
        with patch.object(bootstrap_antigravity.np, "__version__", "9.9.9"):
            with self.assertRaisesRegex(RuntimeError, "2.3.5"):
                bootstrap_antigravity.paired_dataset_bootstrap_antigravity(rows)

    def test_production_zero_rows_replay_exact_frozen_bootstrap(self) -> None:
        rows = np.zeros((1117, 9), dtype=np.float64)
        result = bootstrap_antigravity.paired_dataset_bootstrap_antigravity(rows)
        self.assertEqual(result.micro_delta_ci, (0.0, 0.0))
        self.assertEqual(result.macro_f1_delta_ci, (0.0, 0.0))
        self.assertEqual((result.seed, result.samples, result.population), (44320260809, 20000, 1117))


if __name__ == "__main__":
    unittest.main()
