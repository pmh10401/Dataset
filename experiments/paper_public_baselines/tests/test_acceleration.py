from __future__ import annotations

import unittest

import numpy as np

from experiments.paper_public_baselines.acceleration import assess_mlx
from experiments.paper_public_baselines.config import ALPHAS, METHOD_IDS


class AccelerationTests(unittest.TestCase):
    def test_all_methods_have_explicit_cpu_fallback_evidence(self) -> None:
        smoke = np.arange(32, dtype=np.float32).reshape(2, 16)

        for method_id in METHOD_IDS:
            with self.subTest(method_id=method_id):
                record = assess_mlx(method_id, smoke)
                self.assertEqual(record.selected_backend, "cpu")
                self.assertFalse(record.condition_1_eligible)
                self.assertFalse(record.parity_passed)
                self.assertTrue(record.fallback_reason)
                self.assertEqual(
                    set(record.candidate_indices_equal_by_alpha),
                    {format(alpha, ".12g") for alpha in ALPHAS},
                )
                self.assertIn("mlx", record.versions)
                self.assertIn("mlx_metal", record.versions)

    def test_classical_fallback_preserves_official_sklearn_estimators(self) -> None:
        record = assess_mlx("isolation_forest", np.zeros((2, 16)))

        self.assertIn("official scikit-learn", record.fallback_reason)


if __name__ == "__main__":
    unittest.main()
