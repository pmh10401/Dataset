from dataclasses import replace
import unittest

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG


class ConfigTests(unittest.TestCase):
    def test_exact_preregistered_values(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp417")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.model_seeds, (41701, 41702, 41703))
        self.assertEqual(FROZEN_CONFIG.horizon_fractions, (0.02, 0.10))
        self.assertEqual(FROZEN_CONFIG.candidate_alpha, 0.01)
        self.assertEqual(FROZEN_CONFIG.persistence_fraction, 0.05)
        self.assertEqual(FROZEN_CONFIG.dataset_workload_rate, 0.05)
        self.assertEqual(FROZEN_CONFIG.global_workload_rate, 0.01)
        self.assertEqual(
            (
                FROZEN_CONFIG.expected_hard,
                FROZEN_CONFIG.expected_standard,
                FROZEN_CONFIG.expected_priority,
                FROZEN_CONFIG.expected_automatic_normal,
            ),
            (2218, 806, 31, 35),
        )
        self.assertEqual(
            (
                FROZEN_CONFIG.database_sha256,
                FROZEN_CONFIG.exp291_prediction_sha256,
                FROZEN_CONFIG.exp291_manifest_sha256,
                FROZEN_CONFIG.exp292_prediction_sha256,
                FROZEN_CONFIG.exp292_manifest_sha256,
            ),
            (
                "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
                "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
                "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e",
                "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b",
                "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0",
            ),
        )
        self.assertEqual(FROZEN_CONFIG.perturbation_seeds, (41711, 41712, 41713))
        self.assertEqual(
            (FROZEN_CONFIG.min_train_model, FROZEN_CONFIG.min_train_candidate),
            (50, 100),
        )
        self.assertEqual(
            (
                FROZEN_CONFIG.oof_candidate_rate,
                FROZEN_CONFIG.oof_wilson_upper,
                FROZEN_CONFIG.pooled_synthetic_sensitivity,
                FROZEN_CONFIG.per_type_synthetic_sensitivity,
            ),
            (0.01, 0.05, 0.70, 0.50),
        )
        self.assertEqual(
            (
                FROZEN_CONFIG.mps_max_error,
                FROZEN_CONFIG.mps_min_spearman,
                FROZEN_CONFIG.mps_min_speedup,
            ),
            (1e-4, 0.999, 1.1),
        )
        FROZEN_CONFIG.validate()

    def test_mutation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, candidate_alpha=0.02).validate()


if __name__ == "__main__":
    unittest.main()
