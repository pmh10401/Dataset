import unittest

from experiments.exp410_family_robustness.config import FROZEN_CONFIG
from experiments.exp410_family_robustness.contracts import family_from_dataset


class ConfigContractTests(unittest.TestCase):
    def test_frozen_contract_matches_preregistered_values(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp410")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.expected_families, 128)
        self.assertEqual(FROZEN_CONFIG.expected_hard, 2218)
        self.assertEqual(
            (FROZEN_CONFIG.expected_tp, FROZEN_CONFIG.expected_fp, FROZEN_CONFIG.expected_fn),
            (1915, 303, 2003),
        )
        self.assertEqual(FROZEN_CONFIG.family_weight_caps, (1, 3, 5, 10, None))
        self.assertEqual(FROZEN_CONFIG.bootstrap_iterations, 2000)
        self.assertEqual(FROZEN_CONFIG.bootstrap_seed, 410)
        self.assertEqual(
            FROZEN_CONFIG.prediction_manifest_sha256,
            "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8",
        )
        self.assertEqual(
            FROZEN_CONFIG.evaluation_input_manifest_sha256,
            "13f303c6422b77e10ad7eb4dbbf7ed3506849f4b972ad85ccb2cfc2bf935f6de",
        )
        self.assertEqual(
            FROZEN_CONFIG.evaluation_publication_manifest_sha256,
            "f0439be9ca751e9849dbd321a8673dafe64d4424b27e6f69282a43c1e9bdef10",
        )
        self.assertEqual(FROZEN_CONFIG.expected_database_snapshot_keys, ("db",))
        self.assertFalse(FROZEN_CONFIG.operational_policy_changed)
        self.assertFalse(FROZEN_CONFIG.deployment_eligible)

    def test_family_parser_accepts_only_frozen_normal_dataset_shape(self) -> None:
        self.assertEqual(family_from_dataset("ShapesAll_normal_59"), "ShapesAll")
        self.assertEqual(family_from_dataset("ECG200_normal_-1"), "ECG200")
        for invalid in ("", "ShapesAll", "_normal_0", "ShapesAll_normal_", "ShapesAll_normal_x"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                family_from_dataset(invalid)


if __name__ == "__main__":
    unittest.main()
