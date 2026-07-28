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
