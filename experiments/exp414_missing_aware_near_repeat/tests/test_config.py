import unittest
from dataclasses import replace

from experiments.exp414_missing_aware_near_repeat.config import FROZEN_CONFIG


class ConfigTests(unittest.TestCase):
    def test_frozen_values_match_preregistration(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp414")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.expected_families, 128)
        self.assertEqual(FROZEN_CONFIG.expected_missing_datasets, 19)
        self.assertEqual(FROZEN_CONFIG.expected_missing_runs, 164)
        self.assertEqual(FROZEN_CONFIG.expected_nonfinite_values, 983)
        self.assertEqual(FROZEN_CONFIG.expected_all_missing_runs, 0)
        self.assertEqual(FROZEN_CONFIG.seed, 411)
        self.assertEqual(FROZEN_CONFIG.quantiles, (0.01, 0.05, 0.10))
        FROZEN_CONFIG.validate()

    def test_contract_rejects_mutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, expected_missing_datasets=18).validate()
