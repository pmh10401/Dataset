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
        self.assertEqual(FROZEN_CONFIG.dataset_workload_rate, 0.05)
        self.assertEqual(FROZEN_CONFIG.global_workload_rate, 0.01)
        FROZEN_CONFIG.validate()

    def test_mutation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, candidate_alpha=0.02).validate()


if __name__ == "__main__":
    unittest.main()
