from dataclasses import replace
import unittest

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG as EXP417_CONFIG
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG, Exp422Config


class Exp422ConfigTests(unittest.TestCase):
    def test_exp422_batches_are_exact(self) -> None:
        self.assertEqual(FROZEN_CONFIG.candidate_batch_sizes, (16, 32, 64, 128, 256))

    def test_config_binds_seeds_and_tolerances_to_exp417(self) -> None:
        self.assertEqual(FROZEN_CONFIG.model_seeds, EXP417_CONFIG.model_seeds)
        self.assertEqual(FROZEN_CONFIG.perturbation_seeds, EXP417_CONFIG.perturbation_seeds)
        self.assertEqual(FROZEN_CONFIG.max_normalized_error, EXP417_CONFIG.mps_max_error)
        self.assertEqual(FROZEN_CONFIG.min_spearman, EXP417_CONFIG.mps_min_spearman)
        self.assertEqual(FROZEN_CONFIG.min_speedup, EXP417_CONFIG.mps_min_speedup)
        self.assertEqual(FROZEN_CONFIG.max_series, 64)
        self.assertEqual(FROZEN_CONFIG.max_batch_bytes, 256 * 1024 * 1024)

    def test_config_rejects_modified_or_unsorted_boundaries(self) -> None:
        for changes in (
            {"candidate_batch_sizes": (16, 32, 64, 128)},
            {"candidate_batch_sizes": (32, 16, 64, 128, 256)},
            {"max_normalized_error": 1e-3},
            {"model_seeds": (41701, 41702, 41704)},
            {"max_series": 65},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(Exp422Config(), **changes).validate()


if __name__ == "__main__":
    unittest.main()
