from dataclasses import replace
import unittest

from experiments.exp421_mps_batched_acceleration.config import Exp421Config


class Exp421ConfigTests(unittest.TestCase):
    def test_config_is_exactly_preregistered(self):
        cfg = Exp421Config()
        self.assertEqual(cfg.candidate_batch_sizes, (1, 4, 8, 16, 32))
        self.assertEqual(cfg.max_normalized_error, 1e-4)
        self.assertEqual(cfg.min_spearman, 0.999)
        self.assertEqual(cfg.min_speedup, 1.1)
        self.assertEqual(cfg.frozen_model_seeds, (41701, 41702, 41703))
        self.assertEqual(cfg.frozen_perturbation_seeds, (41711, 41712, 41713))

    def test_config_rejects_changed_scientific_constants(self):
        changes = (
            {"candidate_batch_sizes": (1, 2, 4, 8, 16)},
            {"min_speedup": 1.2},
            {"frozen_model_seeds": (41701, 41702, 41704)},
        )
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(Exp421Config(), **change).validate()

    def test_config_has_positive_resource_bounds(self):
        cfg = Exp421Config()
        self.assertGreater(cfg.max_dataset_workers, 0)
        self.assertGreater(cfg.max_mlx_heavy_processes, 0)


if __name__ == "__main__":
    unittest.main()
