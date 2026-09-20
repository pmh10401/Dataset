from dataclasses import replace
from pathlib import Path
import os
import unittest

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.config_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    EVALUATION_ROOT_ANTIGRAVITY,
    FROZEN_CONFIG,
    OUTPUT_ROOT_ANTIGRAVITY,
    READINESS_PATH_ANTIGRAVITY,
    RUN_PARENT_ANTIGRAVITY,
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
    validate_frozen_config_antigravity,
)


class Exp439ConfigAntigravityTests(unittest.TestCase):
    def test_frozen_identity_paths_and_population(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp439")
        self.assertEqual(
            FROZEN_CONFIG.run_identity,
            "exp439-scale-aware-reconstruction-repair-antigravity-20260807",
        )
        self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
        self.assertEqual(
            DATABASE_PATH_ANTIGRAVITY,
            Path("/Users/mac/Documents/Dataset/univariate_ts.db"),
        )
        self.assertEqual(
            RUN_PARENT_ANTIGRAVITY,
            Path(
                "/Users/mac/Documents/Dataset/"
                "exp439_scale_aware_reconstruction_repair_scheduler_antigravity_20260807"
            ),
        )
        self.assertEqual(OUTPUT_ROOT_ANTIGRAVITY, RUN_PARENT_ANTIGRAVITY / "output_antigravity")
        self.assertEqual(EVALUATION_ROOT_ANTIGRAVITY, RUN_PARENT_ANTIGRAVITY / "evaluation_antigravity")
        self.assertEqual(READINESS_PATH_ANTIGRAVITY, RUN_PARENT_ANTIGRAVITY / "readiness_antigravity.json")

    def test_scientific_constants_are_frozen(self) -> None:
        self.assertEqual(FROZEN_CONFIG.database_sha256, "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c")
        self.assertEqual(FROZEN_CONFIG.minimum_raw_length, 16)
        self.assertEqual(FROZEN_CONFIG.minimum_train_rows, 4)
        self.assertEqual(FROZEN_CONFIG.maximum_depth, 5)
        self.assertEqual(FROZEN_CONFIG.embedding_dimension, 3)
        self.assertEqual(FROZEN_CONFIG.delays, (1, 2, 4))
        self.assertEqual(FROZEN_CONFIG.nearest_k, 3)
        self.assertEqual(FROZEN_CONFIG.quantile, 0.99)
        self.assertEqual(FROZEN_CONFIG.quantile_method, "higher")
        self.assertTrue(FROZEN_CONFIG.strict_greater)
        self.assertEqual(FROZEN_CONFIG.channel_candidate_budget, 10_000)
        self.assertEqual(FROZEN_CONFIG.total_candidate_budget, 30_000)
        self.assertEqual(FROZEN_CONFIG.smoke_seed, 43901)
        self.assertEqual(FROZEN_CONFIG.batch_sizes, (16, 32, 64, 128, 256))
        self.assertEqual(FROZEN_CONFIG.warmup_runs, 2)
        self.assertEqual(FROZEN_CONFIG.timed_runs, 5)
        self.assertEqual(FROZEN_CONFIG.parity_tolerance, 1e-5)
        self.assertEqual(FROZEN_CONFIG.minimum_speedup, 1.1)
        self.assertEqual(FROZEN_CONFIG.bootstrap_seed, 43920260807)
        self.assertEqual(FROZEN_CONFIG.bootstrap_samples, 20_000)
        self.assertTrue(FROZEN_CONFIG.review_only)
        self.assertFalse(FROZEN_CONFIG.operational_alert_eligible)

    def test_cpu_workers_are_max_minus_one(self) -> None:
        self.assertEqual(cpu_worker_count_antigravity(12), 11)
        self.assertEqual(cpu_worker_count_antigravity(2), 1)
        self.assertEqual(cpu_worker_count_antigravity(1), 1)
        self.assertEqual(cpu_worker_count_antigravity(None), max(1, (os.cpu_count() or 1) - 1))
        with self.assertRaisesRegex(TypeError, "logical_cpus"):
            cpu_worker_count_antigravity(True)
        with self.assertRaisesRegex(ValueError, "logical_cpus"):
            cpu_worker_count_antigravity(0)

    def test_config_payload_is_canonical_and_validation_rejects_change(self) -> None:
        payload = frozen_config_payload_antigravity()
        self.assertEqual(payload["delays"], [1, 2, 4])
        self.assertEqual(payload["batch_sizes"], [16, 32, 64, 128, 256])
        self.assertIs(validate_frozen_config_antigravity(FROZEN_CONFIG), FROZEN_CONFIG)
        with self.assertRaisesRegex(ValueError, "frozen Exp439 config"):
            validate_frozen_config_antigravity(replace(FROZEN_CONFIG, quantile=0.95))


if __name__ == "__main__":
    unittest.main()
