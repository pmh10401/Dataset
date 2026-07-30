from dataclasses import replace
import unittest

from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG as EXP417_FROZEN_CONFIG


class Exp419ConfigTests(unittest.TestCase):
    def test_exact_runtime_contract_is_frozen(self):
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp419")
        self.assertEqual(FROZEN_CONFIG.worker_candidates, (1, 2, 4, 6))
        self.assertEqual(FROZEN_CONFIG.start_method, "spawn")
        self.assertEqual(FROZEN_CONFIG.speedup_gate, 2.0)
        self.assertEqual(FROZEN_CONFIG.memory_limit_bytes, 27_487_790_694)
        self.assertEqual(FROZEN_CONFIG.torch_threads, 1)

    def test_modified_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "worker_candidates"):
            replace(FROZEN_CONFIG, worker_candidates=(1, 8)).validate()

    def test_exp417_scientific_config_database_and_lanes_are_bound(self):
        required_fields = {
            "exp417_config",
            "database_sha256",
            "expected_hard",
            "expected_standard",
            "expected_priority",
            "expected_automatic_normal",
            "exp291_prediction_sha256",
            "exp291_manifest_sha256",
            "exp292_prediction_sha256",
            "exp292_manifest_sha256",
        }
        self.assertTrue(required_fields <= set(FROZEN_CONFIG.__dataclass_fields__))
        self.assertIs(FROZEN_CONFIG.exp417_config, EXP417_FROZEN_CONFIG)
        self.assertEqual(FROZEN_CONFIG.database_sha256, EXP417_FROZEN_CONFIG.database_sha256)
        self.assertEqual(FROZEN_CONFIG.expected_hard, 2218)
        self.assertEqual(FROZEN_CONFIG.expected_standard, 806)
        self.assertEqual(FROZEN_CONFIG.expected_priority, 31)
        self.assertEqual(FROZEN_CONFIG.expected_automatic_normal, 35)
        self.assertEqual(FROZEN_CONFIG.exp291_prediction_sha256, EXP417_FROZEN_CONFIG.exp291_prediction_sha256)
        self.assertEqual(FROZEN_CONFIG.exp291_manifest_sha256, EXP417_FROZEN_CONFIG.exp291_manifest_sha256)
        self.assertEqual(FROZEN_CONFIG.exp292_prediction_sha256, EXP417_FROZEN_CONFIG.exp292_prediction_sha256)
        self.assertEqual(FROZEN_CONFIG.exp292_manifest_sha256, EXP417_FROZEN_CONFIG.exp292_manifest_sha256)

    def test_modified_exp417_database_binding_is_rejected(self):
        self.assertIn("database_sha256", FROZEN_CONFIG.__dataclass_fields__)
        with self.assertRaisesRegex(ValueError, "database_sha256"):
            replace(FROZEN_CONFIG, database_sha256="0" * 64).validate()
