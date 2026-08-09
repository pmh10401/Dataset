"""RED/GREEN tests for the frozen Exp444 configuration."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
    scientific_config_sha256_antigravity,
    validate_antigravity,
)


class Exp444ConfigAntigravityTests(unittest.TestCase):
    def test_frozen_config_has_descriptive_governance_antigravity(self) -> None:
        self.assertEqual(FROZEN_CONFIG.source_candidate_count, 30)
        self.assertEqual(FROZEN_CONFIG.maximum_stability_views, 16)
        self.assertEqual(FROZEN_CONFIG.stability_fraction, 0.875)
        self.assertIsNone(FROZEN_CONFIG.scientific_success_criterion)
        self.assertEqual(FROZEN_CONFIG.evaluation_status, "descriptive_complete_no_promotion")
        self.assertEqual(FROZEN_CONFIG.cpu_workers, 11)

    def test_payload_contains_exact_frozen_identities_and_governance(self) -> None:
        payload = frozen_config_payload_antigravity()
        self.assertEqual(payload["experiment_id"], "Exp444")
        self.assertEqual(payload["run_identity"], "exp444-train-only-stability-precision-gate-antigravity-20260810")
        self.assertEqual(payload["run_root"], "/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810")
        self.assertEqual(payload["isolated_environment_root"], "/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314")
        self.assertEqual(payload["dataset_count"], 1117)
        self.assertEqual(payload["source_candidate_count"], 30)
        self.assertEqual(payload["candidate_per_dataset"], 1)
        self.assertEqual(payload["maximum_stability_views"], 16)
        self.assertEqual(payload["stability_fraction"], 0.875)
        self.assertEqual(payload["mask_seed"], 44420260810)
        self.assertEqual(payload["bootstrap_seed"], 44420260811)
        self.assertEqual(payload["bootstrap_samples"], 20000)
        self.assertEqual(payload["acceleration_tolerance"], 1e-5)
        self.assertEqual(payload["acceleration_speedup_min"], 1.1)
        self.assertEqual(payload["expected_cpu_workers_on_current_mac"], 11)
        self.assertIsNone(payload["scientific_success_criterion"])
        self.assertEqual(payload["evaluation_status"], "descriptive_complete_no_promotion")
        self.assertTrue(payload["adaptive_followup"])
        self.assertFalse(payload["independent_validation"])
        self.assertFalse(payload["scientific_winner_selected"])
        self.assertTrue(payload["review_only"])
        self.assertFalse(payload["operational_alert_eligible"])
        self.assertFalse(payload["same_experiment_reselection_allowed"])
        self.assertEqual(payload["requirements"], ["mlx==0.32.0", "numba==0.66.0", "numpy==2.4.6"])

    def test_frozen_input_hashes_are_exact(self) -> None:
        payload = frozen_config_payload_antigravity()
        expected = {
            "database_sha256": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
            "exp404_hard_prediction_sha256": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
            "exp442_prediction_sha256": "b79bad6012ad6bc07050ade9968e815796a2cc871b5c85d63f0bd7dbdf6f81f4",
            "exp442_candidate_ranking_sha256": "d8d858cd74553aa64f6f0a5424064872b59f682fed1c8b2ace63fffba6ad20cc",
            "exp442_selection_sha256": "06004f18dbd9b0df3fa857bb8b1a9fd7eafd4d927f3736fb8cfb25db56b72408",
            "exp442_selected_coordinate_identity_sha256": "db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048",
        }
        for key, value in expected.items():
            self.assertEqual(payload[key], value, key)

    def test_cpu_worker_formula_and_canonical_sha(self) -> None:
        self.assertEqual(cpu_worker_count_antigravity(12), 11)
        self.assertEqual(cpu_worker_count_antigravity(1), 1)
        with self.assertRaises(ValueError):
            cpu_worker_count_antigravity(True)  # type: ignore[arg-type]
        encoded = json.dumps(frozen_config_payload_antigravity(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        self.assertEqual(scientific_config_sha256_antigravity(), hashlib.sha256(encoded).hexdigest())
        self.assertTrue(validate_antigravity())

    def test_validation_rejects_mutated_scientific_or_governance_field(self) -> None:
        for field_name, value in (
            ("experiment_id", "Exp442"),
            ("source_candidate_count", 31),
            ("maximum_stability_views", 15),
            ("stability_fraction", 0.9),
            ("scientific_success_criterion", "f1"),
            ("evaluation_status", "promotion"),
            ("cpu_workers", 12),
            ("environment_lock_sha256", "f" * 64),
            ("review_only", False),
        ):
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                validate_antigravity(replace(FROZEN_CONFIG, **{field_name: value}))


if __name__ == "__main__":
    unittest.main()
