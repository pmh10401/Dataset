"""RED tests for the frozen Exp443 scientific/runtime contract."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
    governance_payload_antigravity,
    scientific_config_sha256_antigravity,
    validate_antigravity,
)


class Exp443ConfigAntigravityTests(unittest.TestCase):
    def test_frozen_payload_contains_exact_scientific_constraints(self) -> None:
        payload = frozen_config_payload_antigravity()
        self.assertEqual(payload["experiment_id"], "Exp443")
        self.assertEqual(payload["run_identity"], "exp443-missing-normal-train-policy-comparison-antigravity-20260809")
        self.assertEqual(
            payload["policies"],
            ["linear_edge_nearest_control", "exclude_nonfinite_row", "leave_one_out_peer_median"],
        )
        self.assertEqual(payload["kernel_requests"], [1024, 4096])
        self.assertEqual(payload["kernel_dimensions"], [1008, 4032])
        self.assertEqual(payload["seeds"], [20260721, 20261721])
        self.assertEqual(payload["alpha"], 0.005)
        self.assertEqual(payload["robust_quantile_range"], [10, 90])
        self.assertEqual(payload["dataset_count"], 1117)
        self.assertEqual(payload["affected_dataset_count"], 19)
        self.assertEqual(payload["affected_normal_train_rows"], 164)
        self.assertEqual(payload["nonfinite_scalar_count"], 983)
        self.assertEqual(payload["all_missing_row_count"], 0)
        self.assertEqual(payload["outer_workers"], 11)
        self.assertEqual(payload["inner_threads"], 1)
        self.assertEqual(payload["bootstrap_seed"], 44320260809)
        self.assertEqual(payload["bootstrap_samples"], 20000)
        self.assertEqual(payload["bootstrap_percentiles"], [2.5, 97.5])
        self.assertFalse(payload["bootstrap_endpoint"])
        self.assertEqual(payload["python_version"], "3.14.6")
        self.assertEqual(payload["isolated_environment_root"], "/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314")

    def test_paths_and_input_sha256_values_are_frozen_and_antigravity_named(self) -> None:
        payload = frozen_config_payload_antigravity()
        for key in ("database_path", "exp249_prediction_path", "exp404_hard_path", "run_root"):
            self.assertTrue(str(payload[key]))
        expected = {
            "database_sha256": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
            "exp249_prediction_sha256": "fc72d2dfbaf6a1653ec804dbe349397214992387083b22bdab0e8c26a66876c2",
            "exp249_prediction_manifest_sha256": "25ade7c3f8362559a29c15e43fead6070a7fac6cab648f03f62bcb20382b8c8f",
            "exp249_artifact_manifest_sha256": "42fee6f9a26b2c6fd19747f83887a431bf5edad144e2af3464033256a0108da9",
            "exp249_cache_manifest_sha256": "44c0ecc80ff7cbae383d49d2b8b0cefed442c49b8a12a7bd0cf8569ce121c971",
            "exp249_audit_sha256": "144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee",
            "exp291_routing_sha256": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
            "exp291_manifest_sha256": "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e",
            "exp414_repair_audit_sha256": "548f493a1608521bc81e4b742cc061fab7298f0d03c69f23845e37528b26ab91",
            "exp414_manifest_sha256": "99556922dc8f5178feaf03bc61eca4bfae628ae8f6b6092c2f6159da8c5dd3cc",
            "exp404_hard_sha256": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
            "exp404_manifest_sha256": "41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8",
        }
        for key, value in expected.items():
            self.assertEqual(payload[key], value, key)
        for key, value in payload.items():
            if key in {"run_root", "isolated_environment_root", "environment_lock_path"}:
                self.assertIn("antigravity", str(value), key)

    def test_governance_is_flat_and_review_only(self) -> None:
        self.assertEqual(
            governance_payload_antigravity(),
            {
                "official_run_purpose": "controlled_missing_normal_train_policy_comparison",
                "adaptive_followup": True,
                "independent_validation": False,
                "external_generalization_supported": False,
                "confirmatory_statistical_claim": False,
                "scientific_winner_selected": False,
                "review_only": True,
                "operational_alert_eligible": False,
                "test_informed_reselection_allowed": False,
            },
        )

    def test_mutating_any_frozen_field_fails_validation(self) -> None:
        self.assertTrue(validate_antigravity())
        mutations = {
            # identity, paths, and input hashes
            "experiment_id": "Exp442",
            "run_identity": "fabricated-run-antigravity",
            "reviewed_design_sha256": "b" * 64,
            "database_path": "/tmp/database_antigravity.db",
            "database_sha256": "c" * 64,
            "exp249_prediction_path": "/tmp/prediction_antigravity.jsonl",
            "exp249_prediction_manifest_path": "/tmp/prediction_manifest_antigravity.json",
            "exp249_artifact_manifest_path": "/tmp/artifact_manifest_antigravity.json",
            "exp249_cache_manifest_path": "/tmp/cache_manifest_antigravity.csv",
            "exp249_audit_path": "/tmp/audit_antigravity.json",
            "exp291_routing_path": "/tmp/routing_antigravity.jsonl",
            "exp291_manifest_path": "/tmp/routing_manifest_antigravity.json",
            "exp414_repair_audit_path": "/tmp/repair_audit_antigravity.csv",
            "exp414_manifest_path": "/tmp/repair_manifest_antigravity.json",
            "exp404_hard_path": "/tmp/hard_antigravity.jsonl",
            "exp404_manifest_path": "/tmp/hard_manifest_antigravity.json",
            "run_root": "/tmp/run_antigravity",
            "isolated_environment_root": "/tmp/venv_antigravity",
            "environment_lock_path": "/tmp/lock_antigravity.txt",
            "exp249_prediction_sha256": "f" * 64,
            "exp249_prediction_manifest_sha256": "1" * 64,
            "exp249_artifact_manifest_sha256": "2" * 64,
            "exp249_cache_manifest_sha256": "3" * 64,
            "exp249_audit_sha256": "4" * 64,
            "exp291_routing_sha256": "5" * 64,
            "exp291_manifest_sha256": "6" * 64,
            "exp414_repair_audit_sha256": "7" * 64,
            "exp414_manifest_sha256": "8" * 64,
            "exp404_hard_sha256": "d" * 64,
            "exp404_manifest_sha256": "9" * 64,
            # population and baseline counts
            "dataset_count": 1116,
            "affected_dataset_count": 18,
            "affected_family_count": 3,
            "affected_normal_train_rows": 163,
            "nonfinite_scalar_count": 982,
            "all_missing_row_count": 1,
            "recompute_units": 56,
            "baseline_tp": 1914,
            "baseline_fp": 304,
            "baseline_fn": 2002,
            "baseline_tn": 191680,
            # model and score settings
            "policies": ("bad",),
            "kernel_requests": (2048, 4096),
            "kernel_dimensions": (1009, 4032),
            "max_dilations": 31,
            "seeds": (20260721, 20261722),
            "alpha": 0.01,
            "folds": 4,
            "knn": 2,
            "max_reference": 999,
            "query_chunk": 64,
            "robust_quantile_range": (5, 95),
            "target_length_clip": (8, 1024),
            # bootstrap and environment
            "bootstrap_seed": 44320260810,
            "bootstrap_samples": 10000,
            "bootstrap_percentiles": (2.0, 98.0),
            "bootstrap_endpoint": True,
            "python_version": "3.14.5",
            "aeon_version": "1.4.0",
            "numpy_version": "2.3.4",
            "scikit_learn_version": "1.7.0",
            "environment_lock_sha256": "e" * 64,
            "outer_workers": 12,
            "inner_threads": 2,
            "logical_cpu_count": 11,
            # governance and routing
            "official_run_purpose": "other",
            "adaptive_followup": False,
            "independent_validation": True,
            "external_generalization_supported": True,
            "confirmatory_statistical_claim": True,
            "scientific_winner_selected": True,
            "review_only": False,
            "operational_alert_eligible": True,
            "test_informed_reselection_allowed": True,
            "acceleration_decision": "mlx",
        }
        for field_name, value in mutations.items():
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    validate_antigravity(replace(FROZEN_CONFIG, **{field_name: value}))

    def test_cpu_worker_contract_is_max_minus_one_and_hash_is_deterministic(self) -> None:
        self.assertEqual(cpu_worker_count_antigravity(12), 11)
        self.assertEqual(cpu_worker_count_antigravity(1), 1)
        self.assertRegex(scientific_config_sha256_antigravity(), r"^[0-9a-f]{64}$")
        payload_bytes = json.dumps(frozen_config_payload_antigravity(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        expected = hashlib.sha256(payload_bytes).hexdigest()
        self.assertEqual(scientific_config_sha256_antigravity(), expected)


if __name__ == "__main__":
    unittest.main()
