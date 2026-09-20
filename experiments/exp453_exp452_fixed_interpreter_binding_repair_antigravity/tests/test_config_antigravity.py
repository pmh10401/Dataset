from __future__ import annotations

import unittest

import experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.config_antigravity as config_module_antigravity
from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.config_antigravity import (
    CANDIDATE_MATRIX_ANTIGRAVITY,
    DATASET_COUNT_ANTIGRAVITY,
    EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
    EXP453_EVALUATION_ROOT_ANTIGRAVITY,
    EXP453_FAILURE_AUDIT_ANTIGRAVITY,
    EXP453_RECEIPT_DIRECTORY_ANTIGRAVITY,
    FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY,
    FIXED_EXECUTABLE_ANTIGRAVITY,
    FIXED_PREFIX_ANTIGRAVITY,
    FROZEN_CONFIG_ANTIGRAVITY,
    PREDICTION_COUNT_ANTIGRAVITY,
    RUNTIME_VERSIONS_ANTIGRAVITY,
)


class ConfigAntigravityTest(unittest.TestCase):
    def test_identity_population_and_runtime_binding_are_exact_antigravity(self) -> None:
        """Break caught: retargeting Exp453 to another runtime or population."""
        self.assertEqual(
            (
                FROZEN_CONFIG_ANTIGRAVITY["experiment"],
                FROZEN_CONFIG_ANTIGRAVITY["schema"],
                FROZEN_CONFIG_ANTIGRAVITY["kind"],
                FROZEN_CONFIG_ANTIGRAVITY["run_id"],
            ),
            ("Exp453", "exp453", "exp452_fixed_interpreter_binding_repair", "exp453-exp452-fixed-interpreter-binding-repair-antigravity-20260813"),
        )
        self.assertEqual((DATASET_COUNT_ANTIGRAVITY, PREDICTION_COUNT_ANTIGRAVITY, FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY), (48, 96, 16))
        self.assertEqual(FIXED_EXECUTABLE_ANTIGRAVITY, "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python")
        self.assertEqual(FIXED_PREFIX_ANTIGRAVITY, "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314")
        self.assertEqual(RUNTIME_VERSIONS_ANTIGRAVITY, ("3.14.6", "2.3.5", "1.4.0", "1.7.2"))

    def test_scientific_profile_is_the_frozen_exp452_profile_antigravity(self) -> None:
        """Break caught: changing a method, seed, threshold, or prediction input."""
        self.assertEqual(EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY, "a6054a44fc4e78f354dcf61d2e003284804680cd4197405364fcb5ef8d822e0d")
        self.assertEqual(sum(len(methods) for _family, methods in CANDIDATE_MATRIX_ANTIGRAVITY), 16)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["seeds"], (20260721, 20261721))
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["alphas"], (0.005,))
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["global_fn_must_be_below"], 1988)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["global_fp_max"], 314)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["global_f1_min"], 0.6264199935)
        with self.assertRaises(TypeError):
            FROZEN_CONFIG_ANTIGRAVITY["alphas"] = (0.01,)  # type: ignore[index]

    def test_new_paths_are_exact_and_distinct_from_exp452_antigravity(self) -> None:
        """Break caught: writing Exp453 output into the immutable Exp452 namespace."""
        self.assertEqual(EXP453_EVALUATION_ROOT_ANTIGRAVITY, "/Users/mac/Documents/Dataset/exp453_exp452_fixed_interpreter_binding_repair_evaluation_antigravity_20260813")
        self.assertEqual(EXP453_RECEIPT_DIRECTORY_ANTIGRAVITY, "/Users/mac/Documents/Dataset/exp453_fixed_interpreter_binding_repair_lifecycle_receipts_antigravity_20260813")
        self.assertEqual(EXP453_FAILURE_AUDIT_ANTIGRAVITY, "/Users/mac/Documents/Dataset/exp453_fixed_interpreter_binding_repair_failure_audit_antigravity.json")
        for value in (EXP453_EVALUATION_ROOT_ANTIGRAVITY, EXP453_RECEIPT_DIRECTORY_ANTIGRAVITY, EXP453_FAILURE_AUDIT_ANTIGRAVITY):
            self.assertIs(type(value), str)
            self.assertIn("exp453", value)

    def test_evaluation_leaf_inventory_is_frozen_before_evaluator_implementation_antigravity(self) -> None:
        """Break caught: a later evaluator silently changes a result filename or order."""
        self.assertTrue(hasattr(config_module_antigravity, "EVALUATION_LEAVES_ANTIGRAVITY"))
        self.assertEqual(config_module_antigravity.EVALUATION_LEAVES_ANTIGRAVITY, (
            "context_antigravity.json", "compatibility_antigravity.json", "candidate_registry_antigravity.json",
            "frozen_predictions_antigravity.jsonl", "prediction_manifest_antigravity.json", "evaluation_antigravity.json",
            "candidate_metrics_antigravity.jsonl", "family_metrics_antigravity.jsonl", "bootstrap_antigravity.json",
            "decision_antigravity.json", "dashboard_snapshot_antigravity.json", "artifact_sha256_antigravity.json",
        ))


if __name__ == "__main__":
    unittest.main()
