from __future__ import annotations

import unittest

from experiments.exp452_exp449_index_order_repair_antigravity.config_antigravity import (
    CANDIDATE_MATRIX_ANTIGRAVITY,
    DATASET_COUNT_ANTIGRAVITY,
    EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
    EXP449_PREDICTION_ROOT_ANTIGRAVITY,
    EXP452_EVALUATION_ROOT_ANTIGRAVITY,
    EXP452_FAILURE_AUDIT_ANTIGRAVITY,
    EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY,
    FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY,
    FROZEN_CONFIG_ANTIGRAVITY,
    INDEX_ORDER_RULE_ANTIGRAVITY,
    PREDICTION_COUNT_ANTIGRAVITY,
)


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_identity_counts_and_predecessor_authority_are_exact_antigravity(self) -> None:
        """Break caught: changing Exp452 identity, population, or predecessor bytes."""
        self.assertEqual(
            (
                FROZEN_CONFIG_ANTIGRAVITY["experiment"],
                FROZEN_CONFIG_ANTIGRAVITY["schema"],
                FROZEN_CONFIG_ANTIGRAVITY["kind"],
                FROZEN_CONFIG_ANTIGRAVITY["run_id"],
            ),
            (
                "Exp452",
                "exp452",
                "exp449_index_order_repair",
                "exp452-exp449-index-order-repair-antigravity-20260813",
            ),
        )
        self.assertEqual(
            (DATASET_COUNT_ANTIGRAVITY, PREDICTION_COUNT_ANTIGRAVITY, FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY),
            (48, 96, 16),
        )
        self.assertEqual(
            EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
            "a6054a44fc4e78f354dcf61d2e003284804680cd4197405364fcb5ef8d822e0d",
        )
        self.assertEqual(INDEX_ORDER_RULE_ANTIGRAVITY, "sorted_ascending_exact_list")

    def test_frozen_paths_are_exact_strings_without_touching_them_antigravity(self) -> None:
        """Break caught: redirecting a future lifecycle to an unapproved filesystem target."""
        self.assertEqual(
            (
                EXP449_PREDICTION_ROOT_ANTIGRAVITY,
                EXP452_EVALUATION_ROOT_ANTIGRAVITY,
                EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY,
                EXP452_FAILURE_AUDIT_ANTIGRAVITY,
            ),
            (
                "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812",
                "/Users/mac/Documents/Dataset/exp452_exp449_index_order_repair_evaluation_antigravity_20260813",
                "/Users/mac/Documents/Dataset/exp452_index_order_repair_lifecycle_receipts_antigravity_20260813",
                "/Users/mac/Documents/Dataset/exp452_index_order_repair_failure_audit_antigravity.json",
            ),
        )
        for value in (
            EXP449_PREDICTION_ROOT_ANTIGRAVITY,
            EXP452_EVALUATION_ROOT_ANTIGRAVITY,
            EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY,
            EXP452_FAILURE_AUDIT_ANTIGRAVITY,
        ):
            self.assertIs(type(value), str)

    def test_candidate_matrix_and_scientific_settings_match_exp449_antigravity(self) -> None:
        """Break caught: retuning a family-method pair, seed, alpha, or score threshold."""
        self.assertEqual(CANDIDATE_MATRIX_ANTIGRAVITY, (
            ("Crop", ("minirocket_local_gap", "multirocket_local_gap")),
            ("ElectricDevices", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
            ("FordA", ("minirocket_local_gap", "hydra_local_gap")),
            ("FordB", ("multirocket_local_gap", "hydra_local_gap")),
            ("Yoga", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
            ("ECG5000", ("multirocket_local_gap", "hydra_local_gap")),
            ("ChlorineConcentration", ("minirocket_local_gap", "multirocket_local_gap")),
            ("StarLightCurves", ("multirocket_local_gap", "hydra_local_gap")),
        ))
        self.assertEqual(sum(len(methods) for _family, methods in CANDIDATE_MATRIX_ANTIGRAVITY), 16)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["seeds"], (20260721, 20261721))
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["alphas"], (0.005,))
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["global_fn_must_be_below"], 1988)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["global_fp_max"], 314)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["global_f1_min"], 0.6264199935)
        self.assertFalse(any("qualified" in key.lower() for key in FROZEN_CONFIG_ANTIGRAVITY))
        with self.assertRaises(TypeError):
            FROZEN_CONFIG_ANTIGRAVITY["alphas"] = (0.01,)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
