from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import unittest


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_preserves_task2_identity_and_gates_antigravity(self) -> None:
        """Break caught: Exp462 identity, TRAIN contract, methods, or gates drift."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp462")
        self.assertEqual(config.kind, "ecg_gradual_signature_qualification")
        self.assertEqual(
            config.run_identity,
            "exp462-ecg-gradual-signature-qualification-antigravity-20260820",
        )
        self.assertEqual(
            Path(config.official_result_root),
            Path("/Users/mac/Documents/Dataset/exp462_ecg_gradual_signature_qualification_antigravity_20260820"),
        )
        self.assertEqual(config.dataset, "ECG5000_normal_2")
        self.assertEqual(config.data_split, "TRAIN_only")
        self.assertEqual((config.normal2_train_rows, config.normal2_width), (1413, 140))
        self.assertEqual(config.hydra_seeds, (20260721, 20261721))
        self.assertEqual(config.folds, 5)
        self.assertEqual(config.alpha, 0.005)
        self.assertEqual(config.knn_k, 3)
        self.assertEqual(
            config.change_order,
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        )
        self.assertEqual(
            config.method_order,
            (
                "hydra_period_rank_max",
                "gradual_signature_local_gap",
                "hydra_period_gradual_rank_max",
            ),
        )
        self.assertEqual(config.primary_candidate, "hydra_period_gradual_rank_max")
        self.assertEqual(config.per_type_auc_min, 0.85)
        self.assertEqual(config.pooled_auc_min, 0.90)
        self.assertEqual(config.normal_wilson_upper_max, 0.05)
        self.assertEqual(config.gradual_drift_auc_min, 0.85)
        self.assertEqual(config.period_change_auc_global_min, 0.907392736783864)
        self.assertEqual(config.predecessor_experiment, "Exp461")
        self.assertEqual(
            config.predecessor_terminal_commit,
            "0340b6ee737da84777283668449738d0c1585566",
        )
        self.assertEqual(config.predecessor_passed_record_count, 9)
        self.assertEqual(config.predecessor_required_record_count, 10)
        self.assertEqual(config.predecessor_failed_gradual_drift_auc, 0.8402306222021025)
        self.assertEqual(config.predecessor_period_min_auc, 0.907392736783864)
        self.assertEqual(config.label_query_count_must_remain, 0)
        self.assertEqual(config.official_command_invocations_must_remain, 0)
        with self.assertRaises(FrozenInstanceError):
            config.primary_candidate = "gradual_signature_local_gap"  # type: ignore[misc]

    def test_config_rejects_mutated_values_types_or_order_antigravity(self) -> None:
        """Break caught: strict dataclass accepts a lookalike, bool, or reordered tuple."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        mutations = (
            ("experiment", "Exp461"),
            ("dataset", "ECG5000_normal_1"),
            ("data_split", "TEST_only"),
            ("normal2_train_rows", 1412),
            ("hydra_seeds", tuple(reversed(config.hydra_seeds))),
            ("folds", True),
            ("alpha", True),
            ("knn_k", 4),
            ("change_order", tuple(reversed(config.change_order))),
            ("method_order", tuple(reversed(config.method_order))),
            ("primary_candidate", "hydra_period_rank_max"),
            ("gradual_drift_auc_min", 0.8402306222021025),
            ("period_change_auc_global_min", 0.9073927367838639),
            ("predecessor_terminal_commit", "3a306ff3e276dae13b28b9733db30e9934da7eb2"),
            ("predecessor_passed_record_count", 10),
            ("label_query_count_must_remain", 1),
            ("official_command_invocations_must_remain", 1),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name, value=value), self.assertRaises(ValueError):
                replace(config, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
