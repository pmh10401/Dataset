from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import unittest


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_preserves_exp463_identity_methods_and_gates_antigravity(self) -> None:
        """Break caught: identity, TRAIN source, M0-M6, or fixed gates drift."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp463")
        self.assertEqual(config.kind, "ecg_prefix_ramp_gated_qualification")
        self.assertEqual(
            config.run_identity,
            "exp463-ecg-prefix-ramp-gated-qualification-antigravity-20260821",
        )
        self.assertEqual(
            Path(config.official_result_root),
            Path("/Users/mac/Documents/Dataset/exp463_ecg_prefix_ramp_gated_qualification_antigravity_20260821"),
        )
        self.assertEqual(
            config.approved_design_sha256,
            "d5af337a09b11a707198eb0d669a9a6a4b330db44db698daa5ca218cdfbb7efb",
        )
        self.assertEqual(config.dataset, "ECG5000_normal_2")
        self.assertEqual(config.data_split, "TRAIN_only")
        self.assertEqual((config.normal2_train_rows, config.normal2_width), (1413, 140))
        self.assertEqual(config.hydra_seeds, (20260721, 20261721))
        self.assertEqual(config.folds, 5)
        self.assertEqual(config.onset, 56)
        self.assertEqual(config.alpha, 0.005)
        self.assertEqual(config.gate, 0.995)
        self.assertEqual(config.change_order, ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"))
        self.assertEqual(config.method_order, ("M0", "M1", "M2", "M3", "M4", "M5", "M6"))
        self.assertEqual(
            dict(config.method_definitions),
            {
                "M0": "Exp461 reviewed hydra_period_rank_max score",
                "M1": "prefix median/MAD normalized one-dimensional ramp from onset 56",
                "M2": "same M1 ramp; if rramp >= 0.995, max(r461, rramp); otherwise exactly r461 (M0/Exp461 semantics)",
                "M3": "same M1 ramp",
                "M4": "TRAIN-fit per-time median template residual ramp",
                "M5": "whole-series median/MAD normalized ramp",
                "M6": "Exp462 24D signature + LocalGap replay control",
            },
        )
        self.assertEqual(config.primary_candidate, "hydra_period_ramp_gated_max")
        self.assertEqual(config.per_type_auc_min, 0.85)
        self.assertEqual(config.pooled_auc_min, 0.90)
        self.assertEqual(config.normal_wilson_upper_max, 0.05)
        self.assertEqual(config.gradual_drift_auc_min, 0.85)
        self.assertEqual(config.gradual_drift_auc_strictly_above_exp462, 0.8402306222021025)
        self.assertEqual(config.period_change_auc_global_min, 0.907392736783864)
        self.assertEqual(
            tuple((row.seed, row.fold_index, row.period_change_auc_min) for row in config.historical_period_fold_thresholds),
            ((20260721, 3, 0.907392736783864), (20261721, 4, 0.9216148584075248)),
        )
        self.assertEqual(config.predecessor_experiment, "Exp462")
        self.assertEqual(config.predecessor_terminal_closeout_commit, "040422f62406e3ca19038b9609903c4ba76e1694")
        self.assertEqual(config.predecessor_status, "stopped_before_official_evaluation_qualification_failed")
        self.assertEqual(config.predecessor_primary_candidate, "hydra_period_gradual_rank_max")
        self.assertEqual(config.predecessor_passed_record_count, 3)
        self.assertEqual(config.predecessor_required_record_count, 10)
        self.assertEqual(config.predecessor_failed_gradual_drift_auc, 0.7953196519289775)
        self.assertEqual(config.predecessor_period_min_auc, 0.8960376741612595)
        self.assertEqual(config.predecessor_result_row_count, 30)
        self.assertEqual(config.predecessor_result_summary_count, 3)
        self.assertEqual(config.predecessor_result_leaf_count, 7)
        self.assertEqual(config.label_query_count_must_remain, 0)
        self.assertEqual(config.official_command_invocations_must_remain, 0)
        with self.assertRaises(FrozenInstanceError):
            config.primary_candidate = "M1"  # type: ignore[misc]

    def test_config_rejects_mutated_values_types_or_order_antigravity(self) -> None:
        """Break caught: strict config accepts a lookalike, bool, or reordered value."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        mutations = (
            ("experiment", "Exp462"),
            ("dataset", "ECG5000_normal_1"),
            ("data_split", "TEST_only"),
            ("normal2_train_rows", 1412),
            ("hydra_seeds", tuple(reversed(config.hydra_seeds))),
            ("folds", True),
            ("onset", 55),
            ("alpha", True),
            ("gate", 0.99),
            ("method_order", tuple(reversed(config.method_order))),
            ("primary_candidate", "M2"),
            ("gradual_drift_auc_strictly_above_exp462", 0.7953196519289775),
            ("predecessor_terminal_closeout_commit", "040422f62406e3ca19038b9609903c4ba76e1695"),
            ("predecessor_result_row_count", 29),
            ("label_query_count_must_remain", 1),
            ("official_command_invocations_must_remain", 1),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name, value=value), self.assertRaises(ValueError):
                replace(config, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
