from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import unittest


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_preserves_exp464_identity_methods_and_exp463_controls_antigravity(self) -> None:
        """Break caught: Exp464 identity, frozen methods, or Exp463 control facts drift."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp464")
        self.assertEqual(
            config.kind,
            "ecg_residual_centered_ramp_confirmation_qualification",
        )
        self.assertEqual(
            config.run_identity,
            "exp464-ecg-residual-centered-ramp-confirmation-antigravity-20260821",
        )
        self.assertEqual(
            Path(config.official_result_root),
            Path(
                "/Users/mac/Documents/Dataset/"
                "exp464_ecg_residual_centered_ramp_confirmation_antigravity_20260821"
            ),
        )
        self.assertEqual(
            config.approved_design_sha256,
            "d0e36ae176598810289fbd5693a5e582ca6f0821a8b8fcd52edd1bcb96ec1f9f",
        )
        self.assertEqual(
            config.fixed_interpreter,
            "/Users/mac/Documents/Dataset/venvs/"
            "detection-coverage-antigravity-aeon140-py314/bin/python",
        )
        self.assertEqual(config.dataset, "ECG5000_normal_2")
        self.assertEqual(config.data_split, "TRAIN_only")
        self.assertEqual((config.normal2_train_rows, config.normal2_width), (1413, 140))
        self.assertEqual(config.hydra_seeds, (20260721, 20261721))
        self.assertEqual(config.folds, 5)
        self.assertEqual(config.onset, 56)
        self.assertEqual(config.cosine_min, 0.95)
        self.assertEqual(
            config.method_order,
            ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
        )
        self.assertEqual(
            dict(config.method_definitions),
            {
                "C0": "Exp463 M0 exact replay control",
                "C1": "Exp463 M2 exact replay control",
                "C2": "Exp463 M4 exact replay control",
                "C3": "template residual centered ramp rank only without M0 max or cosine gate",
                "C4": "same C3 rank with unconditional max against C0 and no cosine gate",
                "C5": "centered ramp confirmation without template subtraction",
                "C6": "template residual centered ramp rank with cosine >= 0.95 confirmed max against C0",
                "C7": "Exp463 M6 exact replay control",
                "C8": "template residual ramp confirmation without centering",
            },
        )
        self.assertEqual(
            config.primary_candidate,
            "template_residual_centered_ramp_confirmed_max",
        )
        self.assertEqual(config.per_type_auc_min, 0.85)
        self.assertEqual(config.pooled_auc_min, 0.90)
        self.assertEqual(config.normal_wilson_upper_max, 0.05)
        self.assertEqual(config.gradual_drift_auc_min, 0.85)
        self.assertEqual(config.gradual_drift_auc_strictly_above_exp463, 0.8402306222021025)
        self.assertEqual(config.period_change_auc_global_min, 0.907392736783864)
        self.assertEqual(
            tuple(
                (row.seed, row.fold_index, row.period_change_auc_min)
                for row in config.historical_period_fold_thresholds
            ),
            (
                (20260721, 3, 0.907392736783864),
                (20261721, 4, 0.9216148584075248),
            ),
        )
        self.assertEqual(config.predecessor_experiment, "Exp463")
        self.assertEqual(
            config.predecessor_terminal_closeout_commit,
            "147225831e9c13e9df5043fa88757bb03e01b79a",
        )
        self.assertEqual(
            config.predecessor_task5_report_sha256,
            "e43abf66e43921941dece398514858be91e39d121c1173b26b352b67ba78e8e5",
        )
        self.assertEqual(config.predecessor_private_run_count, 2)
        self.assertEqual(config.predecessor_result_row_count, 70)
        self.assertEqual(config.predecessor_result_failed_row_count, 30)
        self.assertEqual(config.predecessor_result_summary_count, 7)
        self.assertEqual(config.predecessor_result_leaf_count, 7)
        self.assertEqual(
            config.predecessor_artifact_map_sha256,
            "25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5",
        )
        self.assertEqual(config.label_query_count_must_remain, 0)
        self.assertEqual(config.official_evaluate_count_must_remain, 0)
        self.assertEqual(config.official_inspect_count_must_remain, 0)
        self.assertEqual(
            tuple(config.predecessor_control_summaries),
            ("C0", "C1", "C2", "C7"),
        )
        self.assertEqual(config.predecessor_control_summaries["C0"].passed_count, 9)
        self.assertEqual(config.predecessor_control_summaries["C1"].passed_count, 7)
        self.assertEqual(config.predecessor_control_summaries["C2"].passed_count, 9)
        self.assertEqual(config.predecessor_control_summaries["C7"].passed_count, 3)
        with self.assertRaises(FrozenInstanceError):
            config.primary_candidate = "C4"  # type: ignore[misc]

    def test_config_rejects_type_aliases_reordered_mappings_and_altered_controls_antigravity(self) -> None:
        """Break caught: strict config accepts bool/int aliases, mutable containers, or forged controls."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
            Exp463MethodSummaryAntigravity,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        forged_c0 = Exp463MethodSummaryAntigravity(
            method="C0",
            fold_count=10,
            passed_count=8,
            failed_count=2,
            min_gradual_drift_auc=0.8402306222021025,
            min_period_change_auc=0.907392736783864,
            min_pooled_auc=0.9212266108344651,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        )
        mutations = (
            ("dataset", "ECG5000_normal_1"),
            ("hydra_seeds", list(config.hydra_seeds)),
            ("folds", True),
            ("onset", False),
            ("cosine_min", 0.99),
            ("method_order", list(config.method_order)),
            (
                "method_definitions",
                {
                    "C1": dict(config.method_definitions)["C1"],
                    **{key: value for key, value in config.method_definitions.items() if key != "C1"},
                },
            ),
            ("primary_candidate", "C6"),
            ("predecessor_terminal_closeout_commit", "147225831e9c13e9df5043fa88757bb03e01b70"),
            ("predecessor_result_row_count", 69),
            ("label_query_count_must_remain", 1),
            (
                "predecessor_control_summaries",
                {**dict(config.predecessor_control_summaries), "C0": forged_c0},
            ),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                replace(config, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
