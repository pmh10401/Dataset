from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_preserves_the_approved_scientific_contract_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp458")
        self.assertEqual(
            config.approved_design_sha256,
            "52abe997c0e14badc2c886b888e3b2b2074cf002216891f173ab3b1dd6d7ea5d",
        )
        self.assertEqual(config.dataset, "ECG5000_normal_2")
        self.assertEqual(config.kind, "ecg_normal2_rank_confidence_routing")
        self.assertEqual(
            config.run_identity,
            "exp458-ecg-normal2-rank-confidence-routing-antigravity-20260818",
        )
        self.assertEqual(
            config.official_result_root,
            "/Users/mac/Documents/Dataset/exp458_ecg_normal2_rank_confidence_routing_antigravity_20260818",
        )
        self.assertEqual(config.hydra_seeds, (20260721, 20261721))
        self.assertEqual(config.folds, 5)
        self.assertEqual(
            config.primary_types,
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        )
        self.assertEqual(
            (
                config.per_type_auc_min,
                config.pooled_auc_min,
                config.normal_wilson_upper_max,
            ),
            (0.85, 0.90, 0.05),
        )
        self.assertEqual(config.structure_gate, "added_count > removed_count")
        self.assertEqual(config.normal2_train_trials, 1413)
        self.assertEqual(config.normal2_train_alerts, 6)
        self.assertEqual(config.normal2_train_wilson_upper, 0.00923330730695328)
        self.assertEqual(
            config.task3_rank_auc_by_seed,
            (
                (
                    20260721,
                    (
                        ("level_shift", 0.9796866524522819),
                        ("gradual_drift", 0.9052489545815847),
                        ("amplitude_scaling", 0.9951992643379718),
                        ("period_change", 0.9104679076956519),
                    ),
                    0.9476506947668726,
                ),
                (
                    20261721,
                    (
                        ("level_shift", 0.9761215364958586),
                        ("gradual_drift", 0.895775202359648),
                        ("amplitude_scaling", 0.9943593234193259),
                        ("period_change", 0.9278161686372973),
                    ),
                    0.9485180577280324,
                ),
            ),
        )
        self.assertIs(config.finite_required, True)
        self.assertIs(config.test_metrics_may_inform_routing, False)
        self.assertIs(config.mlx_applicable, False)

        with self.assertRaises(FrozenInstanceError):
            config.pooled_auc_min = 0.91  # type: ignore[misc]

    def test_config_rejects_any_mutated_value_type_or_order_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        class ExactStrSubclassAntigravity(str):
            pass

        config = FROZEN_CONFIG_ANTIGRAVITY
        mutations = (
            ("experiment", "Exp457"),
            ("experiment", ExactStrSubclassAntigravity("Exp458")),
            ("approved_design_sha256", "0" * 64),
            ("dataset", "ECG5000_normal_1"),
            ("kind", "other"),
            ("run_identity", "exp458-other"),
            ("official_result_root", "/Users/mac/Documents/Dataset/other"),
            ("hydra_seeds", tuple(reversed(config.hydra_seeds))),
            ("hydra_seeds", (20260721, 20261721, 20262721)),
            ("folds", True),
            ("folds", 4),
            ("primary_types", tuple(reversed(config.primary_types))),
            ("primary_types", config.primary_types + ("secondary_spike",)),
            ("per_type_auc_min", True),
            ("per_type_auc_min", 0.84),
            ("pooled_auc_min", True),
            ("pooled_auc_min", 0.91),
            ("normal_wilson_upper_max", 0.0500000001),
            ("normal_wilson_upper_max", True),
            ("structure_gate", "added_count >= removed_count"),
            ("normal2_train_trials", 1412),
            ("normal2_train_alerts", 5),
            ("normal2_train_wilson_upper", 0.0),
            ("task3_rank_auc_by_seed", tuple(reversed(config.task3_rank_auc_by_seed))),
            ("finite_required", 1),
            ("test_metrics_may_inform_routing", True),
            ("mlx_applicable", True),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(config, **{field: value})


if __name__ == "__main__":
    unittest.main()
