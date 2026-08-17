from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_preserves_the_approved_scientific_contract_antigravity(self) -> None:
        """Break caught: a result-affecting Exp457 value drifts before qualification."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp457")
        self.assertEqual(
            config.approved_design_sha256,
            "3b702e0224dd86e1d0c93268b8a2e2691ea3858cbd30a4792f9c6539c0424e20",
        )
        self.assertEqual(
            config.ecg_datasets,
            tuple(f"ECG5000_normal_{index}" for index in range(1, 6)),
        )
        self.assertEqual(config.candidate, "hydra_local_gap")
        self.assertEqual(config.chlorine_route, "multirocket_local_gap")
        self.assertEqual(config.hydra_seeds, (20260721, 20261721))
        self.assertEqual((config.folds, config.alpha, config.knn_k), (5, 0.005, 3))
        self.assertEqual((config.kernels, config.groups), (8, 64))
        self.assertEqual(
            config.primary_types,
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        )
        self.assertEqual(
            (
                config.normal_wilson_upper_max,
                config.pooled_sensitivity_min,
                config.per_type_sensitivity_min,
            ),
            (0.05, 0.70, 0.50),
        )
        self.assertEqual(config.structure_gate, "added_count > removed_count")
        self.assertEqual(
            (config.final_fn_max, config.final_fp_max, config.final_f1_strictly_above),
            (834, 66, 0.5764150943396227),
        )
        self.assertIs(config.finite_required, True)
        self.assertIs(config.test_metrics_may_inform_routing, False)
        self.assertIs(config.mlx_applicable, False)

        with self.assertRaises(FrozenInstanceError):
            config.alpha = 0.05  # type: ignore[misc]

    def test_config_rejects_any_mutated_value_type_or_order_antigravity(self) -> None:
        """Break caught: a second config instance weakens or reorders the frozen experiment."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        mutations = (
            ("experiment", "Exp458"),
            ("approved_design_sha256", "0" * 64),
            ("ecg_datasets", tuple(reversed(config.ecg_datasets))),
            ("candidate", "other"),
            ("chlorine_route", "baseline"),
            ("hydra_seeds", tuple(reversed(config.hydra_seeds))),
            ("folds", True),
            ("alpha", True),
            ("alpha", 0.05),
            ("knn_k", 3.0),
            ("kernels", True),
            ("groups", 32),
            ("primary_types", tuple(reversed(config.primary_types))),
            ("primary_types", config.primary_types + ("secondary_spike",)),
            ("normal_wilson_upper_max", float("nan")),
            ("pooled_sensitivity_min", 1),
            ("per_type_sensitivity_min", -0.5),
            ("finite_required", 1),
            ("structure_gate", "added_count >= removed_count"),
            ("final_fn_max", True),
            ("final_fp_max", 67),
            ("final_f1_strictly_above", 0.5),
            ("test_metrics_may_inform_routing", True),
            ("mlx_applicable", True),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(config, **{field: value})

    def test_config_validation_authority_cannot_be_mutated_antigravity(self) -> None:
        """Break caught: changing the validation authority makes a weakened config valid."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity import (
            config_antigravity,
        )

        authority = config_antigravity._FROZEN_VALUES_ANTIGRAVITY
        original_alpha = authority["alpha"]
        try:
            with self.assertRaises(TypeError):
                authority["alpha"] = 0.05  # type: ignore[index]
        finally:
            if authority["alpha"] != original_alpha:
                authority["alpha"] = original_alpha  # type: ignore[index]
        with self.assertRaises(ValueError):
            replace(config_antigravity.FROZEN_CONFIG_ANTIGRAVITY, alpha=0.05)


if __name__ == "__main__":
    unittest.main()
