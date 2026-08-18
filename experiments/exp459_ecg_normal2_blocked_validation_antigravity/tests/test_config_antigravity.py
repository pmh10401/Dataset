from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_preserves_the_approved_blocked_validation_contract_antigravity(
        self,
    ) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp459")
        self.assertEqual(
            config.approved_design_sha256,
            "47d530a0cbe7f7cdb0b85f2c0c546b7ede158d8a85824eba8c8caece16b2e275",
        )
        self.assertEqual(config.dataset, "ECG5000_normal_2")
        self.assertEqual(config.kind, "ecg_normal2_blocked_validation")
        self.assertEqual(
            config.run_identity,
            "exp459-ecg-normal2-blocked-validation-antigravity-20260818",
        )
        self.assertEqual(
            config.official_result_root,
            "/Users/mac/Documents/Dataset/exp459_ecg_normal2_blocked_validation_antigravity_20260818",
        )
        self.assertEqual(
            config.runtime_executable,
            "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        )
        self.assertEqual(
            config.runtime_prefix,
            "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314",
        )
        self.assertEqual(config.runtime_python_version, "3.14.6")
        self.assertEqual(config.runtime_numpy_version, "2.3.5")
        self.assertEqual(config.runtime_aeon_version, "1.4.0")
        self.assertEqual(config.predecessor_experiment, "Exp458")
        self.assertEqual(
            config.predecessor_reviewed_commit,
            "fc530cd2b392394aa7ffbd1408bc06d5b1b7c806",
        )
        self.assertEqual(
            config.predecessor_artifact_map_sha256,
            "7ef36f232db5d2f2396f3dcdeda15af2e93c34e18a6e05dc411e37fe855737f9",
        )
        self.assertEqual(config.blocked_fold_planner, "numpy.arange+numpy.array_split")
        self.assertEqual(config.hydra_seeds, (20260721, 20261721))
        self.assertEqual(config.folds, 5)
        self.assertEqual(config.query_sizes, (283, 283, 283, 282, 282))
        self.assertEqual(
            config.query_ranges,
            ((0, 282), (283, 565), (566, 848), (849, 1130), (1131, 1412)),
        )
        self.assertEqual(config.fit_sizes, (1130, 1130, 1130, 1131, 1131))
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
        self.assertEqual(config.normal2_train_rows, 1413)
        self.assertEqual(config.normal2_width, 140)
        self.assertEqual(config.structure_gate, "added_count > removed_count")
        self.assertIs(config.finite_required, True)
        self.assertIs(config.test_metrics_may_inform_routing, False)
        self.assertIs(config.mlx_applicable, False)

        with self.assertRaises(FrozenInstanceError):
            config.pooled_auc_min = 0.91  # type: ignore[misc]

    def test_config_rejects_any_mutated_value_type_or_order_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        class ExactStrSubclassAntigravity(str):
            pass

        config = FROZEN_CONFIG_ANTIGRAVITY
        mutations = (
            ("experiment", "Exp458"),
            ("experiment", ExactStrSubclassAntigravity("Exp459")),
            ("approved_design_sha256", "0" * 64),
            ("dataset", "ECG5000_normal_1"),
            ("kind", "other"),
            ("run_identity", "exp459-other"),
            ("official_result_root", "/Users/mac/Documents/Dataset/other"),
            ("runtime_executable", "/usr/bin/python3"),
            ("runtime_prefix", "/Users/mac/Documents/Dataset/venvs/other"),
            ("runtime_python_version", "3.14.7"),
            ("runtime_numpy_version", "2.4.6"),
            ("runtime_aeon_version", "1.4.1"),
            ("predecessor_experiment", "Exp457"),
            ("predecessor_reviewed_commit", "0" * 40),
            ("predecessor_artifact_map_sha256", "1" * 64),
            ("blocked_fold_planner", "shuffle"),
            ("hydra_seeds", tuple(reversed(config.hydra_seeds))),
            ("hydra_seeds", (20260721, 20261721, 20262721)),
            ("folds", True),
            ("folds", 4),
            ("query_sizes", tuple(reversed(config.query_sizes))),
            ("query_ranges", tuple(reversed(config.query_ranges))),
            ("fit_sizes", tuple(reversed(config.fit_sizes))),
            ("primary_types", tuple(reversed(config.primary_types))),
            ("primary_types", config.primary_types + ("secondary_spike",)),
            ("per_type_auc_min", True),
            ("per_type_auc_min", 0.84),
            ("pooled_auc_min", True),
            ("pooled_auc_min", 0.91),
            ("normal_wilson_upper_max", 0.0500000001),
            ("normal_wilson_upper_max", True),
            ("normal2_train_rows", 1412),
            ("normal2_width", 141),
            ("structure_gate", "added_count >= removed_count"),
            ("finite_required", 1),
            ("test_metrics_may_inform_routing", True),
            ("mlx_applicable", True),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(config, **{field: value})


if __name__ == "__main__":
    unittest.main()
