from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import unittest


METHOD_ORDER_ANTIGRAVITY = (
    "hydra_local_gap",
    "period_signature_local_gap",
    "hydra_period_rank_max",
)
PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_preserves_the_exact_task2_contract_antigravity(self) -> None:
        from experiments.exp461_ecg_period_signature_qualification_antigravity.config_antigravity import (
            Exp461ConfigAntigravity,
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        self.assertEqual(
            tuple(field.name for field in fields(Exp461ConfigAntigravity)),
            (
                "experiment",
                "kind",
                "run_identity",
                "official_result_root",
                "runtime_executable",
                "approved_design_sha256",
                "preregistration_sha256",
                "runbook_sha256",
                "dataset",
                "data_split",
                "normal2_train_rows",
                "normal2_width",
                "hydra_seeds",
                "folds",
                "alpha",
                "knn_k",
                "kernels",
                "groups",
                "primary_types",
                "method_order",
                "primary_candidate",
                "per_type_auc_min",
                "pooled_auc_min",
                "normal_wilson_upper_max",
                "predecessor_experiment",
                "predecessor_truth_commit",
                "predecessor_run_root",
                "predecessor_artifact_map_sha256",
                "predecessor_dashboard_snapshot_sha256",
                "predecessor_decision_sha256",
                "predecessor_status",
                "predecessor_strict_route",
                "predecessor_passed_record_count",
                "predecessor_failed_fold_count",
                "predecessor_worst_period_change_auc",
                "label_query_count_must_remain",
                "official_command_invocations_must_remain",
            ),
        )
        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp461")
        self.assertEqual(config.kind, "ecg_period_signature_qualification")
        self.assertEqual(
            config.run_identity,
            "exp461-ecg-period-signature-qualification-antigravity-20260820",
        )
        self.assertEqual(
            config.official_result_root,
            "/Users/mac/Documents/Dataset/exp461_ecg_period_signature_qualification_antigravity_20260820",
        )
        self.assertEqual(
            config.runtime_executable,
            "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        )
        self.assertEqual(
            config.approved_design_sha256,
            "3b1d6bc519a4acc000f45f30ba8ae5d610dcd0d2aa5e1cb61bd53837ba28f2c2",
        )
        self.assertEqual(
            config.preregistration_sha256,
            "9595d1f898b309ac899c821867a2f38e6f0eb24f147e6888970e1095534490cb",
        )
        self.assertEqual(
            config.runbook_sha256,
            "3c9ce4c884ae0f5d7c9551b508a8e3092899f481fc3bc1a14adbd92f81f87807",
        )
        self.assertEqual(config.dataset, "ECG5000_normal_2")
        self.assertEqual(config.data_split, "TRAIN_only")
        self.assertEqual(config.normal2_train_rows, 1413)
        self.assertEqual(config.normal2_width, 140)
        self.assertEqual(config.hydra_seeds, (20260721, 20261721))
        self.assertEqual(config.folds, 5)
        self.assertEqual(config.alpha, 0.005)
        self.assertEqual(config.knn_k, 3)
        self.assertEqual(config.kernels, 8)
        self.assertEqual(config.groups, 64)
        self.assertEqual(config.primary_types, PRIMARY_TYPES_ANTIGRAVITY)
        self.assertEqual(config.method_order, METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(config.primary_candidate, "hydra_period_rank_max")
        self.assertEqual(config.per_type_auc_min, 0.85)
        self.assertEqual(config.pooled_auc_min, 0.90)
        self.assertEqual(config.normal_wilson_upper_max, 0.05)
        self.assertEqual(config.predecessor_experiment, "Exp460")
        self.assertEqual(
            config.predecessor_truth_commit,
            "67d39f5c0a75e061e2d107fc9a9f58f52e76c2f1",
        )
        self.assertEqual(
            config.predecessor_run_root,
            "/Users/mac/Documents/Dataset/exp460_ecg_strict_fallback_antigravity_20260820",
        )
        self.assertEqual(
            config.predecessor_artifact_map_sha256,
            "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
        )
        self.assertEqual(
            config.predecessor_dashboard_snapshot_sha256,
            "fea8b4e43d256feddef48545e201f674bfe39fe85559050ed405f4e996709ee5",
        )
        self.assertEqual(
            config.predecessor_decision_sha256,
            "0859be73fb8b7c77cc803d67745ecd1bd42766ea5e3c005ce1bb9d2d3e6342c1",
        )
        self.assertEqual(
            config.predecessor_status,
            "fallback_triggered_performance_not_improved",
        )
        self.assertEqual(config.predecessor_strict_route, "baseline")
        self.assertEqual(config.predecessor_passed_record_count, 8)
        self.assertEqual(config.predecessor_failed_fold_count, 2)
        self.assertEqual(config.predecessor_worst_period_change_auc, 0.8004124541019063)
        self.assertEqual(config.label_query_count_must_remain, 0)
        self.assertEqual(config.official_command_invocations_must_remain, 0)

        with self.assertRaises(FrozenInstanceError):
            config.primary_candidate = "hydra_local_gap"  # type: ignore[misc]

    def test_config_rejects_mutated_values_types_or_order_antigravity(self) -> None:
        from experiments.exp461_ecg_period_signature_qualification_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        class ExactStrSubclassAntigravity(str):
            pass

        config = FROZEN_CONFIG_ANTIGRAVITY
        mutations = (
            ("experiment", "Exp460"),
            ("experiment", ExactStrSubclassAntigravity("Exp461")),
            ("kind", "other"),
            ("run_identity", "exp461-other"),
            ("official_result_root", "/tmp/other"),
            ("runtime_executable", "/usr/bin/python3"),
            ("approved_design_sha256", "0" * 64),
            ("preregistration_sha256", "1" * 64),
            ("runbook_sha256", "2" * 64),
            ("dataset", "ECG5000_normal_1"),
            ("data_split", "TEST_only"),
            ("normal2_train_rows", 1412),
            ("normal2_width", 141),
            ("hydra_seeds", tuple(reversed(config.hydra_seeds))),
            ("hydra_seeds", (20260721, 20261721, 20262721)),
            ("folds", True),
            ("folds", 4),
            ("alpha", True),
            ("alpha", 0.01),
            ("knn_k", True),
            ("knn_k", 4),
            ("kernels", True),
            ("kernels", 16),
            ("groups", True),
            ("groups", 32),
            ("primary_types", tuple(reversed(config.primary_types))),
            ("method_order", tuple(reversed(config.method_order))),
            ("primary_candidate", "period_signature_local_gap"),
            ("per_type_auc_min", True),
            ("per_type_auc_min", 0.84),
            ("pooled_auc_min", True),
            ("pooled_auc_min", 0.91),
            ("normal_wilson_upper_max", True),
            ("normal_wilson_upper_max", 0.0500000001),
            ("predecessor_experiment", "Exp459"),
            ("predecessor_truth_commit", "0" * 40),
            ("predecessor_run_root", "/tmp/exp460"),
            ("predecessor_artifact_map_sha256", "3" * 64),
            ("predecessor_dashboard_snapshot_sha256", "4" * 64),
            ("predecessor_decision_sha256", "5" * 64),
            ("predecessor_status", "stable"),
            ("predecessor_strict_route", "candidate"),
            ("predecessor_passed_record_count", True),
            ("predecessor_passed_record_count", 9),
            ("predecessor_failed_fold_count", True),
            ("predecessor_failed_fold_count", 1),
            ("predecessor_worst_period_change_auc", 0.85),
            ("label_query_count_must_remain", True),
            ("label_query_count_must_remain", 1),
            ("official_command_invocations_must_remain", True),
            ("official_command_invocations_must_remain", 1),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name, value=value), self.assertRaises(
                ValueError
            ):
                replace(config, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
