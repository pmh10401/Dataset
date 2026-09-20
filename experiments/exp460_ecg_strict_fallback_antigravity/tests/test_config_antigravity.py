from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import unittest


RULE_ORDER_ANTIGRAVITY = (
    "always_baseline",
    "always_candidate",
    "majority_8_of_10",
    "strict_10_of_10",
)

BASELINE_CONFUSION_ANTIGRAVITY = (609, 66, 837, 70788)
CANDIDATE_CONFUSION_ANTIGRAVITY = (617, 66, 829, 70788)


class ConfigAntigravityTest(unittest.TestCase):
    def test_import_surface_and_frozen_values_antigravity(self) -> None:
        from experiments.exp460_ecg_strict_fallback_antigravity.config_antigravity import (
            Exp460ConfigAntigravity,
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        self.assertEqual(
            tuple(field.name for field in fields(Exp460ConfigAntigravity)),
            (
                "experiment",
                "kind",
                "dataset",
                "rule_order",
                "official_decision_rule",
                "strict_failure_route",
                "hydra_seeds",
                "folds",
                "period_change_auc_min",
                "baseline_confusion",
                "candidate_confusion",
                "exp459_frozen_authority_digest",
                "exp459_official_artifact_map_sha256",
                "exp459_decision_sha256",
                "label_query_count",
                "runtime_executable",
            ),
        )
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.experiment, "Exp460")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.kind, "ecg_strict_fallback")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.dataset, "ECG5000_normal_2")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.rule_order, RULE_ORDER_ANTIGRAVITY)
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.official_decision_rule,
            "strict_10_of_10_only",
        )
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.strict_failure_route, "baseline")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds, (20260721, 20261721))
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.folds, 5)
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_min, 0.85)
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.baseline_confusion,
            BASELINE_CONFUSION_ANTIGRAVITY,
        )
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.candidate_confusion,
            CANDIDATE_CONFUSION_ANTIGRAVITY,
        )
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.exp459_frozen_authority_digest,
            "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
        )
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.exp459_official_artifact_map_sha256,
            "a30fbe7ed34f88c1ab9b7fd40963edebbf1a4127d8d64c99141224cb2dbb106d",
        )
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.exp459_decision_sha256,
            "527318cacdbc63905ab58237d793803eb286192456eb007362e6bed11c8f8716",
        )
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.label_query_count, 0)
        self.assertEqual(
            FROZEN_CONFIG_ANTIGRAVITY.runtime_executable,
            "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        )
        with self.assertRaises(FrozenInstanceError):
            FROZEN_CONFIG_ANTIGRAVITY.dataset = "ECG5000_normal_1"  # type: ignore[misc]

    def test_config_rejects_mutated_exact_values_antigravity(self) -> None:
        from experiments.exp460_ecg_strict_fallback_antigravity.config_antigravity import (
            Exp460ConfigAntigravity,
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        valid = {
            field.name: getattr(FROZEN_CONFIG_ANTIGRAVITY, field.name)
            for field in fields(type(FROZEN_CONFIG_ANTIGRAVITY))
        }
        mutations = (
            ("experiment", "Exp461"),
            ("kind", "ecg_lenient_fallback"),
            ("dataset", "ECG5000_normal_1"),
            ("rule_order", list(RULE_ORDER_ANTIGRAVITY)),
            ("rule_order", RULE_ORDER_ANTIGRAVITY[:-1]),
            ("rule_order", RULE_ORDER_ANTIGRAVITY[::-1]),
            ("official_decision_rule", "majority_8_of_10"),
            ("strict_failure_route", "candidate"),
            ("hydra_seeds", (20260721, 20261722)),
            ("folds", 4),
            ("period_change_auc_min", 0.9),
            ("baseline_confusion", (610, 66, 837, 70788)),
            ("candidate_confusion", (617, 66, 830, 70788)),
            ("exp459_frozen_authority_digest", "0" * 64),
            ("exp459_official_artifact_map_sha256", "1" * 64),
            ("exp459_decision_sha256", "2" * 64),
            ("label_query_count", True),
            ("label_query_count", 1),
            ("runtime_executable", "/tmp/python"),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name, value=value), self.assertRaises(
                ValueError
            ):
                Exp460ConfigAntigravity(**{**valid, field_name: value})


if __name__ == "__main__":
    unittest.main()
