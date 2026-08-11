"""Frozen Exp447 configuration behavior."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest


class ConfigAntigravityTests(unittest.TestCase):
    """These tests fail if Exp447's scientific/governance constants drift."""

    def test_frozen_identity_and_governance_antigravity(self) -> None:
        # Break caught: a later edit silently changes Exp447's evaluated population
        # or turns this descriptive experiment into a promotion decision.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment_id, "Exp447")
        self.assertEqual(config.dataset_count, 1117)
        self.assertEqual(config.label_rows, 195900)
        self.assertEqual((config.source_count, config.stable_count, config.unstable_count), (30, 26, 4))
        self.assertEqual(config.baseline_confusion, (1915, 303, 2003, 191679))
        self.assertEqual(config.exp442_source_30_confusion, (1930, 318, 1988, 191664))
        self.assertEqual(config.exp442_evaluation_sha256, "f432ffdf530b2e32bc7a62fec6c7d07cff2eff9f3049489b5e03136b445a33cf")
        self.assertEqual(config.exp442_evaluation_artifact_map_sha256, "e1155acf05e00935710243e875eaeb2d13047d6236d026ee93fe654fc33c670b")
        self.assertEqual(config.bootstrap_seed, 44720260811)
        self.assertEqual(config.bootstrap_samples, 20000)
        self.assertIsNone(config.scientific_success_criterion)
        self.assertFalse(config.promotion_allowed)
        self.assertTrue(config.review_only)

    def test_config_cannot_be_mutated_antigravity(self) -> None:
        # Break caught: callers can alter a frozen identity after import.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        with self.assertRaises(FrozenInstanceError):
            FROZEN_CONFIG_ANTIGRAVITY.dataset_count = 1  # type: ignore[misc]

    def test_constructor_rejects_every_scientific_or_governance_override_antigravity(self) -> None:
        # Break caught: a caller creates a second configuration with a different
        # root, source hash, count, bootstrap, or truthy integer governance flag.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.config_antigravity import (
            Exp447FrozenConfigAntigravity,
        )

        for overrides in (
            {"run_root": "/tmp/exp447_antigravity"},
            {"exp446_prediction_sha256": "0" * 64},
            {"dataset_count": 1116},
            {"bootstrap_seed": 1},
            {"review_only": 1},
            {"promotion_allowed": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Exp447FrozenConfigAntigravity(**overrides)


if __name__ == "__main__":
    unittest.main()
