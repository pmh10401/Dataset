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


if __name__ == "__main__":
    unittest.main()
