"""Frozen Exp448 controlled-fork configuration behavior."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest


class ConfigAntigravityTests(unittest.TestCase):
    def test_frozen_identity_and_governance_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment_id, "Exp448")
        self.assertEqual(config.artifact_schema, "exp448")
        self.assertEqual(config.dataset_count, 1117)
        self.assertEqual(config.label_rows, 195900)
        self.assertEqual((config.source_count, config.stable_count, config.unstable_count), (30, 26, 4))
        self.assertEqual(config.baseline_confusion, (1915, 303, 2003, 191679))
        self.assertEqual(config.exp442_source_30_confusion, (1930, 318, 1988, 191664))
        self.assertEqual(config.bootstrap_seed, 44720260811)
        self.assertEqual(config.bootstrap_samples, 20000)
        self.assertIsNone(config.scientific_success_criterion)
        self.assertFalse(config.promotion_allowed)
        self.assertTrue(config.review_only)

    def test_config_cannot_be_mutated_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        with self.assertRaises(FrozenInstanceError):
            FROZEN_CONFIG_ANTIGRAVITY.dataset_count = 1  # type: ignore[misc]

    def test_constructor_rejects_every_scientific_or_governance_override_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.config_antigravity import (
            Exp448FrozenConfigAntigravity,
        )

        for overrides in (
            {"run_root": "/tmp/exp448_antigravity"},
            {"exp446_prediction_sha256": "0" * 64},
            {"dataset_count": 1116},
            {"bootstrap_seed": 1},
            {"review_only": 1},
            {"promotion_allowed": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Exp448FrozenConfigAntigravity(**overrides)


if __name__ == "__main__":
    unittest.main()
