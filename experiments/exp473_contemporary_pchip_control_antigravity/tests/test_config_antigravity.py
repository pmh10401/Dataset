"""Frozen identity checks for the Exp473 contemporary PCHIP control."""

from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.exp473_contemporary_pchip_control_antigravity.config_antigravity import (
    CONDITION_D_PRIME_EXP473_ANTIGRAVITY,
    EXP473_OFFICIAL_ROOT_ANTIGRAVITY,
    EXP473_SMOKE_ROOT_ANTIGRAVITY,
    FROZEN_EXP473_CONFIG_ANTIGRAVITY,
    P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY,
)


class Exp473ConfigTestsAntigravity(unittest.TestCase):
    def test_identity_population_and_isolated_roots_are_frozen(self) -> None:
        config = FROZEN_EXP473_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment_id, "Exp473")
        self.assertEqual((config.dataset_count, config.test_row_count), (1117, 195900))
        self.assertEqual(config.policy_id, "hard_plus_stable")
        self.assertEqual(config.score_threshold, 0.5)
        self.assertEqual(config.condition, CONDITION_D_PRIME_EXP473_ANTIGRAVITY)
        self.assertNotEqual(EXP473_SMOKE_ROOT_ANTIGRAVITY, EXP473_OFFICIAL_ROOT_ANTIGRAVITY)
        self.assertTrue(EXP473_SMOKE_ROOT_ANTIGRAVITY.is_absolute())
        self.assertTrue(EXP473_OFFICIAL_ROOT_ANTIGRAVITY.is_absolute())
        self.assertIn("antigravity", EXP473_SMOKE_ROOT_ANTIGRAVITY.name)
        self.assertIn("antigravity", EXP473_OFFICIAL_ROOT_ANTIGRAVITY.name)
        self.assertEqual(config.smoke_datasets, P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY)
        self.assertEqual(len(config.smoke_datasets), 15)
        self.assertTrue(
            {"Yoga_normal_2", "CricketY_normal_3", "FaceFour_normal_4"}
            <= set(config.smoke_datasets)
        )

    def test_d_prime_changes_only_execution_identity_not_condition_d_science(self) -> None:
        condition = CONDITION_D_PRIME_EXP473_ANTIGRAVITY
        self.assertEqual(
            (
                condition.control_id,
                condition.condition_id,
                condition.length_mode,
                condition.image_mode,
            ),
            ("D_prime", "D", "polyphase_pchip", "area"),
        )
        with self.assertRaisesRegex(ValueError, "approved design"):
            replace(condition, length_mode="linear")


if __name__ == "__main__":
    unittest.main()
