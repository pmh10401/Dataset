"""TDD coverage for deterministic Exp444 TRAIN omission masks."""
from __future__ import annotations

import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp444_train_only_stability_precision_gate_antigravity.masks_antigravity import (
    canonical_mask_identity_sha256_antigravity,
    generate_omission_masks_antigravity,
    omission_count_antigravity,
    required_survival_count_antigravity,
)


class Exp444MaskAntigravityTests(unittest.TestCase):
    def test_n_five_produces_five_unique_single_omission_masks_antigravity(self) -> None:
        masks = generate_omission_masks_antigravity("A", 32, (0, 1, 2, 3, 4))
        self.assertEqual(len(masks), 5)
        self.assertEqual(len({item.excluded_train_indices for item in masks}), 5)
        self.assertTrue(all(len(item.excluded_train_indices) == 1 for item in masks))
        self.assertEqual(required_survival_count_antigravity(len(masks)), 5)
        self.assertEqual(tuple(item.view_index for item in masks), tuple(range(5)))

    def test_large_group_caps_at_sixteen_masks_antigravity(self) -> None:
        masks = generate_omission_masks_antigravity("A", 32, tuple(range(160)))
        self.assertEqual(len(masks), 16)
        self.assertTrue(all(len(item.excluded_train_indices) == 16 for item in masks))
        self.assertEqual(required_survival_count_antigravity(16), 14)
        self.assertTrue(all(len(set(range(160)) - set(item.excluded_train_indices)) >= 4 for item in masks))

    def test_masks_are_order_independent_after_identity_validation_antigravity(self) -> None:
        forward = generate_omission_masks_antigravity("A", 32, tuple(range(20)))
        reverse = generate_omission_masks_antigravity("A", 32, tuple(reversed(range(20))))
        self.assertEqual(forward, reverse)

    def test_same_seed_reproduces_identity_and_dataset_length_separate_antigravity(self) -> None:
        first = generate_omission_masks_antigravity("A", 32, tuple(range(40)))
        second = generate_omission_masks_antigravity("A", 32, tuple(range(40)))
        other_dataset = generate_omission_masks_antigravity("B", 32, tuple(range(40)))
        other_length = generate_omission_masks_antigravity("A", 64, tuple(range(40)))
        self.assertEqual(first, second)
        self.assertNotEqual(
            tuple(item.identity_sha256 for item in first),
            tuple(item.identity_sha256 for item in other_dataset),
        )
        self.assertNotEqual(
            tuple(item.identity_sha256 for item in first),
            tuple(item.identity_sha256 for item in other_length),
        )
        self.assertEqual(FROZEN_CONFIG.mask_seed, 44420260810)

    def test_masks_are_unique_sorted_and_canonically_hashed_antigravity(self) -> None:
        train = (3, 7, 11, 19, 21, 30, 44, 57, 61, 80)
        masks = generate_omission_masks_antigravity("A", 32, train)
        self.assertEqual(len({item.excluded_train_indices for item in masks}), len(masks))
        for item in masks:
            self.assertEqual(item.excluded_train_indices, tuple(sorted(item.excluded_train_indices)))
            self.assertEqual(
                item.identity_sha256,
                canonical_mask_identity_sha256_antigravity(
                    "A", 32, train, item.excluded_train_indices
                ),
            )

    def test_exact_omit_and_survival_arithmetic_antigravity(self) -> None:
        self.assertEqual(omission_count_antigravity(5), 1)
        self.assertEqual(omission_count_antigravity(9), 1)
        self.assertEqual(omission_count_antigravity(10), 1)
        self.assertEqual(omission_count_antigravity(20), 2)
        self.assertEqual(required_survival_count_antigravity(1), 1)
        self.assertEqual(required_survival_count_antigravity(2), 2)
        self.assertEqual(required_survival_count_antigravity(8), 7)
        self.assertEqual(required_survival_count_antigravity(16), 14)

    def test_invalid_groups_cannot_violate_four_row_retain_contract_antigravity(self) -> None:
        with self.assertRaises(ValueError):
            generate_omission_masks_antigravity("A", 32, (0, 1, 2, 3))
        with self.assertRaises(ValueError):
            generate_omission_masks_antigravity("A", 32, (0, 1, 1, 2, 3))
        with self.assertRaises(ValueError):
            generate_omission_masks_antigravity("A", 32, (-1, 0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
