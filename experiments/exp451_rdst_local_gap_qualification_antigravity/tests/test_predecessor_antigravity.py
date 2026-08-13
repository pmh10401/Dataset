"""Task 4 RED/GREEN tests for the immutable Exp448 predecessor boundary."""

from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch


class PredecessorAntigravityTest(unittest.TestCase):
    def test_converts_descriptor_pinned_exp448_authority_without_alias_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import capture_exp448_authority_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity import verify_exp448_predecessor_antigravity

        frozen = capture_exp448_authority_antigravity()
        with patch("experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity.capture_exp448_authority_antigravity", return_value=frozen):
            first = verify_exp448_predecessor_antigravity()
            second = verify_exp448_predecessor_antigravity()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertTrue(dataclasses.is_dataclass(first))
        self.assertEqual(first.population, 1117)
        self.assertEqual(first.target_dataset_count, 48)
        self.assertEqual(first.target_test_rows, 72_300)
        self.assertEqual(first.label_access_count, 0)
        self.assertEqual(first.baseline_confusion, (1930, 314, 1988, 191668))
        self.assertEqual(first.target_family_fn, (246, 200, 88, 75, 63, 56, 56, 53))
        self.assertEqual(len(first.target_dataset_names), len(set(first.target_dataset_names)))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.population = 0  # type: ignore[misc]

    def test_rejects_semantically_drifted_capture_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.predecessor_antigravity import capture_exp448_authority_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity import verify_exp448_predecessor_antigravity

        frozen = capture_exp448_authority_antigravity()
        with patch.object(type(frozen), "population", 1116, create=True), patch("experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity.capture_exp448_authority_antigravity", return_value=frozen), self.assertRaises(ValueError):
            verify_exp448_predecessor_antigravity()


if __name__ == "__main__":
    unittest.main()
