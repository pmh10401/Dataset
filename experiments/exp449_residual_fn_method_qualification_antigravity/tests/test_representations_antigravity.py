"""Behavior tests for the four frozen Exp449 aeon representations.

These catch configuration drift (seed/settings/order), not source-text changes.
"""
from __future__ import annotations

import unittest

import numpy as np
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
    _rows_antigravity,
    make_representation_antigravity,
)


EXPECTED_NAMES_ANTIGRAVITY = (
    "minirocket_local_gap",
    "multirocket_local_gap",
    "hydra_local_gap",
    "hydra_multirocket_local_gap",
)


class RepresentationAntigravityTests(unittest.TestCase):
    """The breaks named below are observable feature-source regressions."""

    def setUp(self) -> None:
        self.rows = np.asarray(
            [[float((row + 1) * (column + 2) % 17) + column / 10 for column in range(16)] for row in range(6)],
            dtype=np.float32,
        )

    def test_each_fixed_representation_is_finite_repeatable_and_has_frozen_dimension_antigravity(self) -> None:
        """Catches a changed aeon seed/settings or an output-dimension drift."""
        expected_dimensions = {
            "minirocket_local_gap": 4032,
            "multirocket_local_gap": 8064,
            "hydra_local_gap": 1024,
            "hydra_multirocket_local_gap": 9088,
        }
        for name in EXPECTED_NAMES_ANTIGRAVITY:
            with self.subTest(name=name):
                first = make_representation_antigravity(name, 20260721)
                second = make_representation_antigravity(name, 20260721)
                first_values = first.fit_transform(self.rows)
                second_values = second.fit_transform(self.rows)
                np.testing.assert_array_equal(first_values, second_values)
                self.assertEqual(first_values.shape, (6, expected_dimensions[name]))
                self.assertTrue(np.isfinite(first_values).all())
                np.testing.assert_array_equal(first.transform(self.rows), first_values)

    def test_hydra_multirocket_concatenation_is_hydra_then_multirocket_antigravity(self) -> None:
        """Catches an accidental reverse/changed concatenation order."""
        hybrid = make_representation_antigravity("hydra_multirocket_local_gap", 20260721)
        actual = hybrid.fit_transform(self.rows)
        expected = np.concatenate((
            np.asarray(hybrid._hydra_scaler.transform(hybrid._hydra.transform(_rows_antigravity(self.rows))), dtype=float),
            np.asarray(hybrid._multirocket_scaler.transform(hybrid._multirocket.transform(_rows_antigravity(self.rows))), dtype=float),
        ), axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_unknown_or_boolean_seed_is_rejected_before_transformer_creation_antigravity(self) -> None:
        """Catches routing an unknown candidate or accepting bool as a numeric seed."""
        with self.assertRaisesRegex(ValueError, "representation"):
            make_representation_antigravity("unknown", 20260721)
        with self.assertRaisesRegex(ValueError, "seed"):
            make_representation_antigravity("minirocket_local_gap", True)


if __name__ == "__main__":
    unittest.main()
