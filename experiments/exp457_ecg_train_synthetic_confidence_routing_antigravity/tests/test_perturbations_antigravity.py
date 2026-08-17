from __future__ import annotations

from types import MappingProxyType
import unittest

import numpy as np


class PrimaryPerturbationsAntigravityTest(unittest.TestCase):
    def test_length_100_fixture_matches_exp417_math_and_exact_order_antigravity(self) -> None:
        """Break caught: onset, MAD scale, ramp endpoints, centering, or warp convention drifts."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.perturbations_antigravity import (
            build_primary_perturbations_antigravity,
        )

        values = np.arange(100, dtype=np.float64)
        original = values.copy()
        perturbations = build_primary_perturbations_antigravity(values)

        self.assertIsInstance(perturbations, MappingProxyType)
        self.assertEqual(
            tuple(perturbations),
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        )
        self.assertNotIn("secondary_spike", perturbations)
        self.assertNotIn("spike", perturbations)
        self.assertTrue(np.array_equal(values, original))

        robust_scale = 1.4826 * 25.0
        expected_level = original.copy()
        expected_level[40:] += 3.0 * robust_scale
        expected_drift = original.copy()
        expected_drift[40:] += np.linspace(0.0, 3.0 * robust_scale, 60)
        expected_amplitude = 49.5 + 1.5 * (original - 49.5)
        expected_period = original.copy()
        expected_period[40:] = 40.0 + np.arange(60, dtype=np.float64) / 1.25

        expected = {
            "level_shift": expected_level,
            "gradual_drift": expected_drift,
            "amplitude_scaling": expected_amplitude,
            "period_change": expected_period,
        }
        for name, wanted in expected.items():
            with self.subTest(name=name):
                actual = perturbations[name]
                self.assertEqual(actual.shape, (100,))
                self.assertFalse(actual.flags.writeable)
                self.assertTrue(np.array_equal(actual, wanted))
                self.assertEqual(actual.tobytes(), wanted.tobytes())
        self.assertEqual(perturbations["gradual_drift"][40], original[40])
        self.assertEqual(
            perturbations["gradual_drift"][-1], original[-1] + 3.0 * robust_scale
        )

    def test_centered_amplitude_uses_the_series_median_antigravity(self) -> None:
        """Break caught: amplitude is multiplied around zero instead of the series median."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.perturbations_antigravity import (
            build_primary_perturbations_antigravity,
        )

        values = np.linspace(10.0, 14.0, 100, dtype=np.float64)
        expected = 12.0 + 1.5 * (values - 12.0)
        self.assertTrue(
            np.array_equal(
                build_primary_perturbations_antigravity(values)["amplitude_scaling"],
                expected,
            )
        )

    def test_invalid_inputs_are_rejected_without_mutation_antigravity(self) -> None:
        """Break caught: non-vector, non-float, non-finite, or zero-scale inputs produce evidence."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.perturbations_antigravity import (
            build_primary_perturbations_antigravity,
        )

        invalid = (
            [0.0, 1.0],
            np.arange(100),
            np.arange(100, dtype=np.float64).reshape(10, 10),
            np.array([0.0], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
        )
        for values in invalid:
            with self.subTest(values=type(values).__name__), self.assertRaises(ValueError):
                build_primary_perturbations_antigravity(values)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
