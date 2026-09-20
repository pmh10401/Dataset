from __future__ import annotations

import unittest

import numpy as np


def independent_prefix_ramp_reference_antigravity(values: np.ndarray) -> float:
    row = np.array(values, dtype=np.float64, copy=True)
    prefix = row[0:56]
    center = float(np.median(prefix))
    scale = float(1.4826 * np.median(np.abs(prefix - center)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("reference scale must be positive")
    z = (row - center) / scale
    basis = np.zeros(140, dtype=np.float64)
    basis[56:] = np.linspace(0.0, 1.0, 84, dtype=np.float64)
    basis /= float(np.linalg.norm(basis))
    return float(np.dot(z, basis))


class RampFeaturesAntigravityTest(unittest.TestCase):
    def test_prefix_ramp_matches_independent_reference_and_contract_antigravity(self) -> None:
        """Break caught: prefix scale, onset, basis normalization, or signed dot drifts."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.ramp_features_antigravity import (
            prefix_ramp_score_antigravity,
        )

        time = np.linspace(-2.0, 3.0, 140, dtype=np.float64)
        values = (
            0.4 * np.sin(time * 2.1)
            + 0.1 * time * time
            + np.where(np.arange(140) >= 56, 0.8 * (np.arange(140) - 56) / 83.0, 0.0)
            + 0.2
        ).astype(np.float32)
        original = np.array(values, copy=True)
        expected = independent_prefix_ramp_reference_antigravity(values)

        actual = prefix_ramp_score_antigravity(values)

        self.assertIs(type(actual), float)
        self.assertTrue(np.isfinite(actual))
        self.assertAlmostEqual(actual, expected, places=12)
        np.testing.assert_array_equal(values, original)

    def test_prefix_ramp_rejects_wrong_shape_nonfinite_and_zero_mad_antigravity(self) -> None:
        """Break caught: malformed rows or a non-identifiable prefix enter scoring."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.ramp_features_antigravity import (
            prefix_ramp_score_antigravity,
        )

        good = np.linspace(-1.0, 1.0, 140, dtype=np.float64)
        invalid = (
            good[:139],
            good.reshape(1, 140),
            np.arange(140, dtype=np.int64),
            np.full(140, 1.0, dtype=np.float64),
            np.r_[np.nan, good[1:]],
            np.r_[np.zeros(56, dtype=np.float64), np.linspace(0.0, 1.0, 84)],
        )
        for values in invalid:
            with self.subTest(shape=getattr(values, "shape", None), dtype=getattr(values, "dtype", None)):
                with self.assertRaises(ValueError):
                    prefix_ramp_score_antigravity(values)

    def test_prefix_ramp_is_deterministic_and_does_not_alias_input_antigravity(self) -> None:
        """Break caught: mutable views, hidden state, or a changed seed affect the feature."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.ramp_features_antigravity import (
            prefix_ramp_score_antigravity,
        )

        values = (np.sin(np.linspace(0.0, 8.0, 140)) + np.linspace(0.0, 1.0, 140)).astype(np.float64)
        first = prefix_ramp_score_antigravity(values)
        values[0] = 99.0
        second = prefix_ramp_score_antigravity(values)
        self.assertNotEqual(first, second)
        restored = values.copy()
        restored[0] = np.sin(0.0)
        self.assertEqual(prefix_ramp_score_antigravity(restored), first)


if __name__ == "__main__":
    unittest.main()
