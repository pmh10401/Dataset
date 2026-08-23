from __future__ import annotations

from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.synthetic_antigravity import (
    primary_rows_antigravity,
)


_CHANGE_ORDER_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


class SyntheticAntigravityTest(unittest.TestCase):
    def test_four_exp464_formulas_are_exact_owned_and_read_only_antigravity(self) -> None:
        """Break caught: a frozen change formula/order or output ownership contract drifts."""
        row = np.linspace(-2.0, 3.0, 140, dtype=np.float64)
        changes = primary_rows_antigravity(row)

        self.assertIsInstance(changes, MappingProxyType)
        self.assertEqual(tuple(changes), _CHANGE_ORDER_ANTIGRAVITY)

        scale = 1.8665827338129495
        expected_level = np.array(row, dtype=np.float64, copy=True)
        expected_level[56:] += 3.0 * scale
        expected_drift = np.array(row, dtype=np.float64, copy=True)
        expected_drift[56:] += np.linspace(0.0, 3.0 * scale, 84, dtype=np.float64)
        expected_amplitude = 0.4999999999999998 + 1.5 * (
            row - 0.4999999999999998
        )
        expected_period = np.array(row, dtype=np.float64, copy=True)
        expected_period[56:] = np.interp(
            np.arange(84, dtype=np.float64) / 1.25,
            np.arange(84, dtype=np.float64),
            row[56:],
        )
        expected = {
            "level_shift": expected_level,
            "gradual_drift": expected_drift,
            "amplitude_scaling": expected_amplitude,
            "period_change": expected_period,
        }

        for name in _CHANGE_ORDER_ANTIGRAVITY:
            actual = changes[name]
            np.testing.assert_allclose(actual, expected[name], rtol=0.0, atol=0.0)
            self.assertEqual(actual.dtype, np.dtype(np.float64))
            self.assertTrue(actual.flags.owndata)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertFalse(actual.flags.writeable)
            self.assertTrue(np.isfinite(actual).all())
            self.assertFalse(np.shares_memory(actual, row))

        np.testing.assert_allclose(
            changes["period_change"][56:],
            np.interp(
                np.arange(84) / 1.25,
                np.arange(84),
                np.linspace(-2.0, 3.0, 140)[56:],
            ),
            rtol=0.0,
            atol=0.0,
        )
        with self.assertRaises(TypeError):
            changes["extra"] = row  # type: ignore[index]

    def test_invalid_rows_are_rejected_antigravity(self) -> None:
        """Break caught: aliases, wrong geometry, nonfinite rows, or zero scale are accepted."""
        invalid_rows = (
            np.linspace(-2.0, 3.0, 140, dtype=np.float32),
            np.linspace(-2.0, 3.0, 139, dtype=np.float64),
            np.linspace(-2.0, 3.0, 140, dtype=np.float64).reshape(1, 140),
            np.full(140, 1.0, dtype=np.float64),
            np.concatenate((np.zeros(139, dtype=np.float64), np.array([np.nan]))),
        )
        for row in invalid_rows:
            with self.subTest(shape=row.shape, dtype=row.dtype):
                with self.assertRaises(ValueError):
                    primary_rows_antigravity(row)
        with self.assertRaises(TypeError):
            primary_rows_antigravity(list(np.linspace(-2.0, 3.0, 140)))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
