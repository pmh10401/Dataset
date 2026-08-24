from __future__ import annotations

import unittest

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.synthetic_antigravity import (
    primary_rows_antigravity as exp466_primary_rows_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.synthetic_antigravity import (
    primary_rows_antigravity as exp468_primary_rows_antigravity,
)


_CHANGE_ORDER_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


class SyntheticAntigravityTest(unittest.TestCase):
    def test_export_is_exact_exp466_primary_generator_antigravity(self) -> None:
        """Break caught: Exp468 reimplements a frozen Exp466 synthetic change."""
        self.assertIs(exp468_primary_rows_antigravity, exp466_primary_rows_antigravity)

    def test_all_four_frozen_scopes_remain_available_antigravity(self) -> None:
        """Break caught: period_change or another frozen scope is omitted or reordered."""
        row = np.linspace(-2.0, 3.0, 140, dtype=np.float64)
        changes = exp468_primary_rows_antigravity(row)

        self.assertEqual(tuple(changes), _CHANGE_ORDER_ANTIGRAVITY)
        for values in changes.values():
            self.assertEqual(values.shape, (140,))
            self.assertEqual(values.dtype, np.dtype(np.float64))
            self.assertFalse(values.flags.writeable)
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
