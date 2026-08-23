from __future__ import annotations

import unittest

from experiments.exp465_ecg200_c8_external_validation_antigravity.interpolation_antigravity import (
    interpolate_to_140_antigravity as exp465_interpolate_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.interpolation_antigravity import (
    interpolate_to_140_antigravity as exp466_interpolate_antigravity,
)


class InterpolationAntigravityTest(unittest.TestCase):
    def test_export_is_exact_exp465_interpolator_antigravity(self) -> None:
        """Break caught: Exp466 wraps or copies the frozen Exp465 interpolator."""
        self.assertIs(exp466_interpolate_antigravity, exp465_interpolate_antigravity)


if __name__ == "__main__":
    unittest.main()
