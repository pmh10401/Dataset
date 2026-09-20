from __future__ import annotations

import unittest

from experiments.exp466_ecg200_phase_registration_margin_antigravity.interpolation_antigravity import (
    interpolate_to_140_antigravity as exp466_interpolate_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.interpolation_antigravity import (
    interpolate_to_140_antigravity as exp468_interpolate_antigravity,
)


class InterpolationAntigravityTest(unittest.TestCase):
    def test_export_is_exact_exp466_interpolator_antigravity(self) -> None:
        """Break caught: Exp468 wraps or copies the frozen Exp466 interpolator."""
        self.assertIs(exp468_interpolate_antigravity, exp466_interpolate_antigravity)


if __name__ == "__main__":
    unittest.main()
