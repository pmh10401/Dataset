from __future__ import annotations

import unittest

from experiments.exp305_wavelet_reconstruction_cross_confirmation.routing import (
    policies_for_support,
)


class Exp305RoutingTests(unittest.TestCase):
    def test_reconstruction_and_wavelet_cross_confirm_zero_existing_support(self):
        policies = policies_for_support(
            existing_support_count=0,
            reconstruction_support=True,
            wavelet_support=True,
        )
        self.assertIn("reconstruction_wavelet_cross_confirmed", policies)
        self.assertIn("six_source_2of6", policies)
        self.assertIn("existing_1of4_or_reconstruction_wavelet", policies)

    def test_reconstruction_alone_is_not_cross_confirmed(self):
        policies = policies_for_support(
            existing_support_count=0,
            reconstruction_support=True,
            wavelet_support=False,
        )
        self.assertIn("reconstruction_only_residual", policies)
        self.assertNotIn("reconstruction_wavelet_cross_confirmed", policies)
        self.assertNotIn("six_source_2of6", policies)

    def test_wavelet_alone_is_reported_but_not_two_source_confirmed(self):
        policies = policies_for_support(
            existing_support_count=0,
            reconstruction_support=False,
            wavelet_support=True,
        )
        self.assertIn("wavelet_only_residual", policies)
        self.assertNotIn("reconstruction_wavelet_cross_confirmed", policies)
        self.assertNotIn("six_source_2of6", policies)

    def test_existing_support_and_wavelet_count_as_two_of_six(self):
        policies = policies_for_support(
            existing_support_count=1,
            reconstruction_support=False,
            wavelet_support=True,
        )
        self.assertIn("verified_1of4", policies)
        self.assertIn("six_source_2of6", policies)


if __name__ == "__main__":
    unittest.main()
