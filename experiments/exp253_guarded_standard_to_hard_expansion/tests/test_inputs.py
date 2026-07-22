from __future__ import annotations

import unittest

from experiments.exp253_guarded_standard_to_hard_expansion.config import ALPHA, MINI_VARIANTS
from experiments.exp253_guarded_standard_to_hard_expansion.inputs import validate_indices


class FrozenInputContractTest(unittest.TestCase):
    def test_required_mini_variants_are_frozen(self) -> None:
        self.assertEqual(ALPHA, 0.005)
        self.assertEqual(
            MINI_VARIANTS,
            {
                "mini1024": ("kernels_1024_seed_intersection", 0.005),
                "mini4096": ("kernels_4096_seed_intersection", 0.005),
            },
        )

    def test_rejects_out_of_range_index(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside TEST range"):
            validate_indices("demo", 3, {3})


if __name__ == "__main__":
    unittest.main()
