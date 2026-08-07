import math
import unittest

import numpy as np

from experiments.exp438_frequency_complexity_evidence_antigravity.aspe_antigravity import (
    amplitude_sensitive_permutation_entropy_antigravity,
    ordinal_pattern_antigravity,
)


class Exp438AspeAntigravityTests(unittest.TestCase):
    def test_constant_signal_uses_explicit_safe_extension(self) -> None:
        result = amplitude_sensitive_permutation_entropy_antigravity(
            np.ones(16), embedding_dimension=3, delay=1
        )
        self.assertEqual(result.value, 0.0)
        self.assertTrue(result.flat_or_zero_weight)

    def test_monotone_signal_has_one_weighted_pattern_and_zero_entropy(self) -> None:
        result = amplitude_sensitive_permutation_entropy_antigravity(
            np.arange(8), embedding_dimension=3, delay=1
        )
        self.assertEqual(result.value, 0.0)
        self.assertFalse(result.flat_or_zero_weight)

    def test_alternating_literal_golden_vector_matches_weighted_entropy(self) -> None:
        # Normalized windows are [0,1,0], [1,0,1], [0,1,0]. Their sample-CV
        # weights are sqrt(3), sqrt(3)/2, sqrt(3), so probabilities are 0.8/0.2.
        expected = -(
            0.8 * math.log2(0.8) + 0.2 * math.log2(0.2)
        ) / math.log2(6.0)
        result = amplitude_sensitive_permutation_entropy_antigravity(
            [0.0, 2.0, 0.0, 2.0, 0.0], embedding_dimension=3, delay=1
        )
        self.assertAlmostEqual(result.value, expected, places=12)
        self.assertFalse(result.flat_or_zero_weight)

    def test_amplitude_changes_entropy_when_ordinal_patterns_are_unchanged(self) -> None:
        balanced = amplitude_sensitive_permutation_entropy_antigravity(
            [0.0, 2.0, 0.0, 2.0, 0.0], 3, 1
        )
        unbalanced = amplitude_sensitive_permutation_entropy_antigravity(
            [0.0, 1.0, 0.0, 10.0, 0.0], 3, 1
        )
        self.assertNotAlmostEqual(balanced.value, unbalanced.value, places=8)

    def test_ties_keep_original_position_order(self) -> None:
        self.assertEqual(ordinal_pattern_antigravity([1.0, 1.0, 0.0]), (2, 0, 1))
        self.assertEqual(ordinal_pattern_antigravity([1.0, 0.0, 1.0]), (1, 0, 2))

    def test_invalid_or_nonfinite_inputs_fail_closed(self) -> None:
        for values, dimension, delay in (
            ([1.0, np.nan, 2.0], 3, 1),
            ([1.0, 2.0], 3, 1),
            ([1.0, 2.0, 3.0], 1, 1),
            ([1.0, 2.0, 3.0], 3, 0),
        ):
            with self.subTest(values=values, dimension=dimension, delay=delay):
                with self.assertRaises(ValueError):
                    amplitude_sensitive_permutation_entropy_antigravity(
                        values, dimension, delay
                    )


if __name__ == "__main__":
    unittest.main()
