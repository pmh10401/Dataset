from hashlib import sha256
import unittest

import numpy as np

from experiments.exp438_frequency_complexity_evidence_antigravity.features_antigravity import (
    fft_split_antigravity as fft_split_exp438_antigravity,
)
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.features_antigravity import (
    amplitude_complexity_vector_antigravity,
    decompose_at_depth_antigravity,
    fft_split_antigravity,
    lower_median_depth_antigravity,
    maximum_depth_antigravity,
    preferred_depth_from_scores_antigravity,
    select_train_depth_antigravity,
)


class Exp439FeaturesAntigravityTests(unittest.TestCase):
    def test_real_chinatown_row_accepts_float32_reconstruction_rounding(self) -> None:
        values = np.asarray(
            [
                554.0,
                267.0,
                172.0,
                109.0,
                106.0,
                77.0,
                23.0,
                43.0,
                69.0,
                159.0,
                365.0,
                706.0,
                1378.0,
                1471.0,
                1347.0,
                1300.0,
                1082.0,
                1200.0,
                1350.0,
                1211.0,
                936.0,
                813.0,
                713.0,
                401.0,
            ],
            dtype="<f4",
        )
        self.assertEqual(
            sha256(values.tobytes()).hexdigest(),
            "33bd5923753b1a3b6e4903ad9e8405d0c761a08f432b7a76c2177921034702a7",
        )

        low, high = fft_split_antigravity(values, depth=1)

        self.assertEqual(low.shape, values.shape)
        self.assertEqual(high.shape, values.shape)
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(high).all())

    def test_four_ulp_component_corruption_is_rejected(self) -> None:
        from experiments.exp439_scale_aware_reconstruction_repair_antigravity.features_antigravity import (
            _validate_reconstruction_antigravity,
        )

        original = np.full(24, 1000.0, dtype=np.float32)
        low = original.copy()
        high = np.zeros_like(original)
        four_ulp = np.float32(4.0 * np.spacing(np.float32(1000.0)))
        low[0] = np.float32(low[0] + four_ulp)

        with self.assertRaisesRegex(
            ValueError,
            "decomposition reconstruction tolerance failed",
        ):
            _validate_reconstruction_antigravity(original, low, high)

    def test_maximum_depth_is_length_bound_and_capped(self) -> None:
        self.assertEqual(maximum_depth_antigravity(16), 1)
        self.assertEqual(maximum_depth_antigravity(64), 3)
        self.assertEqual(maximum_depth_antigravity(1024), 5)
        with self.assertRaisesRegex(ValueError, "minimum raw length"):
            maximum_depth_antigravity(15)

    def test_fft_cutoff_is_exact_and_reconstruction_preserves_length(self) -> None:
        rng = np.random.default_rng(43801)
        original = rng.normal(size=64).astype(np.float32)
        before = original.copy()
        low, high = fft_split_antigravity(original, depth=2)
        self.assertEqual(low.dtype, np.float32)
        self.assertEqual(high.dtype, np.float32)
        self.assertEqual(low.shape, original.shape)
        self.assertEqual(high.shape, original.shape)
        self.assertFalse(low.flags.writeable)
        self.assertFalse(high.flags.writeable)
        np.testing.assert_array_equal(original, before)
        self.assertLessEqual(float(np.max(np.abs(original - (low + high)))), 1e-5)
        cutoff = max(1, (64 // 2) // (2**2))
        spectrum = np.fft.rfft(low.astype(np.float64))
        self.assertLessEqual(float(np.max(np.abs(spectrum[cutoff + 1 :]))), 1e-5)

    def test_exp438_pass_input_keeps_identical_component_bytes(self) -> None:
        original = np.random.default_rng(43801).normal(size=64).astype(np.float32)

        old_low, old_high = fft_split_exp438_antigravity(original, depth=2)
        new_low, new_high = fft_split_antigravity(original, depth=2)

        self.assertEqual(new_low.tobytes(), old_low.tobytes())
        self.assertEqual(new_high.tobytes(), old_high.tobytes())

    def test_invalid_depth_and_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            fft_split_antigravity(np.arange(16), depth=2)
        with self.assertRaises(ValueError):
            fft_split_antigravity([0.0] * 15 + [np.inf], depth=1)

    def test_first_nondecrease_rule_and_lower_median_are_exact(self) -> None:
        self.assertEqual(preferred_depth_from_scores_antigravity([0.8]), 1)
        self.assertEqual(preferred_depth_from_scores_antigravity([0.8, 0.6, 0.61]), 2)
        self.assertEqual(preferred_depth_from_scores_antigravity([0.8, 0.6, 0.4]), 3)
        self.assertEqual(preferred_depth_from_scores_antigravity([0.8, 0.7999995]), 1)
        self.assertEqual(lower_median_depth_antigravity([4, 1, 3, 2]), 2)
        self.assertEqual(lower_median_depth_antigravity([3, 1, 2]), 2)

    def test_aspe_feature_vector_is_original_low_high_by_delay(self) -> None:
        original = np.sin(np.linspace(0.0, 8.0 * np.pi, 64)).astype(np.float32)
        low, high = fft_split_antigravity(original, depth=2)
        values, flags = amplitude_complexity_vector_antigravity(original, low, high)
        self.assertEqual(values.shape, (9,))
        self.assertEqual(values.dtype, np.float32)
        self.assertFalse(values.flags.writeable)
        self.assertEqual(len(flags), 9)
        self.assertTrue(np.isfinite(values).all())

    def test_decomposition_contract_contains_reconstructing_components(self) -> None:
        original = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
        evidence = decompose_at_depth_antigravity(original, 2)
        self.assertEqual(evidence.depth, 2)
        self.assertLessEqual(
            float(np.max(np.abs(original - (evidence.low + evidence.high)))),
            1e-5,
        )
        self.assertEqual(evidence.amplitude_complexity.shape, (9,))

    def test_held_out_train_row_cannot_influence_its_own_depth(self) -> None:
        base = np.stack(
            [
                np.sin(np.linspace(0.0, (index + 1) * np.pi, 64))
                for index in range(5)
            ]
        ).astype(np.float32)
        first = select_train_depth_antigravity(base, excluded_row_index=2)
        changed = base.copy()
        changed[2] = np.random.default_rng(9).normal(size=64)
        second = select_train_depth_antigravity(changed, excluded_row_index=2)
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            select_train_depth_antigravity(base, excluded_row_index=8)


if __name__ == "__main__":
    unittest.main()
