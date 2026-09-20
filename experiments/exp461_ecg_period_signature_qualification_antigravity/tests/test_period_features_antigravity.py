from __future__ import annotations

import unittest

import numpy as np


def reference_signature_antigravity(values: np.ndarray) -> np.ndarray:
    pieces: list[float] = []
    for view in (
        values[0:140],
        values[0:70],
        values[70:140],
    ):
        center = float(np.median(view))
        z = np.asarray(view, dtype=np.float64) - center
        scale = float(1.4826 * np.median(np.abs(z)))
        z = z / scale
        power = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
        positive = power[1:]
        total = float(np.sum(positive))
        pieces.extend(float(np.sum(group) / total) for group in np.array_split(positive, 16))
        denominator = float(np.dot(z, z))
        pieces.extend(
            float(np.dot(z[:-lag], z[lag:]) / denominator)
            for lag in range(1, 25)
        )
    return np.asarray(pieces, dtype=np.float64)


def fixture_row_antigravity() -> np.ndarray:
    x = np.linspace(0.0, 6.0 * np.pi, 140, dtype=np.float64)
    row = (
        0.9 * np.sin(x)
        + 0.35 * np.sin(2.4 * x + 0.2)
        + 0.07 * np.cos(0.5 * x)
        + np.linspace(-0.4, 0.5, 140, dtype=np.float64)
    )
    return np.asarray(row, dtype=np.float64)


class PeriodSignatureAntigravityTest(unittest.TestCase):
    def test_signature_shape_order_dtype_immutability_and_input_nonmutation_antigravity(
        self,
    ) -> None:
        """Break caught: signature aliases input, is writable, or emits the wrong view order."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.period_features_antigravity import (
            period_signature_antigravity,
        )

        row = fixture_row_antigravity()
        original = row.copy()
        signature = period_signature_antigravity(row)

        self.assertEqual(signature.shape, (120,))
        self.assertEqual(signature.dtype, np.dtype(np.float64))
        self.assertTrue(np.isfinite(signature).all())
        self.assertTrue(signature.flags.owndata)
        self.assertTrue(signature.flags.c_contiguous)
        self.assertFalse(signature.flags.writeable)
        np.testing.assert_array_equal(row, original)
        np.testing.assert_allclose(signature, reference_signature_antigravity(row), rtol=0, atol=1e-12)

    def test_signature_matches_independent_three_view_numpy_reference_antigravity(self) -> None:
        """Break caught: FFT bands, autocorrelation lags, or full/front/back order drift."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.period_features_antigravity import (
            period_signature_antigravity,
        )

        row = fixture_row_antigravity()
        expected = reference_signature_antigravity(row)
        actual = period_signature_antigravity(row)

        self.assertEqual(len(expected), 120)
        np.testing.assert_allclose(actual[:40], expected[:40], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[40:80], expected[40:80], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual[80:120], expected[80:120], rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(np.sum(actual[:16])), 1.0, places=14)
        self.assertAlmostEqual(float(np.sum(actual[40:56])), 1.0, places=14)
        self.assertAlmostEqual(float(np.sum(actual[80:96])), 1.0, places=14)

    def test_signature_invariance_and_period_warp_sensitivity_antigravity(self) -> None:
        """Break caught: feature keeps level/scale artifacts or ignores period changes."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.period_features_antigravity import (
            period_signature_antigravity,
        )

        row = fixture_row_antigravity()
        shifted_scaled = 17.0 + 3.5 * row
        source = np.arange(row.size, dtype=np.float64)
        warped = np.interp(source / 1.18, source, row)

        np.testing.assert_allclose(
            period_signature_antigravity(row),
            period_signature_antigravity(shifted_scaled),
            rtol=0,
            atol=1e-12,
        )
        distance = float(
            np.max(np.abs(period_signature_antigravity(row) - period_signature_antigravity(warped)))
        )
        self.assertGreater(distance, 1e-3)

    def test_signature_rejects_invalid_rows_antigravity(self) -> None:
        """Break caught: malformed, nonfinite, integer, or zero-MAD rows enter scoring."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.period_features_antigravity import (
            period_signature_antigravity,
        )

        row = fixture_row_antigravity()
        invalid_rows = (
            np.ones(140, dtype=np.float64),
            np.append(row, 0.0),
            row.reshape(1, 140),
            row.astype(np.int64),
            row.astype(object),
            np.where(np.arange(140) == 4, np.nan, row),
            np.where(np.arange(140) == 5, np.inf, row),
            np.concatenate([np.ones(70), row[70:]]),
            np.concatenate([row[:70], np.ones(70)]),
        )
        for invalid in invalid_rows:
            with self.subTest(shape=getattr(invalid, "shape", None), dtype=getattr(invalid, "dtype", None)):
                with self.assertRaises(ValueError):
                    period_signature_antigravity(invalid)


if __name__ == "__main__":
    unittest.main()
