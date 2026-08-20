from __future__ import annotations

import unittest

import numpy as np


def independent_gradual_reference_antigravity(values: np.ndarray) -> np.ndarray:
    row = np.array(values, dtype=np.float64, copy=True)
    parts: list[float] = []
    for start, stop in ((0, 140), (0, 70), (70, 140)):
        view = row[start:stop]
        center = float(np.median(view))
        centered = view - center
        scale = float(1.4826 * np.median(np.abs(centered)))
        z = centered / scale
        u = np.linspace(0.0, 1.0, len(view), dtype=np.float64)
        for basis in (
            u,
            u * u,
            np.maximum(u - 0.25, 0.0),
            np.maximum(u - 0.50, 0.0),
            np.maximum(u - 0.75, 0.0),
        ):
            centered_basis = basis - float(np.mean(basis))
            unit_basis = centered_basis / float(np.linalg.norm(centered_basis))
            parts.append(float(np.dot(z, unit_basis)))
        quarters = np.array_split(z, 4)
        medians = [float(np.median(part)) for part in quarters]
        parts.extend(
            (
                medians[1] - medians[0],
                medians[2] - medians[1],
                medians[3] - medians[2],
            )
        )
    return np.asarray(parts, dtype=np.float64)


class GradualFeatureAntigravityTest(unittest.TestCase):
    def test_gradual_signature_matches_independent_reference_and_contract_antigravity(self) -> None:
        """Break caught: view order, basis normalization, quarter contrasts, or output contract changes."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.gradual_features_antigravity import (
            gradual_signature_antigravity,
        )

        u = np.linspace(-2.0, 3.0, 140, dtype=np.float64)
        values = (
            0.7 * u
            + 0.15 * u * u
            + 0.6 * np.maximum(u - 0.25, 0.0)
            - 0.35 * np.maximum(u - 1.25, 0.0)
            + 0.12 * np.sin(np.linspace(0.0, 9.0, 140, dtype=np.float64))
        ).astype(np.float32)
        original = np.array(values, copy=True)
        expected = independent_gradual_reference_antigravity(values)

        actual = gradual_signature_antigravity(values)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        self.assertEqual(actual.shape, (24,))
        self.assertEqual(actual.dtype, np.dtype(np.float64))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(actual.flags.owndata)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertFalse(actual.flags.writeable)
        np.testing.assert_array_equal(values, original)

    def test_gradual_signature_rejects_malformed_or_degenerate_rows_antigravity(self) -> None:
        """Break caught: malformed values or zero robust-scale views enter score generation."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.gradual_features_antigravity import (
            gradual_signature_antigravity,
        )

        good = np.linspace(-1.0, 1.0, 140, dtype=np.float64)
        invalid_rows = (
            np.linspace(-1.0, 1.0, 139, dtype=np.float64),
            good.reshape(1, 140),
            np.arange(140, dtype=np.int64),
            np.asarray(["x"] * 140, dtype=object),
            np.full(140, 1.0, dtype=np.float64),
            np.r_[np.zeros(70, dtype=np.float64), np.linspace(0.0, 1.0, 70)],
            np.r_[np.linspace(0.0, 1.0, 70), np.zeros(70, dtype=np.float64)],
            np.r_[np.nan, np.linspace(0.0, 1.0, 139)],
        )
        for values in invalid_rows:
            with self.subTest(shape=getattr(values, "shape", None), dtype=getattr(values, "dtype", None)):
                with self.assertRaises(ValueError):
                    gradual_signature_antigravity(values)


if __name__ == "__main__":
    unittest.main()
