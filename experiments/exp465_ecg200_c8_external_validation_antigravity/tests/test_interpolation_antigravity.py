from __future__ import annotations

import inspect
import unittest

import numpy as np

from experiments.exp465_ecg200_c8_external_validation_antigravity.interpolation_antigravity import (
    interpolate_to_140_antigravity,
)


def _expected_140_antigravity(x96: np.ndarray) -> np.ndarray:
    source = np.linspace(0.0, 1.0, 96, dtype=np.float64)
    target = np.linspace(0.0, 1.0, 140, dtype=np.float64)
    return np.vstack([np.interp(target, source, row) for row in x96])


class InterpolationAntigravityTest(unittest.TestCase):
    maxDiff = None

    def test_module_exports_only_the_fixed_entrypoint_antigravity(self) -> None:
        """Break caught: interpolation module grows alternative resampling entrypoints."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import interpolation_antigravity

        self.assertEqual(
            tuple(interpolation_antigravity.__all__),
            ("interpolate_to_140_antigravity",),
        )
        signature = inspect.signature(interpolate_to_140_antigravity)
        self.assertEqual(tuple(signature.parameters), ("values",))

    def test_exact_formula_matches_reference_rows_endpoints_and_flags_antigravity(self) -> None:
        """Break caught: output drifts from the fixed np.interp formula or loses flags."""
        rng = np.random.default_rng(4652026)
        x96 = rng.standard_normal((7, 96))

        result = interpolate_to_140_antigravity(x96.copy())
        expected = _expected_140_antigravity(x96)

        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (7, 140))
        self.assertIs(result.dtype.type, np.float64)
        self.assertTrue(result.flags.owndata)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(result.flags.writeable)
        with self.assertRaises(ValueError):
            result[0, 0] = 1.0

    def test_constant_nonlinear_and_endpoint_preservation_antigravity(self) -> None:
        """Break caught: constant rows, nonlinear rows, or endpoint preservation break."""
        rows = np.vstack(
            [
                np.full((1, 96), 3.25, dtype=np.float64),
                np.sin(np.linspace(0.0, 7.0, 96, dtype=np.float64)),
                (np.linspace(0.0, 1.0, 96, dtype=np.float64) ** 3).reshape(1, -1),
            ],
        )
        result = interpolate_to_140_antigravity(rows)
        np.testing.assert_array_equal(result[0], np.full(140, 3.25, dtype=np.float64))
        self.assertEqual(result[0, 0], 3.25)
        self.assertEqual(result[0, -1], 3.25)
        for row_index in range(rows.shape[0]):
            self.assertEqual(result[row_index, 0], rows[row_index, 0])
            self.assertEqual(result[row_index, -1], rows[row_index, -1])
        np.testing.assert_array_equal(result, _expected_140_antigravity(rows))

    def test_input_is_not_mutated_and_result_is_owned_antigravity(self) -> None:
        """Break caught: aliasing lets callers mutate the input through the output."""
        values = np.arange(192, dtype=np.float64).reshape(2, 96) * 1.0
        original = values.copy()
        result = interpolate_to_140_antigravity(values)
        self.assertTrue(np.array_equal(values, original))
        self.assertFalse(np.shares_memory(result, values))

    def test_rejects_wrong_shape_dtype_finite_and_alias_inputs_antigravity(self) -> None:
        """Break caught: 1-D input, wrong width, aliases, nonfinite, or empty input pass."""
        rng = np.random.default_rng(4652027)

        with self.assertRaisesRegex(ValueError, "2-D"):
            interpolate_to_140_antigravity(rng.standard_normal(96))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "2-D"):
            interpolate_to_140_antigravity(rng.standard_normal((2, 3, 96)))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "width"):
            interpolate_to_140_antigravity(rng.standard_normal((3, 95)))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            interpolate_to_140_antigravity(np.empty((0, 96), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "float64"):
            interpolate_to_140_antigravity(rng.standard_normal((3, 96)).astype(np.float32))
        with self.assertRaisesRegex(ValueError, "float64"):
            interpolate_to_140_antigravity(np.ones((3, 96), dtype=bool))
        with self.assertRaisesRegex(ValueError, "float64"):
            interpolate_to_140_antigravity(np.ones((3, 96), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "finite"):
            values = rng.standard_normal((3, 96))
            values[1, 5] = np.nan
            interpolate_to_140_antigravity(values)
