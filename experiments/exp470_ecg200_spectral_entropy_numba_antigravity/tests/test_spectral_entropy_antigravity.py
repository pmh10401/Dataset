from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.python_reference_antigravity import (
    score_spectral_entropy_python_reference_antigravity,
)
from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.spectral_entropy_antigravity import (
    EntropyScoresAntigravity,
    assemble_raw_rank_numba_antigravity,
    entropy_rows_numba_antigravity,
    score_spectral_entropy_numba_antigravity,
    strict_confirmation_numba_antigravity,
)


def _fixed_rows_antigravity() -> np.ndarray:
    x = np.arange(140, dtype=np.float64)
    return np.stack(
        (
            np.ones(140, dtype=np.float64),
            np.sin(2.0 * np.pi * x / 14.0),
            x / 139.0,
            np.sin(2.0 * np.pi * x / 7.0) + 0.25 * np.sin(2.0 * np.pi * x / 4.0),
        )
    )


def _as_python_rows_antigravity(rows: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in rows)


class SpectralEntropyNumbaTestAntigravity(unittest.TestCase):
    def test_numba_scores_match_independent_four_row_oracle_antigravity(self) -> None:
        """Break caught: DFT, entropy, LOO raw, rank, or confirmation no longer follows frozen E11 math."""
        rows = _fixed_rows_antigravity()
        expected = score_spectral_entropy_python_reference_antigravity(
            _as_python_rows_antigravity(rows[:3]), _as_python_rows_antigravity(rows[3:])
        )
        actual = score_spectral_entropy_numba_antigravity(rows[:3], rows[3:])

        self.assertIsInstance(actual, EntropyScoresAntigravity)
        self.assertEqual(actual.backend, "numba")
        np.testing.assert_allclose(actual.fit_raw_entropy, expected.fit_raw_entropy, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(actual.query_raw_entropy, expected.query_raw_entropy, rtol=0.0, atol=1e-12)
        self.assertEqual(tuple(actual.query_ranks), expected.query_ranks)
        self.assertEqual(tuple(actual.confirmation_mask), expected.confirmation_mask)
        self.assertAlmostEqual(
            actual.fit_normal_max_raw_entropy,
            expected.fit_normal_max_raw_entropy,
            delta=1e-12,
        )

    def test_entropy_is_finite_and_normalized_for_fixed_signal_families_antigravity(self) -> None:
        """Break caught: a constant, sinusoid, ramp, or mixed wave produces invalid normalized entropy."""
        rows = _fixed_rows_antigravity()
        entropy = entropy_rows_numba_antigravity(rows)

        self.assertEqual(entropy.shape, (4,))
        self.assertTrue(np.all(np.isfinite(entropy)))
        self.assertTrue(np.all(entropy >= 0.0))
        self.assertTrue(np.all(entropy <= 1.0))

    def test_strict_confirmation_rejects_tie_and_accepts_one_ulp_above_antigravity(self) -> None:
        """Break caught: strict confirmation accepts equality or rejects the immediate greater floating-point value."""
        boundary = np.float64(0.25)
        query_raw = np.array((boundary, np.nextafter(boundary, np.inf)), dtype=np.float64)
        result = strict_confirmation_numba_antigravity(query_raw, boundary)

        self.assertEqual(tuple(result), (False, True))

    def test_repeated_calls_have_identical_owned_immutable_outputs_antigravity(self) -> None:
        """Break caught: scorer output becomes nondeterministic, mutable, or no longer owns contiguous exact-dtype arrays."""
        rows = _fixed_rows_antigravity()
        first = score_spectral_entropy_numba_antigravity(rows[:3], rows[3:])
        second = score_spectral_entropy_numba_antigravity(rows[:3], rows[3:])

        for first_array, second_array, dtype in (
            (first.fit_raw_entropy, second.fit_raw_entropy, np.dtype(np.float64)),
            (first.query_raw_entropy, second.query_raw_entropy, np.dtype(np.float64)),
            (first.query_ranks, second.query_ranks, np.dtype(np.float64)),
            (first.confirmation_mask, second.confirmation_mask, np.dtype(np.bool_)),
        ):
            self.assertTrue(first_array.flags.c_contiguous)
            self.assertTrue(first_array.flags.owndata)
            self.assertEqual(first_array.dtype, dtype)
            self.assertFalse(first_array.flags.writeable)
            self.assertEqual(first_array.tobytes(), second_array.tobytes())
        self.assertEqual(first.fit_normal_max_raw_entropy, second.fit_normal_max_raw_entropy)

    def test_invalid_input_contracts_are_rejected_antigravity(self) -> None:
        """Break caught: invalid dimensions, dtype, width, finiteness, or row counts reach compiled scoring."""
        valid_fit = np.zeros((2, 140), dtype=np.float64)
        valid_query = np.zeros((1, 140), dtype=np.float64)
        invalid_cases = (
            (valid_fit.astype(np.float32), valid_query),
            (np.zeros((2, 139), dtype=np.float64), valid_query),
            (np.zeros(140, dtype=np.float64), valid_query),
            (np.array(((0.0,) * 139 + (np.nan,),) * 2, dtype=np.float64), valid_query),
            (np.zeros((1, 140), dtype=np.float64), valid_query),
            (valid_fit, np.empty((0, 140), dtype=np.float64)),
        )
        for fit_rows, query_rows in invalid_cases:
            with self.subTest(fit_shape=getattr(fit_rows, "shape", None), query_shape=getattr(query_rows, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    score_spectral_entropy_numba_antigravity(fit_rows, query_rows)

    def test_compiled_kernels_and_prohibited_score_backends_are_absent_antigravity(self) -> None:
        """Break caught: scoring leaves njit kernels or reintroduces an FFT/backend selector/NumPy scorer."""
        rows = _fixed_rows_antigravity()
        score_spectral_entropy_numba_antigravity(rows[:3], rows[3:])
        self.assertTrue(entropy_rows_numba_antigravity.signatures)
        self.assertTrue(assemble_raw_rank_numba_antigravity.signatures)
        self.assertTrue(strict_confirmation_numba_antigravity.signatures)
        source = inspect.getsource(
            __import__(
                "experiments.exp470_ecg200_spectral_entropy_numba_antigravity.spectral_entropy_antigravity",
                fromlist=["placeholder"],
            )
        )
        self.assertNotIn("np.fft", source)
        self.assertNotIn("backend_selector", source)
        self.assertNotIn("numpy_score", source)
        self.assertNotIn("fastmath=True", source)
        self.assertNotIn("cache=True", source)


if __name__ == "__main__":
    unittest.main()
