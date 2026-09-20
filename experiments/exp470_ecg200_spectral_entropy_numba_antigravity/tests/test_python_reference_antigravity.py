from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.python_reference_antigravity import (
    PythonReferenceScoresAntigravity,
    score_spectral_entropy_python_reference_antigravity,
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


class PythonReferenceTestAntigravity(unittest.TestCase):
    def test_four_row_oracle_returns_immutable_tuple_scores_antigravity(self) -> None:
        """Break caught: the independent four-row entropy oracle stops preserving its immutable tuple contract."""
        rows = _fixed_rows_antigravity()
        scores = score_spectral_entropy_python_reference_antigravity(
            tuple(tuple(float(value) for value in row) for row in rows[:3]),
            tuple(tuple(float(value) for value in row) for row in rows[3:]),
        )

        self.assertIsInstance(scores, PythonReferenceScoresAntigravity)
        self.assertIsInstance(scores.fit_raw_entropy, tuple)
        self.assertIsInstance(scores.query_raw_entropy, tuple)
        self.assertIsInstance(scores.query_ranks, tuple)
        self.assertIsInstance(scores.confirmation_mask, tuple)
        self.assertEqual(len(scores.fit_raw_entropy), 3)
        self.assertEqual(len(scores.query_raw_entropy), 1)
        self.assertTrue(all(math.isfinite(value) for value in scores.fit_raw_entropy))
        self.assertTrue(all(math.isfinite(value) for value in scores.query_raw_entropy))
        with self.assertRaises(AttributeError):
            scores.fit_raw_entropy = ()  # type: ignore[misc]

    def test_reference_uses_only_python_math_and_sequence_operations_antigravity(self) -> None:
        """Break caught: the validation oracle silently gains a NumPy FFT or reduction dependency."""
        source = inspect.getsource(
            __import__(
                "experiments.exp470_ecg200_spectral_entropy_numba_antigravity.python_reference_antigravity",
                fromlist=["placeholder"],
            )
        )

        self.assertIn("math.fsum", source)
        self.assertIn("statistics.median", source)
        self.assertNotIn("numpy", source)
        self.assertNotIn("np.", source)
        self.assertNotIn("fft", source.lower())


if __name__ == "__main__":
    unittest.main()
