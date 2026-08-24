"""Behavior checks for the Exp470 Numba runtime assessment.

Each test names the production regression it catches.  These tests exercise the
real scorer and only use the fixed, public four-row smoke fixture.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.acceleration_antigravity import (
    assess_mlx_smoke_antigravity,
    assess_runtime_antigravity,
    fixed_reference_rows_antigravity,
    main_antigravity,
    score_bytes_antigravity,
    timing_summary_antigravity,
)
from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.spectral_entropy_antigravity import (
    score_spectral_entropy_numba_antigravity,
)


_FIXTURE_SHA256_ANTIGRAVITY = "054bfadb84cd5e4fb8e123207a25c089c6e4cd45366895e74c5e39707068187c"


class AccelerationAssessmentTestAntigravity(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = fixed_reference_rows_antigravity()
        self.fit_rows = self.rows[:3]
        self.query_rows = self.rows[3:]

    def test_fixed_fixture_preserves_frozen_bytes_and_readonly_contract_antigravity(self) -> None:
        """Break caught: an edited signal or mutable/noncanonical fixture invalidates smoke parity."""
        self.assertEqual(self.rows.dtype, np.float64)
        self.assertEqual(self.rows.shape, (4, 140))
        self.assertTrue(self.rows.flags.c_contiguous)
        self.assertFalse(self.rows.flags.writeable)
        self.assertEqual(hashlib.sha256(self.rows.tobytes(order="C")).hexdigest(), _FIXTURE_SHA256_ANTIGRAVITY)

    def test_assessment_records_numba_and_mlx_without_numpy_backend_antigravity(self) -> None:
        """Break caught: runtime evidence omits a backend, selects MLX, or loses Numba checks."""
        result = assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=3)
        self.assertEqual(result.selected_backend, "numba")
        self.assertEqual(set(result.backends), {"numba", "mlx"})
        self.assertEqual(result.fixture_sha256, _FIXTURE_SHA256_ANTIGRAVITY)
        self.assertTrue(result.backends["numba"].finite)
        self.assertTrue(result.backends["numba"].python_reference_passed)
        self.assertTrue(result.backends["numba"].parity_passed)
        self.assertTrue(result.backends["numba"].exact_candidate_indices)
        self.assertTrue(result.backends["numba"].deterministic)
        self.assertIsNone(result.backends.get("numpy"))

    def test_assessment_records_installed_mlx_metal_distribution_version_antigravity(self) -> None:
        """Break caught: MLX metadata reads a module attribute and drops the installed mlx-metal version."""
        result = assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=3)
        self.assertEqual(result.mlx_metal_version, version("mlx-metal"))

    def test_reference_failure_stops_before_backend_selection_antigravity(self) -> None:
        """Break caught: a failed oracle check is silently accepted by the assessment."""
        with self.assertRaisesRegex(RuntimeError, "Python reference"):
            assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=3, reference_tolerance=-1.0)

    def test_invalid_runtime_inputs_are_rejected_antigravity(self) -> None:
        """Break caught: invalid repetitions, speed gate, or nonfinite rows reach timing/scoring."""
        with self.assertRaisesRegex(ValueError, "repetitions"):
            assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=0)
        with self.assertRaisesRegex(ValueError, "reference_tolerance"):
            assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=3, reference_tolerance=float("nan"))
        with self.assertRaisesRegex(ValueError, "required_mlx_speedup"):
            assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=3, required_mlx_speedup=0.0)
        nonfinite = self.query_rows.copy()
        nonfinite[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            assess_runtime_antigravity(self.fit_rows, nonfinite, repetitions=3)

    def test_score_bytes_and_timing_summary_are_deterministic_antigravity(self) -> None:
        """Break caught: score fields or timing summary change across equivalent calls."""
        first = score_spectral_entropy_numba_antigravity(self.fit_rows, self.query_rows)
        second = score_spectral_entropy_numba_antigravity(self.fit_rows, self.query_rows)
        self.assertEqual(score_bytes_antigravity(first), score_bytes_antigravity(second))
        median, iqr = timing_summary_antigravity((0.2, 0.1, 0.3, 0.4))
        self.assertEqual(median, 0.25)
        self.assertAlmostEqual(iqr, 0.2)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            timing_summary_antigravity((0.0,))

    def test_unavailable_mlx_returns_a_real_fallback_record_antigravity(self) -> None:
        """Break caught: a missing MLX package skips the smoke instead of recording a fallback."""
        numba_scores = score_spectral_entropy_numba_antigravity(self.fit_rows, self.query_rows)
        with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
            result = assess_mlx_smoke_antigravity(
                self.fit_rows, self.query_rows, numba_scores, 0.001, 1.1, repetitions=1
            )
        self.assertFalse(result.available)
        self.assertFalse(result.eligible)
        self.assertIn("MLX unavailable", result.fallback_reason or "")

    def test_cli_writes_strict_finite_json_manifest_antigravity(self) -> None:
        """Break caught: generated manifest is not strict JSON or serializes nonfinite values."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "assessment_antigravity.json"
            self.assertEqual(main_antigravity(("--output", str(output), "--repetitions", "3")), 0)
            payload = json.loads(output.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        self.assertEqual(payload["selected_backend"], "numba")
        self.assertEqual(set(payload["backends"]), {"numba", "mlx"})
        self.assertEqual(payload["fixture_sha256"], _FIXTURE_SHA256_ANTIGRAVITY)


if __name__ == "__main__":
    unittest.main()
