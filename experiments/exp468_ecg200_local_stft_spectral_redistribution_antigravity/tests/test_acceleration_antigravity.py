"""Backend parity and acceleration-gate contracts for Exp468 Task 3."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.acceleration_antigravity import (
    AccelerationFoldAntigravity,
    AccelerationWorkloadAntigravity,
    assess_acceleration_antigravity,
    run_acceleration_workload_antigravity,
)


def _fixture_rows_antigravity(count: int, phase: float) -> np.ndarray:
    sample = np.arange(140, dtype=np.float64)
    return np.stack(
        tuple(
            np.sin((1.0 + index % 5) * sample / 7.0 + phase)
            + 0.2 * np.cos((2.0 + index % 3) * sample / 11.0)
            for index in range(count)
        ),
    )


class AccelerationParityAntigravityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workload = AccelerationWorkloadAntigravity(
            folds=(
                AccelerationFoldAntigravity(
                    fit_rows=_fixture_rows_antigravity(4, 0.0),
                    query_scopes=(
                        _fixture_rows_antigravity(2, 0.3),
                        _fixture_rows_antigravity(2, 0.7),
                    ),
                ),
            ),
            numpy_repetitions=1,
            warm_repetitions=1,
        )

    def test_numba_full_output_matches_numpy_oracle_antigravity(self) -> None:
        numpy_result = run_acceleration_workload_antigravity(
            self.workload, backend="numpy",
        )
        numba_result = run_acceleration_workload_antigravity(
            self.workload, backend="numba",
        )

        self.assertEqual(len(numpy_result), len(numba_result))
        for expected_fold, actual_fold in zip(
            numpy_result, numba_result, strict=True,
        ):
            self.assertEqual(len(expected_fold), len(actual_fold))
            for expected, actual in zip(expected_fold, actual_fold, strict=True):
                self.assertTrue(np.isfinite(actual.fit_raw_js).all())
                self.assertTrue(np.isfinite(actual.query_raw_js).all())
                np.testing.assert_allclose(
                    actual.fit_raw_js, expected.fit_raw_js, rtol=0.0, atol=1e-12,
                )
                np.testing.assert_allclose(
                    actual.query_raw_js, expected.query_raw_js, rtol=0.0, atol=1e-12,
                )
                np.testing.assert_array_equal(actual.query_ranks, expected.query_ranks)
                np.testing.assert_array_equal(
                    actual.confirmation_mask, expected.confirmation_mask,
                )

    def test_assessment_records_fixed_gate_and_backend_truth_antigravity(self) -> None:
        assessment = assess_acceleration_antigravity(self.workload)

        self.assertEqual(assessment.required_speedup, 1.1)
        self.assertEqual(assessment.workload_counts["fit_raw"], 4)
        self.assertEqual(assessment.workload_counts["query_raw"], 4)
        self.assertEqual(assessment.workload_counts["transformed_rows"], 8)
        self.assertEqual(assessment.workload_counts["frames"], 40)
        self.assertIn(assessment.selected_backend, ("numpy", "numba", "mlx"))
        self.assertIn("numpy", assessment.backends)
        self.assertIn("numba", assessment.backends)
        self.assertIn("mlx", assessment.backends)
        self.assertEqual(
            assessment.selected_backend != "numpy",
            assessment.gate_passed,
        )
        if assessment.backends["numba"].parity_passed:
            self.assertLessEqual(assessment.backends["numba"].max_raw_abs_error, 1e-12)
            self.assertTrue(assessment.backends["numba"].rank_exact)
            self.assertTrue(assessment.backends["numba"].confirmation_exact)

    def test_workload_validation_rejects_wrong_width_antigravity(self) -> None:
        with self.assertRaises(ValueError):
            AccelerationFoldAntigravity(
                fit_rows=np.zeros((3, 139), dtype=np.float64),
                query_scopes=(np.zeros((1, 140), dtype=np.float64),),
            )


if __name__ == "__main__":
    unittest.main()
