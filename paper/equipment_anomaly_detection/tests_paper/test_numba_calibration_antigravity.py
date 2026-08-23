"""Parity contracts for the Exp467 Numba calibration candidate."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.exp467_numba_calibration_resume_antigravity import (
    hard_route_adapter_antigravity,
    hard_sources_adapter_antigravity,
    minirocket_adapter_antigravity,
    science_artifacts_antigravity,
    scoring_antigravity,
)
from paper.equipment_anomaly_detection import numba_calibration_antigravity

EXP467_ADAPTER_MODULES_ANTIGRAVITY = (
    hard_sources_adapter_antigravity,
    minirocket_adapter_antigravity,
    hard_route_adapter_antigravity,
    science_artifacts_antigravity,
    scoring_antigravity,
)


def _candidate_module_antigravity():
    return numba_calibration_antigravity


class NumbaCalibrationTestsAntigravity(unittest.TestCase):
    def test_exp467_science_boundaries_bind_the_numba_function(self) -> None:
        calibration = _candidate_module_antigravity()
        self.assertIsNotNone(calibration, "Exp467 Numba calibration module is missing")
        for module in EXP467_ADAPTER_MODULES_ANTIGRAVITY:
            with self.subTest(module=module.__name__):
                self.assertIs(
                    module.upper_tail_pvalues,
                    calibration.upper_tail_pvalues_numba_antigravity,
                )

    def test_float64_fast_path_preserves_exact_tail_counts_and_ties(self) -> None:
        module = _candidate_module_antigravity()
        self.assertIsNotNone(module, "Exp467 Numba calibration module is missing")
        reference = np.asarray(
            [
                -np.finfo(np.float64).max,
                -0.0,
                0.0,
                np.nextafter(1.0, 0.0),
                1.0,
                1.0,
                np.finfo(np.float64).max,
            ],
            dtype=np.float64,
        )
        query = np.asarray(
            [
                -0.0,
                np.nextafter(1.0, 0.0),
                1.0,
                np.nextafter(1.0, np.inf),
                np.finfo(np.float64).max,
            ],
            dtype=np.float64,
        )

        actual = module.upper_tail_pvalues_numba_antigravity(reference, query)

        np.testing.assert_array_equal(
            actual,
            np.asarray([0.875, 0.625, 0.5, 0.25, 0.25], dtype=np.float64),
        )

    def test_preregistered_candidate_indices_are_exact(self) -> None:
        module = _candidate_module_antigravity()
        self.assertIsNotNone(module, "Exp467 Numba calibration module is missing")
        reference = np.arange(199.0, dtype=np.float64)
        query = np.asarray([200.0, 198.0, 197.0, 196.0, 194.0])
        expected_by_alpha = {
            0.005: [0],
            0.01: [0, 1],
            0.015: [0, 1, 2],
            0.02: [0, 1, 2, 3],
            0.03: [0, 1, 2, 3, 4],
        }

        pvalue = module.upper_tail_pvalues_numba_antigravity(reference, query)

        for alpha, expected in expected_by_alpha.items():
            with self.subTest(alpha=alpha):
                self.assertEqual(np.flatnonzero(pvalue <= alpha).tolist(), expected)

    def test_mixed_dtype_uses_python_scalar_ordering_above_float64_precision(self) -> None:
        module = _candidate_module_antigravity()
        self.assertIsNotNone(module, "Exp467 Numba calibration module is missing")
        largest_exact_float = 2**53

        actual = module.upper_tail_pvalues_numba_antigravity(
            np.asarray([float(largest_exact_float)]),
            np.asarray([largest_exact_float + 1], dtype=np.int64),
        )

        np.testing.assert_array_equal(actual, np.asarray([0.5]))

    def test_strided_inputs_empty_query_and_repeat_are_deterministic(self) -> None:
        module = _candidate_module_antigravity()
        self.assertIsNotNone(module, "Exp467 Numba calibration module is missing")
        reference = np.arange(20.0, dtype=np.float64)[::2]
        query = np.arange(12.0, dtype=np.float64)[1::3]

        first = module.upper_tail_pvalues_numba_antigravity(reference, query)
        second = module.upper_tail_pvalues_numba_antigravity(reference, query)
        empty = module.upper_tail_pvalues_numba_antigravity(
            reference, np.asarray([], dtype=np.float64)
        )

        np.testing.assert_array_equal(
            first, np.asarray([10 / 11, 9 / 11, 7 / 11, 6 / 11])
        )
        np.testing.assert_array_equal(second, first)
        self.assertEqual(empty.dtype, np.dtype(np.float64))
        self.assertEqual(empty.shape, (0,))

    def test_backend_record_exposes_reproducible_numba_contract(self) -> None:
        module = _candidate_module_antigravity()
        self.assertIsNotNone(module, "Exp467 Numba calibration module is missing")

        record = module.numba_calibration_backend_antigravity()

        self.assertEqual(record["schema"], "exp467-numba-calibration-backend-antigravity.v1")
        self.assertEqual(record["selected_backend"], "numba")
        self.assertEqual(record["fast_path_dtype"], "float64")
        self.assertFalse(record["fastmath"])
        self.assertFalse(record["parallel"])
        self.assertTrue(record["nogil"])
        for key in ("python", "numpy", "numba", "llvmlite", "machine", "processor"):
            self.assertIsInstance(record[key], str)
            self.assertTrue(record[key])

    def test_exp467_backend_provenance_binds_the_numba_contract(self) -> None:
        calibration = _candidate_module_antigravity()
        self.assertIsNotNone(calibration, "Exp467 Numba calibration module is missing")
        from experiments.exp467_numba_calibration_resume_antigravity.backend_antigravity import (
            assess_exp467_backend_antigravity,
        )

        artifact = assess_exp467_backend_antigravity().artifact_record()

        self.assertEqual(
            artifact.get("numba_calibration"),
            calibration.numba_calibration_backend_antigravity(),
        )

    def test_exp467_context_closure_binds_numba_and_every_local_adapter(self) -> None:
        from scratch.run_exp467_numba_calibration_resume_antigravity import (
            _code_closure_antigravity,
        )

        closure = _code_closure_antigravity()
        required = {
            "paper/equipment_anomaly_detection/calibration_paper.py",
            "paper/equipment_anomaly_detection/numba_calibration_antigravity.py",
            "scratch/run_exp467_numba_calibration_resume_antigravity.py",
            "experiments/exp467_numba_calibration_resume_antigravity/hard_sources_adapter_antigravity.py",
            "experiments/exp467_numba_calibration_resume_antigravity/minirocket_adapter_antigravity.py",
            "experiments/exp467_numba_calibration_resume_antigravity/hard_route_adapter_antigravity.py",
            "experiments/exp467_numba_calibration_resume_antigravity/science_artifacts_antigravity.py",
            "experiments/exp467_numba_calibration_resume_antigravity/scoring_antigravity.py",
        }

        self.assertTrue(required <= set(closure))
        self.assertTrue(
            all(len(digest) == 64 for digest in closure.values())
        )


if __name__ == "__main__":
    unittest.main()
