import math
import os
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from experiments.exp411_near_repeat_qualification.acceleration import (
    AccelerationPolicy,
    audit_dataset_pairs,
    boundary_safe_minimum,
    boundary_safe_threshold,
    choose_backend,
    configure_runtime,
    full_pairwise_rms,
    minimum_rms,
)


class AccelerationTests(TestCase):
    def test_numba_full_matrix_uses_direct_rms_at_large_offset(self) -> None:
        rows = np.asarray(
            [
                [1_000_000_000.0, 1_000_000_000.25],
                [1_000_000_000.5, 1_000_000_000.75],
            ],
            dtype=np.float64,
        )

        observed = full_pairwise_rms(rows, backend="numba_cpu")

        np.testing.assert_allclose(
            observed,
            np.asarray(
                [
                    [0.0, 0.5],
                    [0.5, 0.0],
                ]
            ),
            rtol=0.0,
            atol=1e-6,
        )

    def test_numba_minimum_rms_matches_hand_calculation(self) -> None:
        fit = np.asarray([[0.0, 0.0], [3.0, 4.0]])
        query = np.asarray([[0.0, 4.0], [6.0, 8.0]])

        observed = minimum_rms(fit, query, backend="numba_cpu")

        np.testing.assert_allclose(
            observed,
            np.asarray([math.sqrt(4.5), math.sqrt(12.5)]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_runtime_uses_twelve_numba_threads_and_one_nested_thread(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            state = configure_runtime(AccelerationPolicy())

            self.assertEqual(state["numba_threads"], 12)
            self.assertEqual(state["nested_threads"], 1)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                self.assertEqual(os.environ[name], "1")

    def test_backend_selection_uses_only_frozen_train_shape_cost(self) -> None:
        policy = AccelerationPolicy(numpy_cost_max=20)

        self.assertEqual(choose_backend(5, 4, policy), "numpy_cpu")
        self.assertEqual(choose_backend(6, 4, policy), "numba_cpu")

    def test_dataset_pair_audit_is_deterministic(self) -> None:
        rows = np.arange(40, dtype=np.float64).reshape(10, 4)
        digests = tuple(f"{index:064x}" for index in range(10))
        policy = AccelerationPolicy(audit_pairs=7)

        first = audit_dataset_pairs(rows, digests, policy)
        second = audit_dataset_pairs(rows, digests, policy)

        self.assertEqual(first.sampled_pairs, second.sampled_pairs)
        self.assertEqual(len(first.sampled_pairs), 7)
        self.assertEqual(first.backend, "numpy_cpu")
        self.assertLessEqual(first.max_abs_error, 1e-6)
        self.assertIsNone(first.fallback_reason)

    def test_dataset_pair_audit_falls_back_when_tolerance_is_exceeded(self) -> None:
        rows = np.arange(40, dtype=np.float64).reshape(10, 4)
        digests = tuple(f"{index:064x}" for index in range(10))
        policy = AccelerationPolicy(
            audit_pairs=7,
            numpy_cost_max=1,
            distance_tolerance=1e-6,
        )

        with patch(
            "experiments.exp411_near_repeat_qualification.acceleration."
            "_numba_pair_distances",
            return_value=np.full(7, 1.0, dtype=np.float64),
        ):
            audit = audit_dataset_pairs(rows, digests, policy)

        self.assertEqual(audit.backend, "numpy_cpu")
        self.assertGreater(audit.max_abs_error, 1e-6)
        self.assertEqual(audit.fallback_reason, "pair_audit_tolerance_exceeded")

    def test_quantile_boundary_is_recomputed_by_numpy_direct_rms(self) -> None:
        rows = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [10.0, 0.0]],
            dtype=np.float64,
        )
        accelerated_reference = np.asarray(
            [0.7071069, 0.7071067, 2.1213204, 4.2426407],
            dtype=np.float64,
        )

        observed = boundary_safe_threshold(
            rows,
            accelerated_reference,
            0.05,
            tolerance=1e-6,
        )

        self.assertAlmostEqual(observed, math.sqrt(0.5), places=12)

    def test_query_boundary_is_recomputed_but_nonboundary_is_preserved(self) -> None:
        fit = np.asarray([[0.0, 0.0]], dtype=np.float64)
        query = np.asarray([[3.0, 4.0], [6.0, 8.0]], dtype=np.float64)
        threshold = math.sqrt(12.5)
        observed = np.asarray([threshold + 5e-7, 10.0], dtype=np.float64)

        result = boundary_safe_minimum(
            fit,
            query,
            observed,
            (threshold,),
            tolerance=1e-6,
        )

        np.testing.assert_allclose(
            result,
            np.asarray([math.sqrt(12.5), 10.0]),
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
