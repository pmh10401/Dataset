import json
import math
import unittest

import numpy as np

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acceleration_antigravity import (
    BackendAssessmentAntigravity,
    RuntimeAssessmentAntigravity,
    assess_runtime_antigravity,
    fixed_reference_rows_antigravity,
    manifest_payload_antigravity,
)
from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acf_peak_lag_antigravity import (
    score_acf_peak_lag_numba_antigravity,
)
from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.python_reference_antigravity import (
    score_acf_peak_lag_python_reference_antigravity,
)
from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.tests.test_acf_peak_lag_antigravity import (
    periodic_row_antigravity,
    peak_shift_row_numba_antigravity,
)

FROZEN_CONFIG_REQUIRED_SPEEDUP_ANTIGRAVITY = 1.1


class _FrozenConfigProxyAntigravity:
    required_speedup = FROZEN_CONFIG_REQUIRED_SPEEDUP_ANTIGRAVITY


FROZEN_CONFIG = _FrozenConfigProxyAntigravity()


def valid_fit_query_antigravity():
    fit_rows = np.stack(
        (
            periodic_row_antigravity(8.0, 8.0),
            periodic_row_antigravity(9.0, 9.0),
            periodic_row_antigravity(7.0, 11.0),
        )
    )
    query_rows = np.stack(
        (
            periodic_row_antigravity(8.0, 10.0),
            periodic_row_antigravity(10.0, 10.0),
        )
    )
    return np.ascontiguousarray(fit_rows), np.ascontiguousarray(query_rows)


def max_abs_raw_error_antigravity(actual, expected):
    fit_error = max(
        abs(float(left) - float(right))
        for left, right in zip(actual.fit_raw_shift, expected.fit_raw_shift)
    )
    query_error = max(
        abs(float(left) - float(right))
        for left, right in zip(actual.query_raw_shift, expected.query_raw_shift)
    )
    maximum_error = abs(
        float(actual.fit_normal_max_raw_shift)
        - float(expected.fit_normal_max_raw_shift)
    )
    return max(fit_error, query_error, maximum_error)


def exact_discrete_antigravity(actual, expected):
    return all(
        (
            tuple(actual.fit_early_peak_lags) == tuple(expected.fit_early_peak_lags),
            tuple(actual.fit_late_peak_lags) == tuple(expected.fit_late_peak_lags),
            actual.fit_peak_presence.tolist() == expected.fit_peak_presence.tolist(),
            tuple(actual.query_early_peak_lags) == tuple(expected.query_early_peak_lags),
            tuple(actual.query_late_peak_lags) == tuple(expected.query_late_peak_lags),
            actual.query_peak_presence.tolist() == expected.query_peak_presence.tolist(),
            tuple(actual.fit_shifts) == tuple(expected.fit_shifts),
            tuple(actual.query_shifts) == tuple(expected.query_shifts),
            tuple(actual.query_ranks) == tuple(expected.query_ranks),
            actual.confirmation_mask.tolist() == expected.confirmation_mask.tolist(),
        )
    )


class Exp471AccelerationTestAntigravity(unittest.TestCase):
    def setUp(self):
        self.fit_rows, self.query_rows = valid_fit_query_antigravity()

    def test_runtime_records_numba_and_mlx_antigravity(self):
        result = assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=3)
        numba_backend = result.backends["numba"]
        mlx_backend = result.backends["mlx"]

        self.assertIsInstance(result, RuntimeAssessmentAntigravity)
        self.assertEqual(result.selected_backend, "numba")
        self.assertEqual(set(result.backends), {"numba", "mlx"})
        self.assertIsNone(result.backends.get("numpy"))
        self.assertIsInstance(numba_backend, BackendAssessmentAntigravity)
        self.assertTrue(numba_backend.available)
        self.assertTrue(numba_backend.finite)
        self.assertTrue(numba_backend.python_reference_passed)
        self.assertTrue(numba_backend.parity_passed)
        self.assertTrue(numba_backend.exact_candidate_indices)
        self.assertTrue(numba_backend.deterministic)
        self.assertGreaterEqual(numba_backend.max_raw_abs_error, 0.0)
        self.assertLessEqual(numba_backend.max_raw_abs_error, 1e-12)
        self.assertGreaterEqual(numba_backend.cold_seconds, 0.0)
        self.assertGreaterEqual(numba_backend.warm_median_seconds, 0.0)
        self.assertGreaterEqual(numba_backend.warm_iqr_seconds, 0.0)
        self.assertTrue(math.isfinite(numba_backend.cold_seconds))
        self.assertTrue(math.isfinite(numba_backend.warm_median_seconds))
        self.assertTrue(math.isfinite(numba_backend.warm_iqr_seconds))
        self.assertTrue(numba_backend.eligible)
        self.assertIsNone(numba_backend.fallback_reason)
        self.assertEqual(result.selected_backend, numba_backend.backend)
        self.assertFalse(mlx_backend.eligible)
        if mlx_backend.fallback_reason is None:
            self.assertGreaterEqual(float(mlx_backend.speedup), 1.1)

    def test_reference_failure_stops_antigravity(self):
        with self.assertRaisesRegex(RuntimeError, "Python reference"):
            assess_runtime_antigravity(
                self.fit_rows,
                self.query_rows,
                repetitions=1,
                reference_tolerance=-1.0,
            )

    def test_fixed_reference_fixture_is_frozen_and_parity_authority(self):
        rows = fixed_reference_rows_antigravity()
        self.assertEqual(rows.shape, (6, 140))
        self.assertEqual(rows.dtype, np.dtype(np.float64))
        self.assertFalse(rows.flags.writeable)
        self.assertTrue(np.all(np.isfinite(rows)))
        observed_periods = []
        for row in rows:
            early, late, _, _, _ = peak_shift_row_numba_antigravity(row)
            observed_periods.append((int(early), int(late)))
        expected_periods = ((8, 8), (9, 9), (7, 11), (8, 10), (10, 10), (12, 12))
        self.assertEqual(tuple(observed_periods), expected_periods)

        fit_rows, query_rows = rows[:3], rows[3:]
        expected = score_acf_peak_lag_python_reference_antigravity(fit_rows, query_rows)
        observed = score_acf_peak_lag_numba_antigravity(fit_rows, query_rows)
        self.assertLessEqual(max_abs_raw_error_antigravity(observed, expected), 1e-12)
        self.assertTrue(exact_discrete_antigravity(observed, expected))

    def test_repeated_assessment_stays_within_preregistered_parity_gate(self):
        first = assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=5)
        second = assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=5)
        self.assertLessEqual(first.backends["numba"].max_raw_abs_error, 1e-12)
        self.assertLessEqual(second.backends["numba"].max_raw_abs_error, 1e-12)
        self.assertTrue(first.backends["numba"].exact_candidate_indices)
        self.assertTrue(second.backends["numba"].deterministic)
        self.assertGreaterEqual(second.backends["numba"].cold_seconds, 0.0)
        self.assertGreaterEqual(second.backends["numba"].warm_median_seconds, 0.0)
        self.assertGreaterEqual(second.backends["numba"].warm_iqr_seconds, 0.0)

    def test_manifest_records_versions_device_timing_parity_and_boundary(self):
        result = assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=3)
        payload = manifest_payload_antigravity(result)
        canonical = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)

        self.assertIn("assessment", payload)
        self.assertIn("environment", payload)
        self.assertIn("timing", payload)
        self.assertIn("parity", payload)
        self.assertIn("numba_version", payload["environment"])
        self.assertIn("mlx_version", payload["environment"])
        self.assertIn("device", payload["environment"])
        self.assertEqual(payload["selected_backend"], "numba")
        self.assertEqual(payload["private_runs"], 0)
        self.assertIn("official_runs", payload)
        self.assertIn("assessment", canonical)

    def test_invalid_inputs_are_rejected_before_measurement(self):
        bad_cases = (self.fit_rows[:1], self.fit_rows.astype(np.float32), self.fit_rows[:, :139])
        for index, bad_fit in enumerate(bad_cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    assess_runtime_antigravity(bad_fit, self.query_rows, repetitions=1)
        with self.assertRaisesRegex(ValueError, "repetitions"):
            assess_runtime_antigravity(self.fit_rows, self.query_rows, repetitions=0)

        empty_query = np.empty((0, 140), dtype=np.float64)
        result = assess_runtime_antigravity(self.fit_rows, empty_query, repetitions=1)
        self.assertTrue(result.backends["numba"].eligible)


if __name__ == "__main__":
    unittest.main()
