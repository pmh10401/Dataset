import math
import unittest

import numpy as np

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acf_peak_lag_antigravity import (
    score_acf_peak_lag_numba_antigravity,
)
from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.python_reference_antigravity import (
    peak_shift_row_python_antigravity,
    score_acf_peak_lag_python_reference_antigravity,
)


def periodic_row_antigravity(early_period=8.0, late_period=10.0):
    row = np.sin(2.0 * np.pi * np.arange(140, dtype=np.float64) / 9.0)
    row[0:56] = np.sin(2.0 * np.pi * np.arange(56, dtype=np.float64) / float(early_period))
    row[84:140] = np.sin(2.0 * np.pi * np.arange(56, dtype=np.float64) / float(late_period))
    return row


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
            periodic_row_antigravity(11.0, 11.0),
        )
    )
    return np.ascontiguousarray(fit_rows, dtype=np.float64), np.ascontiguousarray(
        query_rows, dtype=np.float64
    )


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


class Exp471PythonReferenceTestAntigravity(unittest.TestCase):
    def test_known_peak_and_presence_authority_antigravity(self):
        observed = peak_shift_row_python_antigravity(periodic_row_antigravity())
        self.assertEqual(observed, (8, 10, True, True, 2))

    def test_offset_and_positive_scale_authority_antigravity(self):
        row = periodic_row_antigravity()
        transformed = 7.0 + 3.0 * row
        self.assertEqual(
            peak_shift_row_python_antigravity(row),
            peak_shift_row_python_antigravity(transformed),
        )

    def test_reference_rejects_invalid_and_degenerate_inputs_antigravity(self):
        fit_rows, query_rows = valid_fit_query_antigravity()
        wide_fit = np.ones((3, 141), dtype=np.float64)
        invalid_calls = (
            lambda: score_acf_peak_lag_python_reference_antigravity(
                fit_rows[0], query_rows
            ),
            lambda: score_acf_peak_lag_python_reference_antigravity(
                fit_rows[:1], query_rows
            ),
            lambda: score_acf_peak_lag_python_reference_antigravity(
                fit_rows[:, :139], query_rows
            ),
            lambda: score_acf_peak_lag_python_reference_antigravity(
                wide_fit, query_rows
            ),
            lambda: score_acf_peak_lag_python_reference_antigravity(
                fit_rows.astype(np.float32), query_rows
            ),
            lambda: score_acf_peak_lag_python_reference_antigravity(
                fit_rows, query_rows[0]
            ),
            lambda: score_acf_peak_lag_python_reference_antigravity(
                fit_rows, query_rows[:, :139]
            ),
            lambda: score_acf_peak_lag_python_reference_antigravity(
                fit_rows, np.ascontiguousarray(query_rows, dtype=np.float32)
            ),
        )
        for invalid_call in invalid_calls:
            with self.assertRaises(ValueError):
                invalid_call()

        nan_query = query_rows.copy()
        nan_query[0, 10] = np.nan
        inf_query = query_rows.copy()
        inf_query[1, 100] = np.inf
        constant = np.ones((2, 140), dtype=np.float64)
        for bad_query in (nan_query, inf_query):
            with self.assertRaises(ValueError):
                score_acf_peak_lag_python_reference_antigravity(fit_rows, bad_query)
        with self.assertRaises(ValueError):
            score_acf_peak_lag_python_reference_antigravity(constant, query_rows)

    def test_reference_matches_numba_exact_discrete_fields_and_raw_tolerance_antigravity(self):
        fit_rows, query_rows = valid_fit_query_antigravity()
        expected = score_acf_peak_lag_python_reference_antigravity(fit_rows, query_rows)
        actual = score_acf_peak_lag_numba_antigravity(fit_rows, query_rows)

        self.assertEqual(expected.backend, "python_reference")
        self.assertEqual(actual.backend, "numba")
        self.assertEqual(
            tuple(expected.fit_early_peak_lags), tuple(actual.fit_early_peak_lags)
        )
        self.assertEqual(
            tuple(expected.fit_late_peak_lags), tuple(actual.fit_late_peak_lags)
        )
        self.assertEqual(
            expected.fit_peak_presence.tolist(), actual.fit_peak_presence.tolist()
        )
        self.assertEqual(
            tuple(expected.query_early_peak_lags), tuple(actual.query_early_peak_lags)
        )
        self.assertEqual(
            tuple(expected.query_late_peak_lags), tuple(actual.query_late_peak_lags)
        )
        self.assertEqual(
            expected.query_peak_presence.tolist(), actual.query_peak_presence.tolist()
        )
        self.assertEqual(tuple(expected.fit_shifts), tuple(actual.fit_shifts))
        self.assertEqual(tuple(expected.query_shifts), tuple(actual.query_shifts))
        self.assertEqual(tuple(expected.query_ranks), tuple(actual.query_ranks))
        self.assertEqual(
            expected.confirmation_mask.tolist(), actual.confirmation_mask.tolist()
        )
        self.assertLessEqual(max_abs_raw_error_antigravity(actual, expected), 1e-12)

    def test_empty_query_preserves_canonical_result_shapes_antigravity(self):
        fit_rows, _ = valid_fit_query_antigravity()
        empty_query_rows = np.empty((0, 140), dtype=np.float64)
        expected = score_acf_peak_lag_python_reference_antigravity(
            fit_rows, empty_query_rows
        )
        actual = score_acf_peak_lag_numba_antigravity(fit_rows, empty_query_rows)

        for observed in (expected, actual):
            self.assertEqual(observed.query_early_peak_lags.shape, (0,))
            self.assertEqual(observed.query_late_peak_lags.shape, (0,))
            self.assertEqual(observed.query_peak_presence.shape, (0, 2))
            self.assertEqual(observed.query_peak_presence.dtype, np.dtype(np.bool_))
            self.assertEqual(observed.query_shifts.shape, (0,))
            self.assertEqual(observed.query_raw_shift.shape, (0,))
            self.assertEqual(observed.query_ranks.shape, (0,))
            self.assertEqual(observed.confirmation_mask.shape, (0,))
            self.assertTrue(math.isfinite(observed.fit_normal_max_raw_shift))

    def test_confirmation_is_strictly_above_positive_fit_maximum_antigravity(self):
        fit_rows, query_rows = valid_fit_query_antigravity()
        observed = score_acf_peak_lag_python_reference_antigravity(fit_rows, query_rows)
        self.assertEqual(observed.query_shifts.shape, (2,))
        self.assertEqual(observed.query_ranks.shape, (2,))
        for raw_value, rank_value, confirmed in zip(
            observed.query_raw_shift,
            observed.query_ranks,
            observed.confirmation_mask,
        ):
            self.assertTrue(math.isfinite(float(raw_value)))
            self.assertTrue(math.isfinite(float(rank_value)))
            self.assertGreaterEqual(float(rank_value), 1.0 / 4.0)
            self.assertLessEqual(float(rank_value), 1.0)
            self.assertIs(bool(confirmed), bool(confirmed))
            self.assertEqual(
                bool(confirmed),
                float(raw_value) > 0.0
                and float(raw_value) > float(observed.fit_normal_max_raw_shift),
            )


if __name__ == "__main__":
    unittest.main()
