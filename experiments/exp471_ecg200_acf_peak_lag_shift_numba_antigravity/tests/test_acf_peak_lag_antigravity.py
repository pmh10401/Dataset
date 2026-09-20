import math
import unittest

import numba
import numpy as np

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acf_peak_lag_antigravity import (
    ACFPeakLagScoresAntigravity,
    assemble_raw_rank_numba_antigravity,
    dominant_peak_from_acf_numba_antigravity,
    peak_shift_row_numba_antigravity,
    score_acf_peak_lag_numba_antigravity,
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
            periodic_row_antigravity(10.0, 10.0),
        )
    )
    return np.ascontiguousarray(fit_rows, dtype=np.float64), np.ascontiguousarray(
        query_rows, dtype=np.float64
    )


class Exp471AcfPeakLagNumbaTestAntigravity(unittest.TestCase):
    def test_known_peak_shift_antigravity(self):
        observed = peak_shift_row_numba_antigravity(periodic_row_antigravity())
        self.assertEqual(observed, (8, 10, True, True, 2))

    def test_peak_rule_uses_smallest_tie_and_zero_sentinel_antigravity(self):
        tie = np.zeros(25, dtype=np.float64)
        tie[1], tie[2], tie[3], tie[7], tie[8], tie[9] = (
            0.1,
            0.8,
            0.1,
            0.1,
            0.8,
            0.1,
        )
        self.assertEqual(dominant_peak_from_acf_numba_antigravity(tie), (3, True))
        self.assertEqual(
            dominant_peak_from_acf_numba_antigravity(np.linspace(1.0, 0.1, 25)),
            (0, False),
        )

    def test_offset_and_positive_scale_leave_shift_unchanged_antigravity(self):
        row = periodic_row_antigravity()
        transformed = 7.0 + 3.0 * row
        self.assertEqual(
            peak_shift_row_numba_antigravity(row),
            peak_shift_row_numba_antigravity(transformed),
        )

    def test_rejects_invalid_inputs_antigravity(self):
        fit_rows, query_rows = valid_fit_query_antigravity()
        wide_fit = np.ones((3, 141), dtype=np.float64)
        cases = (
            (fit_rows[0], query_rows),
            (fit_rows[:1], query_rows),
            (fit_rows[:, :139], query_rows),
            (wide_fit, query_rows),
            (fit_rows.astype(np.float32), query_rows),
        )
        for index, (bad_fit, bad_query) in enumerate(cases):
            with self.assertRaises(ValueError, msg=f"invalid fit case {index}"):
                score_acf_peak_lag_numba_antigravity(bad_fit, bad_query)

        with self.assertRaises(ValueError):
            score_acf_peak_lag_numba_antigravity(fit_rows, query_rows[0])
        with self.assertRaises(ValueError):
            short_query = query_rows[:, :139].copy()
            score_acf_peak_lag_numba_antigravity(fit_rows, short_query)
        with self.assertRaises(ValueError):
            float32_query = np.ascontiguousarray(query_rows, dtype=np.float32)
            score_acf_peak_lag_numba_antigravity(fit_rows, float32_query)
        non_finite_fit = fit_rows.copy()
        non_finite_fit[0, 55] = np.inf
        with self.assertRaises(ValueError):
            score_acf_peak_lag_numba_antigravity(non_finite_fit, query_rows)

    def test_rejects_constant_and_zero_mad_windows_antigravity(self):
        fit_rows, query_rows = valid_fit_query_antigravity()
        constant_row = np.ones(140, dtype=np.float64)
        constant_fit = np.repeat(constant_row[None, :], 2, axis=0)
        with self.assertRaises(ValueError):
            score_acf_peak_lag_numba_antigravity(constant_fit, query_rows)

        early_constant = np.zeros(140, dtype=np.float64)
        early_constant[84:140] = periodic_row_antigravity()[84:140]
        early_constant_fit = np.repeat(early_constant[None, :], 2, axis=0)
        with self.assertRaises(ValueError):
            score_acf_peak_lag_numba_antigravity(early_constant_fit, query_rows)

        late_constant = periodic_row_antigravity().copy()
        late_constant[84:140] = 2.0
        late_constant_fit = np.repeat(late_constant[None, :], 2, axis=0)
        with self.assertRaises(ValueError):
            score_acf_peak_lag_numba_antigravity(late_constant_fit, query_rows)

    def test_result_arrays_have_frozen_types_and_shapes_antigravity(self):
        fit_rows, query_rows = valid_fit_query_antigravity()
        observed = score_acf_peak_lag_numba_antigravity(fit_rows, query_rows)
        self.assertIsInstance(observed, ACFPeakLagScoresAntigravity)
        self.assertEqual(observed.backend, "numba")
        self.assertIsInstance(observed.fit_normal_max_raw_shift, float)

        expected_shapes = {
            "fit_early_peak_lags": (3,),
            "fit_late_peak_lags": (3,),
            "fit_peak_presence": (3, 2),
            "query_early_peak_lags": (2,),
            "query_late_peak_lags": (2,),
            "query_peak_presence": (2, 2),
            "fit_shifts": (3,),
            "query_shifts": (2,),
            "fit_raw_shift": (3,),
            "query_raw_shift": (2,),
            "query_ranks": (2,),
            "confirmation_mask": (2,),
        }
        integer_fields = {
            "fit_early_peak_lags",
            "fit_late_peak_lags",
            "query_early_peak_lags",
            "query_late_peak_lags",
            "fit_shifts",
            "query_shifts",
        }
        boolean_fields = {
            "fit_peak_presence",
            "query_peak_presence",
            "confirmation_mask",
        }
        for field_name, expected_shape in expected_shapes.items():
            value = getattr(observed, field_name)
            self.assertIsInstance(value, np.ndarray, field_name)
            self.assertEqual(value.shape, expected_shape, field_name)
            self.assertEqual(value.flags.c_contiguous, True, field_name)
            self.assertEqual(value.flags.writeable, False, field_name)
            if field_name in integer_fields:
                self.assertEqual(value.dtype, np.dtype(np.int64), field_name)
            elif field_name in boolean_fields:
                self.assertEqual(value.dtype, np.dtype(np.bool_), field_name)
            else:
                self.assertEqual(value.dtype, np.dtype(np.float64), field_name)

        with self.assertRaises(ValueError):
            observed.fit_shifts[0] = 99
        with self.assertRaises(ValueError):
            observed.query_ranks[0] = 99.0
        with self.assertRaises(ValueError):
            observed.confirmation_mask[0] = False

    def test_public_kernels_are_float64_nopython_dispatchers_antigravity(self):
        self.assertIsInstance(
            dominant_peak_from_acf_numba_antigravity,
            numba.core.dispatcher.Dispatcher,
        )
        self.assertIsInstance(
            peak_shift_row_numba_antigravity,
            numba.core.dispatcher.Dispatcher,
        )
        self.assertIsInstance(
            assemble_raw_rank_numba_antigravity,
            numba.core.dispatcher.Dispatcher,
        )

        tie = np.zeros(25, dtype=np.float64)
        tie[5] = 1.0
        dominant_peak_from_acf_numba_antigravity(tie)
        peak_shift_row_numba_antigravity(periodic_row_antigravity())
        fit_shifts = np.asarray((0, 2, 4), dtype=np.float64)
        query_shifts = np.asarray((6,), dtype=np.float64)
        assemble_raw_rank_numba_antigravity(fit_shifts, query_shifts)

        for kernel in (
            dominant_peak_from_acf_numba_antigravity,
            peak_shift_row_numba_antigravity,
            assemble_raw_rank_numba_antigravity,
        ):
            self.assertGreater(len(kernel.nopython_signatures), 0)
            self.assertTrue(
                any("float64" in str(signature) for signature in kernel.nopython_signatures),
                (kernel.__name__, kernel.nopython_signatures),
            )

    def test_assemble_raw_rank_matches_preregistered_small_case_antigravity(self):
        fit_shifts = np.asarray((-2, 0, 2), dtype=np.float64)
        query_shifts = np.asarray((4, -4, 0), dtype=np.float64)
        fit_raw, query_raw, ranks, confirmed, maximum = assemble_raw_rank_numba_antigravity(
            fit_shifts, query_shifts
        )
        np.testing.assert_allclose(fit_raw, (3.0, 0.0, 3.0), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(query_raw, (4.0, 4.0, 0.0), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(ranks, (1.0, 1.0, 0.5), rtol=0.0, atol=0.0)
        self.assertEqual(confirmed.tolist(), [True, True, False])
        self.assertEqual(maximum, 3.0)


if __name__ == "__main__":
    unittest.main()
