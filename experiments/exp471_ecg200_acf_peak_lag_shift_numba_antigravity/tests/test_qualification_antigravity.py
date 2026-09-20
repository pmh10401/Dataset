from __future__ import annotations

from types import MappingProxyType
import unittest

import numpy as np


class Exp471QualificationTestAntigravity(unittest.TestCase):
    def test_compiled_pair_credit_counts_fixed_broken_and_ties_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.qualification_antigravity import (
            _count_pair_repairs_numba_antigravity,
            count_pair_repairs_antigravity,
        )

        fixed = count_pair_repairs_antigravity(
            np.array((0.5, 0.8), dtype=np.float64),
            np.array((0.5, 0.7), dtype=np.float64),
            np.array((0.6, 0.6), dtype=np.float64),
            np.array((0.7, 0.9), dtype=np.float64),
        )
        self.assertEqual((fixed.fixed, fixed.broken), (3, 0))
        broken = count_pair_repairs_antigravity(
            np.array((0.2, 0.8), dtype=np.float64),
            np.array((0.2, 0.9), dtype=np.float64),
            np.array((0.3, 0.7), dtype=np.float64),
            np.array((0.1, 0.95), dtype=np.float64),
        )
        self.assertEqual((broken.fixed, broken.broken), (0, 1))
        self.assertTrue(_count_pair_repairs_numba_antigravity.signatures)

    def test_blocked_folds_cover_69_rows_once_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.qualification_antigravity import (
            blocked_fold_indices_antigravity,
        )

        folds = blocked_fold_indices_antigravity()
        self.assertEqual(tuple(query.size for _fit, query in folds), (14, 14, 14, 14, 13))
        np.testing.assert_array_equal(
            np.concatenate(tuple(query for _fit, query in folds)),
            np.arange(69, dtype=np.int64),
        )
        for fit, query in folds:
            self.assertFalse(fit.flags.writeable)
            self.assertFalse(query.flags.writeable)

    def test_ten_zero_records_aggregate_without_auc_gate_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.qualification_antigravity import (
            blocked_fold_indices_antigravity,
            qualify_records_antigravity,
        )
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.scoring_antigravity import (
            FoldMethodScoresAntigravity,
        )

        train = np.zeros((69, 140), dtype=np.float64)
        changes = ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
        scopes = ("normal",) + changes
        records = {}
        for seed in (20260721, 20261721):
            for fold, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
                size = query.size
                zeros = np.zeros(size, dtype=np.float64)
                fit_zeros = np.zeros(fit.size, dtype=np.float64)
                ranks = np.ones(size, dtype=np.float64)
                false = np.zeros(size, dtype=np.bool_)
                records[(seed, fold)] = FoldMethodScoresAntigravity(
                    seed=seed,
                    fold_index=fold,
                    methods=("C0", "E12", "C12"),
                    original={"C0": zeros, "E12": ranks, "C12": zeros},
                    primary={
                        "C0": {change: zeros for change in changes},
                        "E12": {change: ranks for change in changes},
                        "C12": {change: zeros for change in changes},
                    },
                    fit_early_peak_lags=np.zeros(fit.size, dtype=np.int64),
                    fit_late_peak_lags=np.zeros(fit.size, dtype=np.int64),
                    fit_peak_presence=np.zeros((fit.size, 2), dtype=np.bool_),
                    fit_shifts=np.zeros(fit.size, dtype=np.int64),
                    fit_raw_shift=fit_zeros,
                    query_early_peak_lags={scope: np.zeros(size, dtype=np.int64) for scope in scopes},
                    query_late_peak_lags={scope: np.zeros(size, dtype=np.int64) for scope in scopes},
                    query_peak_presence={scope: np.zeros((size, 2), dtype=np.bool_) for scope in scopes},
                    query_shifts={scope: np.zeros(size, dtype=np.int64) for scope in scopes},
                    query_raw_shift={scope: zeros for scope in scopes},
                    fit_normal_max_raw_shift=0.0,
                    confirmed={scope: false for scope in scopes},
                    input_train140=train,
                    input_fit_indices=fit,
                    input_query_indices=query,
                    backend="numba",
                )
        qualification = qualify_records_antigravity(MappingProxyType(records))
        self.assertEqual(tuple((row.seed, row.fold_index) for row in qualification.fold_results), tuple(records))
        self.assertEqual(dict(qualification.period_fixed), {20260721: 0, 20261721: 0})
        self.assertEqual(dict(qualification.period_broken), {20260721: 0, 20261721: 0})
        self.assertEqual(qualification.label_query_count, 0)
        self.assertTrue(qualification.fallback_exact)
        self.assertTrue(qualification.finite)


if __name__ == "__main__":
    unittest.main()
