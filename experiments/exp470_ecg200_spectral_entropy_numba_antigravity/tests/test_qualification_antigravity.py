from __future__ import annotations

from types import MappingProxyType
import inspect
import unittest

import numpy as np


class Exp470QualificationTestAntigravity(unittest.TestCase):
    def test_compiled_pair_credit_counts_fixed_broken_and_ties_antigravity(self) -> None:
        """Break caught: ties or pair direction produce a false repair count."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import (
            count_pair_repairs_antigravity,
        )

        result = count_pair_repairs_antigravity(
            np.array((0.2, 0.8), dtype=np.float64),
            np.array((0.2, 0.9), dtype=np.float64),
            np.array((0.3, 0.7), dtype=np.float64),
            np.array((0.1, 0.95), dtype=np.float64),
        )
        self.assertEqual((result.fixed, result.broken), (0, 1))
        fixed = count_pair_repairs_antigravity(
            np.array((0.5, 0.8), dtype=np.float64),
            np.array((0.5, 0.7), dtype=np.float64),
            np.array((0.6, 0.6), dtype=np.float64),
            np.array((0.7, 0.9), dtype=np.float64),
        )
        self.assertEqual((fixed.fixed, fixed.broken), (3, 0))
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import _count_pair_repairs_numba_antigravity
        self.assertTrue(_count_pair_repairs_numba_antigravity.signatures)

    def test_pair_vectors_reject_length_dtype_and_nonfinite_antigravity(self) -> None:
        """Break caught: malformed score evidence enters pair counting."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import (
            validate_four_score_vectors_antigravity,
        )

        valid = np.array((0.1,), dtype=np.float64)
        for bad in (
            np.array((), dtype=np.float64), np.array((0.1,), dtype=np.float32),
            np.array((np.nan,), dtype=np.float64),
        ):
            with self.subTest(dtype=bad.dtype, size=bad.size):
                with self.assertRaises((TypeError, ValueError)):
                    validate_four_score_vectors_antigravity(valid, valid, bad, valid)
        with self.assertRaises(ValueError):
            validate_four_score_vectors_antigravity(valid, valid, valid, np.array((0.1, 0.2), dtype=np.float64))

    def test_blocked_folds_cover_69_rows_once_antigravity(self) -> None:
        """Break caught: a query block is missing, duplicated, or noncontiguous."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import (
            blocked_fold_indices_antigravity,
        )

        folds = blocked_fold_indices_antigravity()
        self.assertEqual(tuple(len(query) for _fit, query in folds), (14, 14, 14, 14, 13))
        np.testing.assert_array_equal(np.concatenate(tuple(query for _fit, query in folds)), np.arange(69, dtype=np.int64))
        for fit, query in folds:
            self.assertFalse(fit.flags.writeable)
            self.assertFalse(query.flags.writeable)

    def test_ten_synthetic_records_preserve_order_coverage_and_aggregate_counts_antigravity(self) -> None:
        """Break caught: qualification pools, reorders, or loses a synthetic fold's source evidence."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import (
            blocked_fold_indices_antigravity, qualify_records_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import FoldMethodScoresAntigravity

        train = np.zeros((69, 140), dtype=np.float64)
        changes = ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
        records = {}
        for seed in (20260721, 20261721):
            for fold, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
                size = query.size
                zeros = np.zeros(size, dtype=np.float64)
                ranks = np.ones(size, dtype=np.float64)
                confirmed = np.zeros(size, dtype=np.bool_)
                records[(seed, fold)] = FoldMethodScoresAntigravity(
                    seed=seed, fold_index=fold, methods=("C0", "E11", "C11"),
                    original={"C0": zeros, "E11": ranks, "C11": zeros},
                    primary={method: {change: values for change in changes} for method, values in (("C0", zeros), ("E11", ranks), ("C11", zeros))},
                    fit_raw_entropy=np.zeros(fit.size, dtype=np.float64),
                    query_raw_entropy={scope: zeros for scope in ("normal",) + changes},
                    fit_normal_max_raw_entropy=0.0,
                    confirmed={scope: confirmed for scope in ("normal",) + changes},
                    input_train140=train, input_fit_indices=fit, input_query_indices=query, backend="numba",
                )
        qualification = qualify_records_antigravity(MappingProxyType(records))
        self.assertEqual(tuple((row.seed, row.fold_index) for row in qualification.fold_results), tuple(records))
        self.assertEqual(dict(qualification.period_fixed), {20260721: 0, 20261721: 0})
        self.assertEqual(dict(qualification.period_broken), {20260721: 0, 20261721: 0})
        self.assertEqual(qualification.label_query_count, 0)

    def test_qualification_requires_byte_exact_fallback_and_compiles_normal_count_antigravity(self) -> None:
        """Break caught: signed-zero C11 fallback drift or Python confirmation counting escapes qualification."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import (
            _count_confirmed_numba_antigravity, qualify_records_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import FoldMethodScoresAntigravity

        train = np.zeros((69, 140), dtype=np.float64)
        changes = ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
        records = {}
        for seed in (20260721, 20261721):
            for fold, (fit, query) in enumerate(self._folds_antigravity()):
                size = query.size
                zeros = np.zeros(size, dtype=np.float64)
                ranks = np.ones(size, dtype=np.float64)
                confirmed = np.zeros(size, dtype=np.bool_)
                c0 = np.array(zeros, copy=True)
                c0[0] = -0.0
                c11 = np.array(c0, copy=True)
                records[(seed, fold)] = FoldMethodScoresAntigravity(seed=seed, fold_index=fold, methods=("C0", "E11", "C11"), original={"C0": c0, "E11": ranks, "C11": c11}, primary={method: {change: values for change in changes} for method, values in (("C0", zeros), ("E11", ranks), ("C11", zeros))}, fit_raw_entropy=zeros if fit.size == size else np.zeros(fit.size, dtype=np.float64), query_raw_entropy={scope: zeros for scope in ("normal",) + changes}, fit_normal_max_raw_entropy=0.0, confirmed={scope: confirmed for scope in ("normal",) + changes}, input_train140=train, input_fit_indices=fit, input_query_indices=query, backend="numba")
        drifted = records[(20260721, 0)].original["C11"]
        drifted.setflags(write=True)
        drifted[0] = 0.0
        drifted.setflags(write=False)
        qualification = qualify_records_antigravity(MappingProxyType(records))
        self.assertFalse(qualification.fallback_exact)
        self.assertTrue(_count_confirmed_numba_antigravity.signatures)
        self.assertNotIn("np.count_nonzero", inspect.getsource(qualify_records_antigravity))

    def test_nonzero_ten_record_fixture_aggregates_by_seed_and_change_direction_antigravity(self) -> None:
        """Break caught: period/protected/confirmation counts are pooled, reversed, or assigned to the wrong seed."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import qualify_records_antigravity
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import FoldMethodScoresAntigravity

        train = np.zeros((69, 140), dtype=np.float64)
        changes = ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
        records = {}
        for seed in (20260721, 20261721):
            for fold, (fit, query) in enumerate(self._folds_antigravity()):
                size = query.size
                c0_normal = np.zeros(size, dtype=np.float64)
                c11_normal = np.zeros(size, dtype=np.float64)
                raw = {scope: np.zeros(size, dtype=np.float64) for scope in ("normal",) + changes}
                c0_primary = {name: np.zeros(size, dtype=np.float64) for name in changes}
                c11_primary = {name: np.zeros(size, dtype=np.float64) for name in changes}
                if (seed, fold) in ((20260721, 0), (20261721, 1)):
                    c0_normal.fill(0.9); c11_normal.fill(1.0); raw["normal"].fill(1.0)
                    if (seed, fold) == (20260721, 0):
                        c0_primary["period_change"].fill(0.0); c11_primary["period_change"].fill(1.0); raw["period_change"].fill(1.0)
                        c0_primary["level_shift"].fill(1.0); c11_primary["level_shift"].fill(1.0)
                    else:
                        c0_primary["period_change"].fill(1.0); c11_primary["period_change"].fill(1.0)
                        c0_primary["amplitude_scaling"].fill(1.0); c11_primary["amplitude_scaling"].fill(1.0)
                ranks = {scope: np.ones(size, dtype=np.float64) for scope in raw}
                confirmed = {scope: raw[scope] > 0.0 for scope in raw}
                records[(seed, fold)] = FoldMethodScoresAntigravity(seed=seed, fold_index=fold, methods=("C0", "E11", "C11"), original={"C0": c0_normal, "E11": ranks["normal"], "C11": c11_normal}, primary={"C0": c0_primary, "E11": {name: ranks[name] for name in changes}, "C11": c11_primary}, fit_raw_entropy=np.zeros(fit.size, dtype=np.float64), query_raw_entropy=raw, fit_normal_max_raw_entropy=0.0, confirmed=confirmed, input_train140=train, input_fit_indices=fit, input_query_indices=query, backend="numba")
        qualification = qualify_records_antigravity(MappingProxyType(records))
        self.assertEqual(dict(qualification.period_fixed), {20260721: 196, 20261721: 0})
        self.assertEqual(dict(qualification.period_broken), {20260721: 0, 20261721: 196})
        self.assertEqual(dict(qualification.protected_broken), {"level_shift": 196, "gradual_drift": 0, "amplitude_scaling": 196})
        self.assertEqual(dict(qualification.normal_confirmations), {20260721: 14, 20261721: 14})

    @staticmethod
    def _folds_antigravity():
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.qualification_antigravity import blocked_fold_indices_antigravity
        return blocked_fold_indices_antigravity()


if __name__ == "__main__":
    unittest.main()
