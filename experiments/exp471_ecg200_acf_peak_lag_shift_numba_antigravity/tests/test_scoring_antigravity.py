from __future__ import annotations

import dataclasses
import inspect
import unittest

import numpy as np


class Exp471ScoringTestAntigravity(unittest.TestCase):
    def test_compiled_c12_keeps_byte_exact_c0_fallback_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.scoring_antigravity import (
            _combine_c12_numba_antigravity,
        )

        c0 = np.array((-0.0, 0.7, 0.4, 0.1), dtype=np.float64)
        ranks = np.array((0.9, 0.8, 0.3, 0.6), dtype=np.float64)
        confirmed = np.array((False, False, True, True), dtype=np.bool_)
        actual = _combine_c12_numba_antigravity(c0, ranks, confirmed)
        expected = np.array((-0.0, 0.7, 0.4, 0.6), dtype=np.float64)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertTrue(_combine_c12_numba_antigravity.signatures)

    def test_score_signature_has_no_backend_selector_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.scoring_antigravity import (
            score_fold_methods_antigravity,
        )

        signature = inspect.signature(score_fold_methods_antigravity)
        self.assertEqual(
            tuple(signature.parameters),
            ("train140", "fit_indices", "query_indices", "seed", "fold_index", "authority"),
        )
        for name in ("seed", "fold_index", "authority"):
            self.assertIs(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_one_synthetic_fold_retains_acf_evidence_and_exact_fallback_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.predecessor_antigravity import (
            capture_exp465_authority_antigravity,
        )
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.scoring_antigravity import (
            score_fold_methods_antigravity,
        )

        train = np.random.default_rng(4714).normal(size=(69, 140)).astype(np.float64)
        result = score_fold_methods_antigravity(
            train,
            np.arange(55, dtype=np.int64),
            np.arange(55, 69, dtype=np.int64),
            seed=20260721,
            fold_index=0,
            authority=capture_exp465_authority_antigravity(),
        )
        self.assertEqual((result.methods, result.backend), (("C0", "E12", "C12"), "numba"))
        self.assertEqual(result.fit_peak_presence.shape, (55, 2))
        self.assertEqual(result.query_peak_presence["normal"].shape, (14, 2))
        for scope in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change"):
            c0 = result.original["C0"] if scope == "normal" else result.primary["C0"][scope]
            e12 = result.original["E12"] if scope == "normal" else result.primary["E12"][scope]
            c12 = result.original["C12"] if scope == "normal" else result.primary["C12"][scope]
            fallback = (~result.confirmed[scope]) | (e12 <= c0)
            self.assertEqual(c12[fallback].tobytes(), c0[fallback].tobytes())
            np.testing.assert_array_equal(
                result.query_shifts[scope],
                result.query_late_peak_lags[scope] - result.query_early_peak_lags[scope],
            )

        drifted = dict(result.query_shifts)
        changed = np.array(drifted["normal"], copy=True)
        changed[0] += 1
        drifted["normal"] = changed
        with self.assertRaisesRegex(ValueError, "shift"):
            dataclasses.replace(result, query_shifts=drifted)


if __name__ == "__main__":
    unittest.main()
