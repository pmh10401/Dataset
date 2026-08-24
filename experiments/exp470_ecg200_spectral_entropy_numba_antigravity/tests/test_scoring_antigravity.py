from __future__ import annotations

import inspect
from types import MappingProxyType
import unittest

import numpy as np


class Exp470ScoringTestAntigravity(unittest.TestCase):
    def test_compiled_c11_rule_keeps_c0_on_unconfirmed_or_not_larger_rank_antigravity(self) -> None:
        """Break caught: C11 changes fallback values or accepts a non-improving E11 rank."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import (
            _combine_c11_numba_antigravity,
        )

        c0 = np.array((0.2, 0.7, 0.4, 0.1), dtype=np.float64)
        ranks = np.array((0.9, 0.8, 0.3, 0.6), dtype=np.float64)
        confirmed = np.array((True, False, True, True), dtype=np.bool_)
        actual = _combine_c11_numba_antigravity(c0, ranks, confirmed)
        np.testing.assert_array_equal(actual, np.array((0.9, 0.7, 0.4, 0.6), dtype=np.float64))
        self.assertTrue(_combine_c11_numba_antigravity.signatures)

    def test_fold_record_freezes_exact_numba_e11_evidence_antigravity(self) -> None:
        """Break caught: a fold accepts mutable, reordered, or non-Numba scoring evidence."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import (
            FoldMethodScoresAntigravity,
        )

        train = np.zeros((69, 140), dtype=np.float64)
        fit = np.arange(2, 69, dtype=np.int64)
        query = np.array((0, 1), dtype=np.int64)
        raw = {scope: np.array((0.1, 0.3), dtype=np.float64) for scope in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")}
        confirmed = {scope: np.array((False, True), dtype=np.bool_) for scope in raw}
        c0 = np.array((0.4, 0.8), dtype=np.float64)
        ranks = np.array((1.0 / 68.0, 1.0), dtype=np.float64)
        c11 = np.array((0.4, 1.0), dtype=np.float64)
        record = FoldMethodScoresAntigravity(
            seed=20260721, fold_index=0, methods=("C0", "E11", "C11"),
            original={"C0": c0, "E11": ranks, "C11": c11},
            primary={method: {change: values for change in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")} for method, values in (("C0", c0), ("E11", ranks), ("C11", c11))},
            fit_raw_entropy=np.full(fit.size, 0.2, dtype=np.float64),
            query_raw_entropy=raw, fit_normal_max_raw_entropy=0.2, confirmed=confirmed,
            input_train140=train, input_fit_indices=fit, input_query_indices=query, backend="numba",
        )
        self.assertIsInstance(record.original, MappingProxyType)
        self.assertEqual(tuple(record.original), ("C0", "E11", "C11"))
        self.assertEqual(tuple(record.query_raw_entropy), ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change"))
        for values in (*record.original.values(), record.fit_raw_entropy, *record.query_raw_entropy.values()):
            self.assertTrue(values.flags.owndata)
            self.assertTrue(values.flags.c_contiguous)
            self.assertFalse(values.flags.writeable)
        with self.assertRaises(TypeError):
            record.original["C0"] = c0  # type: ignore[index]

    def test_score_signature_has_no_backend_selector_antigravity(self) -> None:
        """Break caught: scoring exposes a fallback backend or a mutable method order."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import (
            score_fold_methods_antigravity,
        )

        signature = inspect.signature(score_fold_methods_antigravity)
        self.assertEqual(tuple(signature.parameters), ("train140", "fit_indices", "query_indices", "seed", "fold_index", "authority"))
        for name in ("seed", "fold_index", "authority"):
            self.assertIs(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_one_synthetic_fold_uses_c0_and_numba_e11_with_exact_fallback_antigravity(self) -> None:
        """Break caught: the integration path skips C0, changes E11 provenance, or mutates fallback bytes."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.predecessor_antigravity import capture_exp465_authority_antigravity
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.scoring_antigravity import score_fold_methods_antigravity

        train = np.random.default_rng(4704).normal(size=(69, 140)).astype(np.float64)
        result = score_fold_methods_antigravity(
            train, np.arange(55, dtype=np.int64), np.arange(55, 69, dtype=np.int64),
            seed=20260721, fold_index=0, authority=capture_exp465_authority_antigravity(),
        )
        self.assertEqual((result.methods, result.backend), (("C0", "E11", "C11"), "numba"))
        for scope in ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change"):
            c0 = result.original["C0"] if scope == "normal" else result.primary["C0"][scope]
            c11 = result.original["C11"] if scope == "normal" else result.primary["C11"][scope]
            self.assertEqual(c11[~result.confirmed[scope]].tobytes(), c0[~result.confirmed[scope]].tobytes())


if __name__ == "__main__":
    unittest.main()
