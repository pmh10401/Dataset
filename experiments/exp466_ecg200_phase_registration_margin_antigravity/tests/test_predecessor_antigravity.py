from __future__ import annotations

import inspect
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp465_ecg200_c8_external_validation_antigravity.scoring_antigravity import (
    score_fold_methods_antigravity as score_exp465_fold_methods_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.predecessor_antigravity import (
    C0FoldScoresAntigravity,
    Exp465AuthorityAntigravity,
    capture_exp465_authority_antigravity,
    score_c0_fold_antigravity,
)


_CHANGE_ORDER_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_CLOSEOUT_COMMIT_ANTIGRAVITY = "23528af929da9dbf57cbcdf87c06abd58cb2363b"


class PredecessorAntigravityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = capture_exp465_authority_antigravity()

    def test_authority_binds_closeout_and_exact_exp465_source_antigravity(self) -> None:
        """Break caught: Exp466 accepts a different Exp465 closeout or scoring source."""
        self.assertIs(type(self.authority), Exp465AuthorityAntigravity)
        self.assertEqual(self.authority.closeout_commit, _CLOSEOUT_COMMIT_ANTIGRAVITY)
        self.assertIs(self.authority.score_source, score_exp465_fold_methods_antigravity)

        signature = inspect.signature(score_c0_fold_antigravity)
        self.assertEqual(
            tuple(signature.parameters),
            ("train140", "fit_indices", "query_indices", "seed", "fold_index", "authority"),
        )
        for name in ("seed", "fold_index", "authority"):
            self.assertIs(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_c0_normal_and_primary_scores_are_byte_equal_to_exp465_antigravity(self) -> None:
        """Break caught: the C0 adapter copies, recomputes, or changes Exp465 score bytes."""
        rng = np.random.default_rng(4662026)
        train140 = rng.standard_normal((20, 140), dtype=np.float64)
        train140.setflags(write=False)
        fit_indices = np.arange(6, dtype=np.int64)
        query_indices = np.array([6, 7, 8, 10, 12], dtype=np.int64)

        actual = score_c0_fold_antigravity(
            train140,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=0,
            authority=self.authority,
        )
        expected = score_exp465_fold_methods_antigravity(
            train140,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=0,
            authority=self.authority.predecessor_authority,
        )

        self.assertIs(type(actual), C0FoldScoresAntigravity)
        self.assertEqual(actual.normal_scores.tobytes(), expected.original["C0"].tobytes())
        self.assertEqual(tuple(actual.primary_scores), _CHANGE_ORDER_ANTIGRAVITY)
        self.assertIsInstance(actual.primary_scores, MappingProxyType)
        for name in _CHANGE_ORDER_ANTIGRAVITY:
            self.assertEqual(
                actual.primary_scores[name].tobytes(),
                expected.primary["C0"][name].tobytes(),
            )
        for values in (actual.normal_scores, *actual.primary_scores.values()):
            self.assertEqual(values.dtype, np.dtype(np.float64))
            self.assertFalse(values.flags.writeable)
            self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
