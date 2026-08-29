from __future__ import annotations

import inspect
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.phase_registration_antigravity import (
    score_phase_registration_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
    score_c0_fold_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.synthetic_antigravity import (
    primary_rows_antigravity,
)


_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


class ScoringAntigravityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(4662026)
        cls.train = rng.standard_normal((20, 140), dtype=np.float64)
        cls.train.setflags(write=False)
        cls.fit = np.arange(6, dtype=np.int64)
        cls.query = np.array((6, 7, 8, 10, 12), dtype=np.int64)
        cls.authority = capture_exp465_authority_antigravity()

    def test_fixed_c0_p9_c9_assembly_and_raw_phase_evidence_antigravity(self) -> None:
        """Break caught: C0 is replaced, P9 drifts, or C9 fuses outside confirmation."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.scoring_antigravity import (
            FoldMethodScoresAntigravity,
            score_fold_methods_antigravity,
        )

        signature = inspect.signature(score_fold_methods_antigravity)
        self.assertEqual(
            tuple(signature.parameters),
            ("train140", "fit_indices", "query_indices", "seed", "fold_index", "authority"),
        )
        for name in ("seed", "fold_index", "authority"):
            self.assertIs(
                signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )

        result = score_fold_methods_antigravity(
            self.train,
            self.fit,
            self.query,
            seed=20260721,
            fold_index=0,
            authority=self.authority,
        )
        self.assertIs(type(result), FoldMethodScoresAntigravity)
        self.assertEqual(result.methods, ("C0", "P9", "C9"))
        self.assertEqual((result.seed, result.fold_index), (20260721, 0))
        self.assertEqual(tuple(result.original), result.methods)
        self.assertEqual(tuple(result.primary), result.methods)
        self.assertEqual(tuple(result.confirmed), _SCOPES_ANTIGRAVITY)
        self.assertEqual(tuple(result.query_margins), _SCOPES_ANTIGRAVITY)

        c0 = score_c0_fold_antigravity(
            self.train,
            self.fit,
            self.query,
            seed=20260721,
            fold_index=0,
            authority=self.authority,
        )
        self.assertEqual(result.original["C0"].tobytes(), c0.normal_scores.tobytes())
        for change_type in _CHANGE_TYPES_ANTIGRAVITY:
            self.assertEqual(
                result.primary["C0"][change_type].tobytes(),
                c0.primary_scores[change_type].tobytes(),
            )

        fit_rows = self.train[self.fit]
        query_rows = {"normal": self.train[self.query]}
        generated = tuple(primary_rows_antigravity(row) for row in self.train[self.query])
        query_rows.update(
            {
                change_type: np.stack(
                    tuple(changes[change_type] for changes in generated),
                )
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            }
        )
        for scope in _SCOPES_ANTIGRAVITY:
            phase = score_phase_registration_antigravity(fit_rows, query_rows[scope])
            np.testing.assert_array_equal(result.fit_margins, phase.fit_margins)
            np.testing.assert_array_equal(result.query_margins[scope], phase.query_margins)
            np.testing.assert_array_equal(result.confirmed[scope], phase.confirmed)
            actual_p9 = (
                result.original["P9"]
                if scope == "normal"
                else result.primary["P9"][scope]
            )
            np.testing.assert_array_equal(actual_p9, phase.ranks)

        np.testing.assert_array_equal(
            result.original["C9"],
            np.where(
                result.confirmed["normal"],
                np.maximum(result.original["C0"], result.original["P9"]),
                result.original["C0"],
            ),
        )
        for scope in _CHANGE_TYPES_ANTIGRAVITY:
            np.testing.assert_array_equal(
                result.primary["C9"][scope],
                np.where(
                    result.confirmed[scope],
                    np.maximum(
                        result.primary["C0"][scope],
                        result.primary["P9"][scope],
                    ),
                    result.primary["C0"][scope],
                ),
            )

    def test_fold_score_owns_and_freezes_every_array_antigravity(self) -> None:
        """Break caught: score or replay evidence aliases mutable caller storage."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.scoring_antigravity import (
            score_fold_methods_antigravity,
        )

        result = score_fold_methods_antigravity(
            self.train,
            self.fit,
            self.query,
            seed=20260721,
            fold_index=0,
            authority=self.authority,
        )
        self.assertIsInstance(result.original, MappingProxyType)
        self.assertIsInstance(result.primary, MappingProxyType)
        self.assertIsInstance(result.confirmed, MappingProxyType)
        self.assertIsInstance(result.query_margins, MappingProxyType)
        arrays = (
            result.input_train140,
            result.input_fit_indices,
            result.input_query_indices,
            result.fit_margins,
            *result.original.values(),
            *(values for method in result.primary.values() for values in method.values()),
            *result.confirmed.values(),
            *result.query_margins.values(),
        )
        for values in arrays:
            self.assertTrue(values.flags.owndata)
            self.assertTrue(values.flags.c_contiguous)
            self.assertFalse(values.flags.writeable)
        self.assertFalse(np.shares_memory(result.input_train140, self.train))
        self.assertFalse(np.shares_memory(result.input_fit_indices, self.fit))
        self.assertFalse(np.shares_memory(result.input_query_indices, self.query))
        normal_false = ~result.confirmed["normal"]
        self.assertEqual(
            result.original["C9"][normal_false].tobytes(),
            result.original["C0"][normal_false].tobytes(),
        )


if __name__ == "__main__":
    unittest.main()
