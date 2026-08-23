"""Task 4 contracts for immutable C0/S10/C10 fold scoring."""

from __future__ import annotations

import dataclasses
import inspect
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.local_stft_antigravity import (
    score_local_stft_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
    score_c0_fold_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.scoring_antigravity import (
    APPROVED_BACKEND_DECISION_ANTIGRAVITY,
    FoldMethodScoresAntigravity,
    score_fold_methods_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.synthetic_antigravity import (
    primary_rows_antigravity,
)


_APPROVAL_SHA256_ANTIGRAVITY = (
    "6780edb2683c992b3df361c72e15542a6e974cf3e7cbbc0df65df2739c0203a9"
)
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


def _train_fixture_antigravity() -> np.ndarray:
    x = np.arange(140, dtype=np.float64)
    base = (
        0.01 * x
        + np.sin(2.0 * np.pi * x / 21.0)
        + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    )
    rows = base[None, :] + np.random.default_rng(4684).normal(
        scale=0.01,
        size=(69, 140),
    )
    result = np.asarray(rows, dtype=np.float64, order="C")
    result.setflags(write=False)
    return result


class ScoringAntigravityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = _train_fixture_antigravity()
        cls.fit = np.arange(55, dtype=np.int64)
        cls.query = np.arange(55, 69, dtype=np.int64)
        cls.authority = capture_exp465_authority_antigravity()
        cls.result = score_fold_methods_antigravity(
            cls.train,
            cls.fit,
            cls.query,
            seed=20260721,
            fold_index=0,
            authority=cls.authority,
        )

    def test_approved_numpy_decision_is_internal_and_not_a_selector_antigravity(self) -> None:
        """Break caught: Task 4 bypasses the approved fallback or exposes a backend knob."""

        signature = inspect.signature(score_fold_methods_antigravity)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "train140",
                "fit_indices",
                "query_indices",
                "seed",
                "fold_index",
                "authority",
            ),
        )
        for name in ("seed", "fold_index", "authority"):
            self.assertIs(
                signature.parameters[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )
        decision = APPROVED_BACKEND_DECISION_ANTIGRAVITY
        self.assertEqual(decision.selected_backend, "numpy")
        self.assertIs(decision.acceleration_gate_passed, False)
        self.assertIs(decision.fallback_authorized, True)
        self.assertEqual(decision.approval_evidence_sha256, _APPROVAL_SHA256_ANTIGRAVITY)
        self.assertEqual(self.result.backend, "numpy")

    def test_c0_s10_c10_use_one_fit_calibration_and_exact_scope_splits_antigravity(
        self,
    ) -> None:
        """Break caught: a query leaks into calibration or a scope is split/reordered."""

        result = self.result
        self.assertIs(type(result), FoldMethodScoresAntigravity)
        self.assertEqual((result.seed, result.fold_index), (20260721, 0))
        self.assertEqual(result.methods, ("C0", "S10", "C10"))
        self.assertEqual(tuple(result.original), result.methods)
        self.assertEqual(tuple(result.primary), result.methods)
        self.assertEqual(tuple(result.query_raw_js), _SCOPES_ANTIGRAVITY)
        self.assertEqual(tuple(result.confirmed), _SCOPES_ANTIGRAVITY)

        c0 = score_c0_fold_antigravity(
            self.train,
            self.fit,
            self.query,
            seed=20260721,
            fold_index=0,
            authority=self.authority,
        )
        np.testing.assert_array_equal(result.original["C0"], c0.normal_scores)
        for change_type in _CHANGE_TYPES_ANTIGRAVITY:
            np.testing.assert_array_equal(
                result.primary["C0"][change_type],
                c0.primary_scores[change_type],
            )

        generated = tuple(primary_rows_antigravity(row) for row in self.train[self.query])
        query_rows = {
            "normal": self.train[self.query],
            **{
                change_type: np.stack(
                    tuple(changes[change_type] for changes in generated),
                )
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            },
        }
        combined = np.concatenate(
            tuple(query_rows[scope] for scope in _SCOPES_ANTIGRAVITY),
            axis=0,
        )
        spectral = score_local_stft_antigravity(
            self.train[self.fit],
            combined,
            backend="numpy",
        )
        np.testing.assert_array_equal(result.fit_raw_js, spectral.fit_raw_js)
        self.assertEqual(result.fit_normal_max_raw_js, spectral.fit_normal_max_raw_js)
        width = len(self.query)
        for scope_index, scope in enumerate(_SCOPES_ANTIGRAVITY):
            start = scope_index * width
            stop = start + width
            np.testing.assert_array_equal(
                result.query_raw_js[scope],
                spectral.query_raw_js[start:stop],
            )
            np.testing.assert_array_equal(
                result.confirmed[scope],
                spectral.confirmation_mask[start:stop],
            )
            stored_s10 = (
                result.original["S10"]
                if scope == "normal"
                else result.primary["S10"][scope]
            )
            np.testing.assert_array_equal(stored_s10, spectral.query_ranks[start:stop])

    def test_c10_is_strict_confirmed_max_and_exact_c0_fallback_antigravity(self) -> None:
        """Break caught: C10 changes an unconfirmed value or uses a non-strict boundary."""

        result = self.result
        for scope in _SCOPES_ANTIGRAVITY:
            c0 = (
                result.original["C0"]
                if scope == "normal"
                else result.primary["C0"][scope]
            )
            s10 = (
                result.original["S10"]
                if scope == "normal"
                else result.primary["S10"][scope]
            )
            c10 = (
                result.original["C10"]
                if scope == "normal"
                else result.primary["C10"][scope]
            )
            expected_confirmation = (
                (result.query_raw_js[scope] > 0.0)
                & (result.query_raw_js[scope] > result.fit_normal_max_raw_js)
            )
            np.testing.assert_array_equal(result.confirmed[scope], expected_confirmation)
            np.testing.assert_array_equal(
                c10,
                np.where(expected_confirmation, np.maximum(c0, s10), c0),
            )
            np.testing.assert_array_equal(c10[~expected_confirmation], c0[~expected_confirmation])

        raw = {
            scope: np.array(values, dtype=np.float64, copy=True)
            for scope, values in result.query_raw_js.items()
        }
        confirmed = {
            scope: np.array(values, dtype=np.bool_, copy=True)
            for scope, values in result.confirmed.items()
        }
        raw["normal"][0] = result.fit_normal_max_raw_js
        confirmed["normal"][0] = True
        with self.assertRaisesRegex(ValueError, "strict raw-JS"):
            dataclasses.replace(
                result,
                query_raw_js=MappingProxyType(raw),
                confirmed=MappingProxyType(confirmed),
            )

    def test_fold_record_owns_freezes_and_types_every_array_antigravity(self) -> None:
        """Break caught: retained replay evidence aliases caller memory or changes dtype."""

        result = self.result
        self.assertIsInstance(result.original, MappingProxyType)
        self.assertIsInstance(result.primary, MappingProxyType)
        self.assertIsInstance(result.query_raw_js, MappingProxyType)
        self.assertIsInstance(result.confirmed, MappingProxyType)
        float_arrays = (
            result.input_train140,
            result.fit_raw_js,
            *result.query_raw_js.values(),
            *result.original.values(),
            *(values for method in result.primary.values() for values in method.values()),
        )
        for values in float_arrays:
            self.assertEqual(values.dtype, np.dtype(np.float64))
            self.assertTrue(values.flags.owndata)
            self.assertTrue(values.flags.c_contiguous)
            self.assertFalse(values.flags.writeable)
        for values in result.confirmed.values():
            self.assertEqual(values.dtype, np.dtype(np.bool_))
            self.assertTrue(values.flags.owndata)
            self.assertTrue(values.flags.c_contiguous)
            self.assertFalse(values.flags.writeable)
        for values in (result.input_fit_indices, result.input_query_indices):
            self.assertEqual(values.dtype, np.dtype(np.int64))
            self.assertTrue(values.flags.owndata)
            self.assertTrue(values.flags.c_contiguous)
            self.assertFalse(values.flags.writeable)
        self.assertFalse(np.shares_memory(result.input_train140, self.train))
        self.assertFalse(np.shares_memory(result.input_fit_indices, self.fit))
        self.assertFalse(np.shares_memory(result.input_query_indices, self.query))

    def test_invalid_indices_overlap_nonfinite_and_method_drift_are_rejected_antigravity(
        self,
    ) -> None:
        """Break caught: malformed raw inputs can enter an auditable fold record."""

        calls = (
            (
                self.fit.astype(np.int32),
                self.query,
                "int64",
            ),
            (
                np.array((*self.fit[:-1], self.fit[-2]), dtype=np.int64),
                self.query,
                "unique",
            ),
            (
                self.fit,
                np.array((54, 55), dtype=np.int64),
                "disjoint",
            ),
            (
                self.fit,
                np.array((69,), dtype=np.int64),
                "out-of-range",
            ),
        )
        for fit, query, message in calls:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    score_fold_methods_antigravity(
                        self.train,
                        fit,
                        query,
                        seed=20260721,
                        fold_index=0,
                        authority=self.authority,
                    )

        nonfinite = np.array(self.train, dtype=np.float64, copy=True)
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            score_fold_methods_antigravity(
                nonfinite,
                self.fit,
                self.query,
                seed=20260721,
                fold_index=0,
                authority=self.authority,
            )
        with self.assertRaisesRegex(ValueError, "C0/S10/C10"):
            dataclasses.replace(self.result, methods=("C0", "S10", "C9"))


if __name__ == "__main__":
    unittest.main()
