from __future__ import annotations

import inspect
import unittest
from types import MappingProxyType
from typing import Any

import numpy as np

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
    capture_exp463_authority_antigravity,
)
from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
    score_fold_methods_antigravity as score_exp464_fold_methods_antigravity,
)
from experiments.exp465_ecg200_c8_external_validation_antigravity.config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
)


_METHOD_FIELDS_ANTIGRAVITY = (
    "original",
    "primary",
    "confirmation_counts",
    "fit_amplitudes",
    "query_amplitudes",
    "query_cosines",
    "raw_evidence_kind",
    "raw_evidence_source_method",
    "raw_fit_vector_blobs",
    "raw_query_vector_blobs",
    "feature_dimensions",
    "provenance",
)


def _assert_equal_value_antigravity(testcase: unittest.TestCase, actual: object, expected: object) -> None:
    if type(actual) is not type(expected):
        testcase.fail(f"type differs: {type(actual)} != {type(expected)}")
    if isinstance(actual, MappingProxyType) and isinstance(expected, MappingProxyType):
        testcase.assertEqual(tuple(actual), tuple(expected))
        for key in expected:
            _assert_equal_value_antigravity(testcase, actual[key], expected[key])  # type: ignore[index]
        return
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        testcase.assertIs(actual.dtype.type, expected.dtype.type)
        return
    testcase.assertEqual(actual, expected)


class ScoringAntigravityTest(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        rng = np.random.default_rng(4652026)
        self.train = rng.standard_normal((20, FROZEN_CONFIG_ANTIGRAVITY.interpolated_width))
        self.train.setflags(write=False)
        self.fit_indices = np.arange(6, dtype=np.int64)
        self.query_indices = np.array([6, 7, 8, 10, 12], dtype=np.int64)
        self.authority = capture_exp463_authority_antigravity()

    def test_fixed_entrypoint_projects_exact_exp464_fields_and_flags_antigravity(self) -> None:
        """Break caught: scoring rewrites Exp464 math, changes method order/fields, or loses immutability."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import scoring_antigravity

        self.assertEqual(
            tuple(scoring_antigravity.__all__),
            ("FoldMethodScoresAntigravity", "project_methods_antigravity", "score_fold_methods_antigravity"),
        )
        signature = inspect.signature(scoring_antigravity.score_fold_methods_antigravity)
        self.assertEqual(
            tuple(signature.parameters),
            ("train140", "fit_indices", "query_indices", "seed", "fold_index", "authority"),
        )
        for parameter_name in ("seed", "fold_index", "authority"):
            self.assertTrue(signature.parameters[parameter_name].kind is inspect.Parameter.KEYWORD_ONLY)

        result = scoring_antigravity.score_fold_methods_antigravity(
            self.train,
            self.fit_indices,
            self.query_indices,
            seed=FROZEN_CONFIG_ANTIGRAVITY.seeds[0],
            fold_index=0,
            authority=self.authority,
        )
        source = score_exp464_fold_methods_antigravity(
            self.train,
            self.fit_indices,
            self.query_indices,
            seed=FROZEN_CONFIG_ANTIGRAVITY.seeds[0],
            fold_index=0,
            authority=self.authority,
        )
        self.assertEqual(result.methods, FROZEN_CONFIG_ANTIGRAVITY.method_order)
        self.assertEqual(result.methods, ("C0", "C6", "C8"))
        for field_name in _METHOD_FIELDS_ANTIGRAVITY:
            actual_mapping = getattr(result, field_name)
            expected_mapping = getattr(source, field_name)
            self.assertIsInstance(actual_mapping, MappingProxyType)
            self.assertEqual(tuple(actual_mapping), result.methods)
            for method in result.methods:
                _assert_equal_value_antigravity(self, actual_mapping[method], expected_mapping[method])
        for field_name in _METHOD_FIELDS_ANTIGRAVITY:
            with self.assertRaises(TypeError):
                getattr(result, field_name)["C0"] = getattr(source, field_name)["C0"]  # type: ignore[index]

    def test_golden_ten_seed_fold_calls_have_exact_exp464_parity_antigravity(self) -> None:
        """Break caught: any of the ten frozen seed-fold projections drifts from direct Exp464 output."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.scoring_antigravity import (
            project_methods_antigravity,
            score_fold_methods_antigravity,
        )

        call_count = 0
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds):
                with self.subTest(seed=seed, fold_index=fold_index):
                    source = score_exp464_fold_methods_antigravity(
                        self.train,
                        self.fit_indices,
                        self.query_indices,
                        seed=seed,
                        fold_index=fold_index,
                        authority=self.authority,
                    )
                    projected_directly = project_methods_antigravity(source, FROZEN_CONFIG_ANTIGRAVITY.method_order)
                    actual = score_fold_methods_antigravity(
                        self.train,
                        self.fit_indices,
                        self.query_indices,
                        seed=seed,
                        fold_index=fold_index,
                        authority=self.authority,
                    )
                    self.assertEqual(actual.methods, ("C0", "C6", "C8"))
                    for field_name in _METHOD_FIELDS_ANTIGRAVITY:
                        actual_mapping = getattr(actual, field_name)
                        expected_mapping = getattr(source, field_name)
                        for method in FROZEN_CONFIG_ANTIGRAVITY.method_order:
                            _assert_equal_value_antigravity(
                                self,
                                actual_mapping[method],
                                expected_mapping[method],
                            )
                            _assert_equal_value_antigravity(
                                self,
                                actual_mapping[method],
                                getattr(projected_directly, field_name)[method],
                            )
                    call_count += 1
        self.assertEqual(call_count, 10)

    def test_entrypoint_spy_proves_one_exp464_scorer_call_per_invocation_antigravity(self) -> None:
        """Break caught: the Exp465 entrypoint skips, repeats, or bypasses the imported Exp464 scorer."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import scoring_antigravity

        imported_scorer = scoring_antigravity.score_exp464_fold_methods_antigravity
        spy_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        spy_sources: list[Any] = []

        def score_exp464_scorer_spy_antigravity(*args: Any, **kwargs: Any):
            spy_calls.append((args, kwargs))
            source = imported_scorer(*args, **kwargs)
            spy_sources.append(source)
            return source

        scoring_antigravity.score_exp464_fold_methods_antigravity = score_exp464_scorer_spy_antigravity
        try:
            expected_call_count = 0
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
                for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds):
                    with self.subTest(seed=seed, fold_index=fold_index):
                        actual = scoring_antigravity.score_fold_methods_antigravity(
                            self.train,
                            self.fit_indices,
                            self.query_indices,
                            seed=seed,
                            fold_index=fold_index,
                            authority=self.authority,
                        )
                        expected_call_count += 1
                        self.assertEqual(len(spy_calls), expected_call_count)
                        positional_arguments, keyword_arguments = spy_calls[-1]
                        self.assertEqual(positional_arguments, (self.train, self.fit_indices, self.query_indices))
                        self.assertEqual(keyword_arguments, {
                            "seed": seed,
                            "fold_index": fold_index,
                            "authority": self.authority,
                        })
                        source = spy_sources[-1]
                        for field_name in _METHOD_FIELDS_ANTIGRAVITY:
                            actual_mapping = getattr(actual, field_name)
                            source_mapping = getattr(source, field_name)
                            self.assertEqual(tuple(actual_mapping), FROZEN_CONFIG_ANTIGRAVITY.method_order)
                            for method in FROZEN_CONFIG_ANTIGRAVITY.method_order:
                                _assert_equal_value_antigravity(
                                    self,
                                    actual_mapping[method],
                                    source_mapping[method],
                                )
        finally:
            scoring_antigravity.score_exp464_fold_methods_antigravity = imported_scorer
        self.assertEqual(expected_call_count, 10)
        self.assertIs(scoring_antigravity.score_exp464_fold_methods_antigravity, imported_scorer)
