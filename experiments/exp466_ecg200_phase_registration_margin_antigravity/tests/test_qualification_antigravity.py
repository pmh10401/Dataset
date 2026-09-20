from __future__ import annotations

import dataclasses
from functools import lru_cache
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
)


_SEEDS_ANTIGRAVITY = (20260721, 20261721)
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


@lru_cache(maxsize=1)
def authoritative_train_antigravity() -> np.ndarray:
    """Return the one deterministic 69-row synthetic Task 4 train fixture."""
    x = np.arange(140, dtype=np.float64)
    base = (
        0.01 * x
        + np.sin(2.0 * np.pi * x / 21.0)
        + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    )
    train = base[None, :] + np.random.default_rng(4664).normal(
        scale=0.01,
        size=(69, 140),
    )
    train = np.asarray(train, dtype=np.float64, order="C")
    train.setflags(write=False)
    return train


def score_records_antigravity(train: np.ndarray):
    from experiments.exp466_ecg200_phase_registration_margin_antigravity.scoring_antigravity import (
        score_fold_methods_antigravity,
    )

    authority = capture_exp465_authority_antigravity()
    records = {}
    for seed in _SEEDS_ANTIGRAVITY:
        for fold_index, query in enumerate(
            np.array_split(np.arange(69, dtype=np.int64), 5),
        ):
            query_set = set(int(value) for value in query)
            fit = np.asarray(
                tuple(index for index in range(69) if index not in query_set),
                dtype=np.int64,
            )
            records[(seed, fold_index)] = score_fold_methods_antigravity(
                train,
                fit,
                query,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    return MappingProxyType(records)


@lru_cache(maxsize=1)
def authoritative_records_antigravity():
    """Ten source-free score records over one deterministic 69-row fixture."""
    return score_records_antigravity(authoritative_train_antigravity())


@lru_cache(maxsize=1)
def normal_confirmation_records_antigravity():
    """Real rescored fixture with one OOF normal confirmation per seed."""
    from experiments.exp466_ecg200_phase_registration_margin_antigravity.synthetic_antigravity import (
        primary_rows_antigravity,
    )

    train = np.array(authoritative_train_antigravity(), dtype=np.float64, copy=True)
    train[0] = primary_rows_antigravity(train[0])["period_change"]
    train = np.asarray(train, dtype=np.float64, order="C")
    train.setflags(write=False)
    return score_records_antigravity(train)


class QualificationAntigravityTest(unittest.TestCase):
    def test_exact_ten_records_replay_inputs_and_cover_69_unique_oof_antigravity(self) -> None:
        """Break caught: qualification trusts C0/P9/C9 fields or partial OOF coverage."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.contracts_antigravity import (
            Exp466QualificationAntigravity,
        )
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.qualification_antigravity import (
            qualify_antigravity,
        )

        records = authoritative_records_antigravity()
        result = qualify_antigravity(records)
        self.assertIs(type(result), Exp466QualificationAntigravity)
        self.assertEqual(len(result.fold_results), 10)
        self.assertEqual(
            tuple((row.seed, row.fold_index) for row in result.fold_results),
            tuple((seed, fold) for seed in _SEEDS_ANTIGRAVITY for fold in range(5)),
        )
        self.assertEqual(dict(result.normal_trial_counts), {20260721: 69, 20261721: 69})
        self.assertEqual(dict(result.normal_confirmations), {20260721: 0, 20261721: 0})
        self.assertEqual(result.label_query_count, 0)
        self.assertEqual(result.official_evaluate_count, 0)
        self.assertEqual(result.official_inspect_count, 0)
        self.assertIsNone(result.artifact_map_sha256)

        for seed in _SEEDS_ANTIGRAVITY:
            oof = np.concatenate(
                tuple(
                    records[(seed, fold)].input_query_indices
                    for fold in range(5)
                ),
            )
            np.testing.assert_array_equal(np.sort(oof), np.arange(69, dtype=np.int64))
            self.assertEqual(len(np.unique(oof)), 69)

        first_key = (_SEEDS_ANTIGRAVITY[0], 0)
        first = records[first_key]
        altered_margins = {
            scope: np.array(values, dtype=np.float64, copy=True)
            for scope, values in first.query_margins.items()
        }
        altered_margins["normal"][0] += 0.125
        forged = dataclasses.replace(
            first,
            query_margins=MappingProxyType(altered_margins),
        )
        forged_records = MappingProxyType(
            {
                key: forged if key == first_key else value
                for key, value in records.items()
            },
        )
        with self.assertRaisesRegex(ValueError, "P9 raw evidence"):
            qualify_antigravity(forged_records)

    def test_frozen_failure_reasons_cover_every_gate_boundary_antigravity(self) -> None:
        """Break caught: equality or just-below thresholds silently pass a C9 fold."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.qualification_antigravity import (
            _failure_reasons_antigravity,
        )

        passing = {
            "per_type_auc": (0.85, 0.90, 0.90, 0.95),
            "mean_auc": 0.90,
            "all_scores_finite": True,
            "c0_gradual_auc": 0.90,
            "c0_period_auc": 0.90,
            "c9_gradual_auc": 0.90,
            "c9_period_auc": 0.95,
            "period_confirmations": 1,
            "normal_confirmations": 0,
        }
        self.assertEqual(_failure_reasons_antigravity(**passing), ())
        cases = (
            (
                {"c9_period_auc": 0.90},
                ("period_auc_not_strictly_above_c0",),
            ),
            (
                {"c9_gradual_auc": 0.899999},
                ("gradual_auc_below_c0",),
            ),
            (
                {"period_confirmations": 0},
                ("period_confirmation_missing",),
            ),
            (
                {"normal_confirmations": 1},
                ("normal_confirmation_present",),
            ),
            (
                {"per_type_auc": (0.85, 0.90, 0.90, 0.849999)},
                ("period_change_auc_below_0.85",),
            ),
            (
                {"mean_auc": 0.899999},
                ("mean_auc_below_0.90",),
            ),
            (
                {"all_scores_finite": False},
                ("nonfinite_scores",),
            ),
        )
        for changes, expected in cases:
            values = dict(passing)
            values.update(changes)
            with self.subTest(changes=changes):
                self.assertEqual(_failure_reasons_antigravity(**values), expected)

    def test_real_normal_confirmation_is_a_failed_result_not_an_exception_antigravity(
        self,
    ) -> None:
        """Break caught: a measured normal confirmation cannot be published as a failure."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.qualification_antigravity import (
            qualify_antigravity,
        )

        qualification = qualify_antigravity(normal_confirmation_records_antigravity())
        self.assertEqual(
            dict(qualification.normal_confirmations),
            {20260721: 1, 20261721: 1},
        )
        normal_failed = tuple(
            (row.seed, row.fold_index)
            for row in qualification.fold_results
            if "normal_confirmation_present" in row.failure_reasons
        )
        self.assertEqual(normal_failed, ((20260721, 0), (20261721, 0)))
        self.assertTrue(
            all(
                not row.passed
                for row in qualification.fold_results
                if row.fold_index == 0
            ),
        )

        decision = evaluate_qualification_antigravity(qualification)
        self.assertFalse(decision.eligible_for_exp467_design)
        self.assertIn("c9_not_10_of_10", decision.failure_reasons)
        self.assertIn("normal_confirmation_present", decision.failure_reasons)
        self.assertIn(
            "seed_20260721_normal_confirmations_not_zero",
            decision.failure_reasons,
        )
        self.assertIn(
            "seed_20261721_normal_confirmations_not_zero",
            decision.failure_reasons,
        )

    def test_rejects_wrong_record_order_and_duplicate_oof_before_replay_antigravity(self) -> None:
        """Break caught: ten records with reordered keys or repeated normal trials are accepted."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.qualification_antigravity import (
            qualify_antigravity,
        )

        records = authoritative_records_antigravity()
        reversed_records = MappingProxyType(dict(reversed(tuple(records.items()))))
        with self.assertRaisesRegex(ValueError, "seed-major order"):
            qualify_antigravity(reversed_records)

        first_key = (_SEEDS_ANTIGRAVITY[0], 0)
        second_key = (_SEEDS_ANTIGRAVITY[0], 1)
        duplicate_query = records[first_key].input_query_indices
        duplicate_query_set = set(int(value) for value in duplicate_query)
        duplicate_fit = np.asarray(
            tuple(index for index in range(69) if index not in duplicate_query_set),
            dtype=np.int64,
        )
        duplicate = dataclasses.replace(
            records[second_key],
            input_fit_indices=duplicate_fit,
            input_query_indices=duplicate_query,
        )
        duplicate_records = MappingProxyType(
            {
                key: duplicate if key == second_key else value
                for key, value in records.items()
            },
        )
        with self.assertRaisesRegex(ValueError, "69 unique"):
            qualify_antigravity(duplicate_records)


if __name__ == "__main__":
    unittest.main()
