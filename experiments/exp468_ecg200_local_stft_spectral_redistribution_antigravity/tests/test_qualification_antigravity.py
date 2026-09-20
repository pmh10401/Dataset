"""Task 4 contracts for exact blocked-fold replay and qualification gates."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
    mann_whitney_auc_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.contracts_antigravity import (
    Exp468QualificationAntigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.qualification_antigravity import (
    _failure_reasons_antigravity,
    blocked_fold_indices_antigravity,
    qualify_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.scoring_antigravity import (
    FoldMethodScoresAntigravity,
    score_fold_methods_antigravity,
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
    """Return a deterministic 69x140 TRAIN-only stand-in with no TEST dependency."""

    x = np.arange(140, dtype=np.float64)
    base = (
        0.01 * x
        + np.sin(2.0 * np.pi * x / 21.0)
        + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    )
    rows = base[None, :] + np.random.default_rng(46840).normal(
        scale=0.01,
        size=(69, 140),
    )
    result = np.asarray(rows, dtype=np.float64, order="C")
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def authoritative_records_antigravity() -> MappingProxyType:
    """Score the exact two-seed, five-block fixture once for replay tests."""

    train = authoritative_train_antigravity()
    authority = capture_exp465_authority_antigravity()
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    folds = blocked_fold_indices_antigravity()
    for seed in _SEEDS_ANTIGRAVITY:
        for fold_index, (fit, query) in enumerate(folds):
            records[(seed, fold_index)] = score_fold_methods_antigravity(
                train,
                fit,
                query,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    return MappingProxyType(records)


def _replace_record_antigravity(
    records: MappingProxyType,
    key: tuple[int, int],
    replacement: FoldMethodScoresAntigravity,
) -> MappingProxyType:
    return MappingProxyType(
        {
            identity: replacement if identity == key else record
            for identity, record in records.items()
        },
    )


def _mutate_owned_antigravity(values: np.ndarray, index: int = 0) -> None:
    values.setflags(write=True)
    if values.dtype == np.dtype(np.bool_):
        values[index] = not bool(values[index])
    else:
        target = np.inf if float(values[index]) < 1.0 else -np.inf
        values[index] = np.nextafter(values[index], target)
    values.setflags(write=False)


class QualificationAntigravityTest(unittest.TestCase):
    def test_public_blocked_folds_are_exact_owned_and_cover_69_once_antigravity(self) -> None:
        """Break caught: contiguous fold identities are shuffled while coverage still looks valid."""

        folds = blocked_fold_indices_antigravity()
        self.assertEqual(len(folds), 5)
        self.assertEqual(tuple(len(query) for _fit, query in folds), (14, 14, 14, 14, 13))
        self.assertEqual(tuple(len(fit) for fit, _query in folds), (55, 55, 55, 55, 56))
        expected_queries = np.array_split(np.arange(69, dtype=np.int64), 5)
        for fold_index, ((fit, query), expected_query) in enumerate(
            zip(folds, expected_queries, strict=True),
        ):
            with self.subTest(fold_index=fold_index):
                expected_fit = np.setdiff1d(
                    np.arange(69, dtype=np.int64),
                    expected_query,
                    assume_unique=True,
                )
                np.testing.assert_array_equal(query, expected_query)
                np.testing.assert_array_equal(fit, expected_fit)
                for values in (fit, query):
                    self.assertEqual(values.dtype, np.dtype(np.int64))
                    self.assertTrue(values.flags.owndata)
                    self.assertTrue(values.flags.c_contiguous)
                    self.assertFalse(values.flags.writeable)
        np.testing.assert_array_equal(
            np.concatenate(tuple(query for _fit, query in folds)),
            np.arange(69, dtype=np.int64),
        )

    def test_exact_ten_records_replay_and_compute_per_fold_auc_antigravity(self) -> None:
        """Break caught: qualification trusts stored scores or pools AUC across folds."""

        records = authoritative_records_antigravity()
        result = qualify_antigravity(records)
        self.assertIs(type(result), Exp468QualificationAntigravity)
        self.assertEqual(len(result.fold_results), 10)
        self.assertEqual(
            tuple((row.seed, row.fold_index) for row in result.fold_results),
            tuple((seed, fold) for seed in _SEEDS_ANTIGRAVITY for fold in range(5)),
        )
        self.assertEqual(dict(result.normal_trial_counts), {20260721: 69, 20261721: 69})
        self.assertEqual(result.label_query_count, 0)
        self.assertIsNone(result.artifact_map_sha256)

        for row in result.fold_results:
            record = records[(row.seed, row.fold_index)]
            expected_c0 = tuple(
                mann_whitney_auc_antigravity(
                    record.original["C0"],
                    record.primary["C0"][change_type],
                )
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            )
            expected_c10 = tuple(
                mann_whitney_auc_antigravity(
                    record.original["C10"],
                    record.primary["C10"][change_type],
                )
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            )
            self.assertEqual(row.c0_per_type_auc, expected_c0)
            self.assertEqual(row.c10_per_type_auc, expected_c10)
            self.assertEqual(
                row.period_confirmations,
                int(np.count_nonzero(record.confirmed["period_change"])),
            )
            self.assertEqual(
                row.normal_confirmations,
                int(np.count_nonzero(record.confirmed["normal"])),
            )

    def test_raw_replay_rejects_c0_fit_query_rank_confirmation_and_c10_tampering_antigravity(
        self,
    ) -> None:
        """Break caught: a self-consistent-looking stored field bypasses raw-input replay."""

        records = authoritative_records_antigravity()
        key = (_SEEDS_ANTIGRAVITY[0], 0)
        cases = (
            ("C0", lambda row: row.original["C0"], "C0 input replay"),
            ("fit_raw", lambda row: row.fit_raw_js, "S10 raw evidence"),
            ("query_raw", lambda row: row.query_raw_js["normal"], "S10 raw evidence"),
            ("rank", lambda row: row.original["S10"], "S10 raw evidence"),
            ("confirmation", lambda row: row.confirmed["normal"], "S10 raw evidence"),
            ("C10", lambda row: row.original["C10"], "C10 score"),
        )
        for name, select, message in cases:
            with self.subTest(name=name):
                forged = dataclasses.replace(records[key])
                _mutate_owned_antigravity(select(forged))
                with self.assertRaisesRegex(ValueError, message):
                    qualify_antigravity(_replace_record_antigravity(records, key, forged))

    def test_rejects_order_fold_exchange_and_train_byte_drift_before_replay_antigravity(
        self,
    ) -> None:
        """Break caught: OOF-complete but wrong fold identities or mixed raw inputs pass."""

        records = authoritative_records_antigravity()
        reversed_records = MappingProxyType(dict(reversed(tuple(records.items()))))
        with self.assertRaisesRegex(ValueError, "seed-major order"):
            qualify_antigravity(reversed_records)

        first_key = (_SEEDS_ANTIGRAVITY[0], 0)
        second_key = (_SEEDS_ANTIGRAVITY[0], 1)
        first = records[first_key]
        second = records[second_key]
        swapped_first = dataclasses.replace(
            first,
            input_fit_indices=second.input_fit_indices,
            input_query_indices=second.input_query_indices,
        )
        swapped_second = dataclasses.replace(
            second,
            input_fit_indices=first.input_fit_indices,
            input_query_indices=first.input_query_indices,
        )
        exchanged = MappingProxyType(
            {
                key: (
                    swapped_first
                    if key == first_key
                    else swapped_second
                    if key == second_key
                    else value
                )
                for key, value in records.items()
            },
        )
        with self.assertRaisesRegex(ValueError, "exact blocked fold"):
            qualify_antigravity(exchanged)

        changed_train = np.array(first.input_train140, dtype=np.float64, copy=True)
        changed_train[0, 0] = np.nextafter(changed_train[0, 0], np.inf)
        mixed = dataclasses.replace(first, input_train140=changed_train)
        with self.assertRaisesRegex(ValueError, "same exact 69x140 TRAIN bytes"):
            qualify_antigravity(_replace_record_antigravity(records, first_key, mixed))

    def test_every_frozen_gate_boundary_and_reason_order_without_mean_gate_antigravity(
        self,
    ) -> None:
        """Break caught: a boundary becomes inclusive or a mean-AUC gate is introduced."""

        passing = {
            "c0_per_type_auc": (0.70, 0.70, 0.70, 0.84),
            "c10_per_type_auc": (0.70, 0.70, 0.70, 0.85),
            "period_confirmations": 1,
            "normal_confirmations": 0,
            "all_scores_finite": True,
        }
        self.assertEqual(_failure_reasons_antigravity(**passing), ())
        cases = (
            (
                {
                    "c0_per_type_auc": (0.70, 0.70, 0.70, 0.80),
                    "c10_per_type_auc": (0.70, 0.70, 0.70, 0.849999),
                },
                ("period_auc_below_0.85",),
            ),
            (
                {
                    "c0_per_type_auc": (0.70, 0.70, 0.70, 0.90),
                    "c10_per_type_auc": (0.70, 0.70, 0.70, 0.90),
                },
                ("period_auc_not_strictly_above_c0",),
            ),
            ({"period_confirmations": 0}, ("period_confirmation_missing",)),
            (
                {
                    "c0_per_type_auc": (0.90, 0.70, 0.70, 0.84),
                    "c10_per_type_auc": (0.899999, 0.70, 0.70, 0.85),
                },
                ("level_shift_auc_below_c0",),
            ),
            (
                {
                    "c0_per_type_auc": (0.70, 0.90, 0.70, 0.84),
                    "c10_per_type_auc": (0.70, 0.899999, 0.70, 0.85),
                },
                ("gradual_drift_auc_below_c0",),
            ),
            (
                {
                    "c0_per_type_auc": (0.70, 0.70, 0.90, 0.84),
                    "c10_per_type_auc": (0.70, 0.70, 0.899999, 0.85),
                },
                ("amplitude_scaling_auc_below_c0",),
            ),
            ({"normal_confirmations": 1}, ("normal_confirmation_present",)),
            ({"all_scores_finite": False}, ("nonfinite_scores",)),
        )
        for changes, expected in cases:
            values = dict(passing)
            values.update(changes)
            with self.subTest(changes=changes):
                self.assertEqual(_failure_reasons_antigravity(**values), expected)

        low_mean = dict(passing)
        low_mean.update(
            c0_per_type_auc=(0.0, 0.0, 0.0, 0.84),
            c10_per_type_auc=(0.0, 0.0, 0.0, 0.85),
        )
        self.assertLess(sum(low_mean["c10_per_type_auc"]) / 4.0, 0.90)
        self.assertEqual(_failure_reasons_antigravity(**low_mean), ())

        composite = _failure_reasons_antigravity(
            c0_per_type_auc=(0.9, 0.9, 0.9, 0.9),
            c10_per_type_auc=(0.8, 0.8, 0.8, 0.8),
            period_confirmations=0,
            normal_confirmations=1,
            all_scores_finite=False,
        )
        self.assertEqual(
            composite,
            (
                "nonfinite_scores",
                "period_auc_below_0.85",
                "period_auc_not_strictly_above_c0",
                "period_confirmation_missing",
                "level_shift_auc_below_c0",
                "gradual_drift_auc_below_c0",
                "amplitude_scaling_auc_below_c0",
                "normal_confirmation_present",
            ),
        )


if __name__ == "__main__":
    unittest.main()
