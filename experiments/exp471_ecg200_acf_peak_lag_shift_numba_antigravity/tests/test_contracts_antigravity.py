from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
from types import MappingProxyType
import unittest


_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_SEEDS_ANTIGRAVITY = (20260721, 20261721)


class Exp471ContractsTestAntigravity(unittest.TestCase):
    def pair_repairs_antigravity(self, **changes):
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.contracts_antigravity import (
            PairRepairCountsAntigravity,
        )

        values = {"fixed": 1, "broken": 0}
        values.update(changes)
        return PairRepairCountsAntigravity(**values)

    def fold_result_antigravity(self, *, seed=20260721, fold_index=0, **changes):
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.contracts_antigravity import (
            Exp471FoldResultAntigravity,
        )

        values = {
            "seed": seed,
            "fold_index": fold_index,
            "c0_per_type_auc": dict.fromkeys(_CHANGE_TYPES_ANTIGRAVITY, 0.70),
            "c12_per_type_auc": dict.fromkeys(_CHANGE_TYPES_ANTIGRAVITY, 0.80),
            "pair_repairs": dict.fromkeys(
                _CHANGE_TYPES_ANTIGRAVITY,
                self.pair_repairs_antigravity(),
            ),
            "normal_confirmations": 0,
            "fallback_exact": True,
            "finite": True,
            "passed": True,
            "failure_reasons": (),
        }
        values.update(changes)
        return Exp471FoldResultAntigravity(**values)

    def valid_qualification_antigravity(self):
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.contracts_antigravity import (
            Exp471QualificationAntigravity,
        )

        rows = tuple(
            self.fold_result_antigravity(seed=seed, fold_index=fold_index)
            for seed in _SEEDS_ANTIGRAVITY
            for fold_index in range(5)
        )
        return Exp471QualificationAntigravity(
            fold_results=rows,
            period_fixed={20260721: 1, 20261721: 0},
            period_broken={20260721: 0, 20261721: 1},
            protected_broken={
                "level_shift": 0,
                "gradual_drift": 0,
                "amplitude_scaling": 0,
            },
            normal_confirmations={20260721: 0, 20261721: 0},
            fallback_exact=True,
            finite=True,
            label_query_count=0,
            artifact_map_sha256=None,
        )

    def test_fold_result_freezes_expected_per_type_mappings_antigravity(self) -> None:
        row = self.fold_result_antigravity()
        self.assertIsInstance(row.c0_per_type_auc, MappingProxyType)
        self.assertIsInstance(row.c12_per_type_auc, MappingProxyType)
        self.assertIsInstance(row.pair_repairs, MappingProxyType)
        self.assertEqual(tuple(row.c0_per_type_auc), _CHANGE_TYPES_ANTIGRAVITY)
        self.assertEqual(tuple(row.c12_per_type_auc), _CHANGE_TYPES_ANTIGRAVITY)
        self.assertEqual(tuple(row.pair_repairs), _CHANGE_TYPES_ANTIGRAVITY)
        with self.assertRaises(TypeError):
            row.pair_repairs["period_change"] = self.pair_repairs_antigravity()  # type: ignore[index]
        invalid_mappings = (
            ("c0_per_type_auc", {"level_shift": 0.70}),
            ("c12_per_type_auc", dict(reversed(tuple(row.c12_per_type_auc.items())))),
            ("c12_per_type_auc", {name: 1.01 for name in _CHANGE_TYPES_ANTIGRAVITY}),
            ("c12_per_type_auc", {name: float("nan") for name in _CHANGE_TYPES_ANTIGRAVITY}),
            ("pair_repairs", {name: self.pair_repairs_antigravity() for name in _CHANGE_TYPES_ANTIGRAVITY[:-1]}),
            ("pair_repairs", dict(reversed(tuple(row.pair_repairs.items())))),
        )
        for field_name, invalid in invalid_mappings:
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                dataclasses.replace(row, **{field_name: invalid})

    def test_pair_counts_and_fold_scalars_reject_aliases_antigravity(self) -> None:
        pair_repairs = self.pair_repairs_antigravity()
        with self.assertRaises(FrozenInstanceError):
            pair_repairs.fixed = 2  # type: ignore[misc]
        count_cases = ({"fixed": -1}, {"fixed": True}, {"broken": -1}, {"broken": True})
        for changes in count_cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.pair_repairs_antigravity(**changes)
        row = self.fold_result_antigravity()
        scalar_cases = (
            {"seed": True},
            {"seed": 1},
            {"fold_index": True},
            {"fold_index": 5},
            {"normal_confirmations": -1},
            {"normal_confirmations": True},
            {"fallback_exact": 1},
            {"finite": 1},
            {"passed": 1},
            {"failure_reasons": ["failure"]},
            {"failure_reasons": (1,)},
        )
        for changes in scalar_cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(row, **changes)

    def test_fold_result_preserves_any_tuple_of_failure_reasons_antigravity(self) -> None:
        reason_cases = (
            (True, ("",)),
            (False, ()),
            (False, ("", "period")),
            (True, ("period",)),
        )
        for passed, reasons in reason_cases:
            with self.subTest(passed=passed, reasons=reasons):
                row = self.fold_result_antigravity(passed=passed, failure_reasons=reasons)
                self.assertIs(row.failure_reasons, reasons)
                self.assertEqual((row.passed, row.failure_reasons), (passed, reasons))

    def test_qualification_requires_seed_major_rows_and_frozen_maps_antigravity(self) -> None:
        valid = self.valid_qualification_antigravity()
        expected_rows = tuple((seed, index) for seed in _SEEDS_ANTIGRAVITY for index in range(5))
        self.assertEqual(
            tuple((row.seed, row.fold_index) for row in valid.fold_results),
            expected_rows,
        )
        for field_name in ("period_fixed", "period_broken", "protected_broken", "normal_confirmations"):
            self.assertIsInstance(getattr(valid, field_name), MappingProxyType)
        self.assertEqual(tuple(valid.period_fixed), _SEEDS_ANTIGRAVITY)
        self.assertEqual(tuple(valid.period_broken), _SEEDS_ANTIGRAVITY)
        self.assertEqual(tuple(valid.normal_confirmations), _SEEDS_ANTIGRAVITY)
        self.assertEqual(tuple(valid.protected_broken), _CHANGE_TYPES_ANTIGRAVITY[:3])
        reordered_rows = (valid.fold_results[1], valid.fold_results[0], *valid.fold_results[2:])
        invalid_row_cases = (
            list(valid.fold_results),
            valid.fold_results[:-1],
            reordered_rows,
        )
        for fold_results in invalid_row_cases:
            with self.subTest(length=len(fold_results)), self.assertRaises(ValueError):
                dataclasses.replace(valid, fold_results=fold_results)
        map_cases = (
            ("period_fixed", {20261721: 0, 20260721: 0}),
            ("period_broken", {20260721: -1, 20261721: 0}),
            ("protected_broken", {"level_shift": 0, "gradual_drift": 0, "period_change": 0}),
            ("normal_confirmations", {20260721: True, 20261721: 0}),
        )
        for field_name, invalid in map_cases:
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                dataclasses.replace(valid, **{field_name: invalid})

    def test_qualification_preserves_governance_and_artifact_digest_boundary_antigravity(self) -> None:
        valid = self.valid_qualification_antigravity()
        with self.assertRaises(FrozenInstanceError):
            valid.label_query_count = 1  # type: ignore[misc]
        governance_cases = (
            {"fallback_exact": 1},
            {"finite": 1},
            {"label_query_count": True},
            {"label_query_count": 1},
            {"artifact_map_sha256": "A" * 64},
            {"artifact_map_sha256": "a" * 63},
        )
        for changes in governance_cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(valid, **changes)
        with_digest = dataclasses.replace(valid, artifact_map_sha256="a" * 64)
        self.assertEqual(with_digest.artifact_map_sha256, "a" * 64)
