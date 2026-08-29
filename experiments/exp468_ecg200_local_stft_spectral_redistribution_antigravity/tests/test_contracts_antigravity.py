from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
from types import MappingProxyType
import unittest


class Exp468ContractsTestAntigravity(unittest.TestCase):
    def fold_result_antigravity(self, *, seed=20260721, fold_index=0, **changes):
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.contracts_antigravity import (
            Exp468FoldResultAntigravity,
        )

        values = {
            "seed": seed,
            "fold_index": fold_index,
            "c0_per_type_auc": (0.70, 0.71, 0.72, 0.73),
            "c10_per_type_auc": (0.80, 0.81, 0.82, 0.83),
            "period_confirmations": 1,
            "normal_confirmations": 0,
            "all_scores_finite": True,
            "passed": True,
            "failure_reasons": (),
        }
        values.update(changes)
        return Exp468FoldResultAntigravity(**values)

    def valid_qualification_antigravity(self):
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.contracts_antigravity import (
            Exp468QualificationAntigravity,
        )

        rows = tuple(
            self.fold_result_antigravity(seed=seed, fold_index=fold_index)
            for seed in (20260721, 20261721)
            for fold_index in range(5)
        )
        return Exp468QualificationAntigravity(
            fold_results=rows,
            normal_trial_counts=MappingProxyType({20260721: 69, 20261721: 69}),
            normal_confirmations=MappingProxyType({20260721: 0, 20261721: 0}),
            label_query_count=0,
            artifact_map_sha256=None,
        )

    def test_fold_auc_tuples_are_exact_finite_and_positionally_bound_antigravity(self) -> None:
        """Break caught: an AUC vector loses its exact shape, float type, or change order."""
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        row = self.fold_result_antigravity()
        self.assertEqual(len(row.c0_per_type_auc), len(FROZEN_CONFIG_ANTIGRAVITY.change_types))
        self.assertEqual(len(row.c10_per_type_auc), len(FROZEN_CONFIG_ANTIGRAVITY.change_types))
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY.change_types[-1], "period_change")
        for field_name in ("c0_per_type_auc", "c10_per_type_auc"):
            invalid_values = (
                [0.80, 0.81, 0.82, 0.83],
                (0.80, 0.81, 0.82),
                (0.80, 0.81, float("nan"), 0.83),
                (0.80, 0.81, float("inf"), 0.83),
                (0.80, 0.81, -0.01, 0.83),
                (0.80, 0.81, 1.01, 0.83),
                (0.80, 0.81, 1, 0.83),
            )
            for invalid in invalid_values:
                with self.subTest(field_name=field_name, invalid=invalid), self.assertRaises(
                    ValueError,
                ):
                    dataclasses.replace(row, **{field_name: invalid})

    def test_fold_seed_index_counts_and_boolean_types_are_bounded_antigravity(self) -> None:
        """Break caught: a row enters an unknown seed/fold or uses bool as an integer count."""
        row = self.fold_result_antigravity()
        for changes in (
            {"seed": True},
            {"seed": 20260722},
            {"fold_index": True},
            {"fold_index": -1},
            {"fold_index": 5},
            {"period_confirmations": -1},
            {"period_confirmations": True},
            {"normal_confirmations": -1},
            {"normal_confirmations": True},
            {"all_scores_finite": 1},
            {"passed": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(row, **changes)

    def test_passed_and_unique_immutable_failure_reasons_agree_antigravity(self) -> None:
        """Break caught: pass state disagrees with empty, immutable, unique failure reasons."""
        row = self.fold_result_antigravity()
        failed = dataclasses.replace(
            row,
            passed=False,
            failure_reasons=("period_auc_below_0.85",),
        )
        self.assertEqual(failed.failure_reasons, ("period_auc_below_0.85",))
        for changes in (
            {"passed": False, "failure_reasons": ()},
            {"passed": True, "failure_reasons": ("reason",)},
            {"passed": False, "failure_reasons": ["reason"]},
            {"passed": False, "failure_reasons": ("",)},
            {"passed": False, "failure_reasons": ("reason", "reason")},
            {"passed": False, "failure_reasons": (1,)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(row, **changes)

    def test_task1_contract_does_not_reimplement_scientific_gate_antigravity(self) -> None:
        """Break caught: Task 1 prematurely calculates Task 4's scientific decision."""
        row = self.fold_result_antigravity(
            c10_per_type_auc=(0.10, 0.20, 0.30, 0.40),
            period_confirmations=0,
            normal_confirmations=7,
            all_scores_finite=False,
        )
        self.assertTrue(row.passed)
        self.assertEqual(row.failure_reasons, ())

    def test_qualification_requires_exact_seed_major_ten_rows_antigravity(self) -> None:
        """Break caught: qualification accepts missing, mutable, or reordered fold rows."""
        valid = self.valid_qualification_antigravity()
        expected_keys = tuple(
            (seed, fold_index)
            for seed in (20260721, 20261721)
            for fold_index in range(5)
        )
        self.assertEqual(
            tuple((row.seed, row.fold_index) for row in valid.fold_results),
            expected_keys,
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            dataclasses.replace(valid, fold_results=list(valid.fold_results))
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            dataclasses.replace(valid, fold_results=valid.fold_results[:-1])
        reordered = (valid.fold_results[1], valid.fold_results[0], *valid.fold_results[2:])
        with self.assertRaisesRegex(ValueError, "seed/fold order"):
            dataclasses.replace(valid, fold_results=reordered)

    def test_qualification_copies_ordered_immutable_seed_counts_antigravity(self) -> None:
        """Break caught: mutable, reordered, or non-69 seed counts enter qualification."""
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.contracts_antigravity import (
            Exp468QualificationAntigravity,
        )

        rows = self.valid_qualification_antigravity().fold_results
        trial_source = {20260721: 69, 20261721: 69}
        confirmation_source = {20260721: 0, 20261721: 1}
        valid = Exp468QualificationAntigravity(
            fold_results=rows,
            normal_trial_counts=MappingProxyType(trial_source),
            normal_confirmations=MappingProxyType(confirmation_source),
            label_query_count=0,
            artifact_map_sha256=None,
        )
        trial_source[20260721] = 0
        confirmation_source[20261721] = 69
        self.assertIsInstance(valid.normal_trial_counts, MappingProxyType)
        self.assertIsInstance(valid.normal_confirmations, MappingProxyType)
        self.assertEqual(dict(valid.normal_trial_counts), {20260721: 69, 20261721: 69})
        self.assertEqual(dict(valid.normal_confirmations), {20260721: 0, 20261721: 1})
        with self.assertRaises(TypeError):
            valid.normal_confirmations[20260721] = 1  # type: ignore[index]
        for field_name in ("normal_trial_counts", "normal_confirmations"):
            with self.subTest(field_name=field_name), self.assertRaisesRegex(
                ValueError,
                "immutable",
            ):
                dataclasses.replace(
                    valid,
                    **{field_name: {20260721: 69, 20261721: 69}},
                )
        with self.assertRaisesRegex(ValueError, "seed order"):
            dataclasses.replace(
                valid,
                normal_confirmations=MappingProxyType({20261721: 0, 20260721: 0}),
            )
        for count in (-1, 70, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                dataclasses.replace(
                    valid,
                    normal_confirmations=MappingProxyType({20260721: count, 20261721: 0}),
                )
        with self.assertRaisesRegex(ValueError, "69"):
            dataclasses.replace(
                valid,
                normal_trial_counts=MappingProxyType({20260721: 68, 20261721: 69}),
            )

    def test_qualification_keeps_pre_run_zero_and_null_truth_antigravity(self) -> None:
        """Break caught: Task 1 claims a label query or result artifact before execution."""
        valid = self.valid_qualification_antigravity()
        with self.assertRaises(FrozenInstanceError):
            valid.label_query_count = 1  # type: ignore[misc]
        for changes in (
            {"label_query_count": True},
            {"label_query_count": 1},
            {"artifact_map_sha256": "b" * 64},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(valid, **changes)


if __name__ == "__main__":
    unittest.main()
