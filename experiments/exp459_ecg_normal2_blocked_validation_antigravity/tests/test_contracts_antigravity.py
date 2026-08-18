from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType
import unittest

import numpy as np


PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
DATASET_ANTIGRAVITY = "ECG5000_normal_2"
SEEDS_ANTIGRAVITY = (20260721, 20261721)


def blocked_fold_antigravity(fold_index: int = 0) -> object:
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.blocked_folds_antigravity import (
        blocked_plan_antigravity,
    )

    return blocked_plan_antigravity(1413)[fold_index]


def fold_evidence_antigravity(
    *,
    seed: int = SEEDS_ANTIGRAVITY[0],
    fold_index: int = 0,
    per_type: tuple[float, float, float, float] = (0.95, 0.94, 0.96, 0.93),
    alerts: int = 0,
    passed: bool | None = None,
) -> object:
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
        FoldEvidenceAntigravity,
        wilson_upper_antigravity,
    )

    fold = blocked_fold_antigravity(fold_index)
    pooled = float(sum(per_type) / len(per_type))
    wilson = wilson_upper_antigravity(alerts, len(fold.query_indices))
    passed_value = bool(
        wilson <= 0.05 and pooled >= 0.90 and all(value >= 0.85 for value in per_type)
    ) if passed is None else passed
    return FoldEvidenceAntigravity(
        dataset=DATASET_ANTIGRAVITY,
        seed=seed,
        fold_index=fold_index,
        fit_indices=fold.fit_indices,
        query_indices=fold.query_indices,
        normal_trials=len(fold.query_indices),
        normal_alerts=alerts,
        normal_wilson_upper=wilson,
        per_type_auc={
            name: value for name, value in zip(PRIMARY_TYPES_ANTIGRAVITY, per_type, strict=True)
        },
        pooled_auc=pooled,
        passed=passed_value,
    )


def seed_summary_antigravity(
    *,
    seed: int = SEEDS_ANTIGRAVITY[0],
    failing_fold_index: int | None = None,
) -> object:
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
        SeedSummaryAntigravity,
    )

    records = []
    for fold_index in range(5):
        if failing_fold_index == fold_index:
            records.append(
                fold_evidence_antigravity(
                    seed=seed,
                    fold_index=fold_index,
                    per_type=(0.84, 0.95, 0.95, 0.95),
                    passed=False,
                )
            )
        else:
            records.append(fold_evidence_antigravity(seed=seed, fold_index=fold_index))
    return SeedSummaryAntigravity(
        dataset=DATASET_ANTIGRAVITY,
        seed=seed,
        fold_records=tuple(records),
        passed=failing_fold_index is None,
    )


class ContractsAntigravityTest(unittest.TestCase):
    def test_contract_fields_are_exact_and_instances_are_immutable_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            BlockedFoldAntigravity,
            BlockedQualificationAntigravity,
            FoldEvidenceAntigravity,
            SeedSummaryAntigravity,
            ValidationDecisionAntigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(BlockedFoldAntigravity)),
            ("fold_index", "fit_indices", "query_indices"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(FoldEvidenceAntigravity)),
            (
                "dataset",
                "seed",
                "fold_index",
                "fit_indices",
                "query_indices",
                "normal_trials",
                "normal_alerts",
                "normal_wilson_upper",
                "per_type_auc",
                "pooled_auc",
                "passed",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(SeedSummaryAntigravity)),
            ("dataset", "seed", "fold_records", "passed"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(BlockedQualificationAntigravity)),
            ("dataset", "fold_evidence", "seed_summaries", "passed"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(ValidationDecisionAntigravity)),
            ("dataset", "qualification", "status", "passed"),
        )

        evidence = fold_evidence_antigravity()
        self.assertIsInstance(evidence.per_type_auc, MappingProxyType)
        self.assertEqual(tuple(evidence.per_type_auc), PRIMARY_TYPES_ANTIGRAVITY)
        with self.assertRaises(TypeError):
            evidence.per_type_auc["level_shift"] = 0.0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            evidence.passed = False  # type: ignore[misc]

    def test_blocked_fold_copies_integer_arrays_without_aliasing_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            BlockedFoldAntigravity,
        )

        fit = np.array([0, 1, 2, 5], dtype=np.int64)
        query = np.array([3, 4], dtype=np.int64)
        fold = BlockedFoldAntigravity(1, fit, query)
        fit[0] = 99
        query[0] = 77
        self.assertEqual(fold.fit_indices.tolist(), [0, 1, 2, 5])
        self.assertEqual(fold.query_indices.tolist(), [3, 4])
        self.assertFalse(np.shares_memory(fold.fit_indices, fit))
        self.assertFalse(np.shares_memory(fold.query_indices, query))
        self.assertFalse(fold.fit_indices.flags.writeable)
        self.assertFalse(fold.query_indices.flags.writeable)

    def test_blocked_fold_rejects_type_order_uniqueness_and_overlap_attacks_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            BlockedFoldAntigravity,
        )

        cases = (
            dict(fold_index=True, fit_indices=np.array([0, 1]), query_indices=np.array([2, 3])),
            dict(fold_index=-1, fit_indices=np.array([0, 1]), query_indices=np.array([2, 3])),
            dict(fold_index=0, fit_indices=np.array([[0, 1]]), query_indices=np.array([2, 3])),
            dict(fold_index=0, fit_indices=np.array([1, 0]), query_indices=np.array([2, 3])),
            dict(fold_index=0, fit_indices=np.array([0, 0]), query_indices=np.array([2, 3])),
            dict(fold_index=0, fit_indices=np.array([0, 1], dtype=bool), query_indices=np.array([2, 3])),
            dict(fold_index=0, fit_indices=np.array([0, 1]), query_indices=np.array([1, 2])),
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                BlockedFoldAntigravity(**case)

    def test_fold_evidence_requires_exact_dataset_seed_fold_counts_and_metrics_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            FoldEvidenceAntigravity,
        )

        class ExactStrSubclassAntigravity(str):
            pass

        fold = blocked_fold_antigravity(0)
        valid = dict(
            dataset=DATASET_ANTIGRAVITY,
            seed=SEEDS_ANTIGRAVITY[0],
            fold_index=0,
            fit_indices=fold.fit_indices,
            query_indices=fold.query_indices,
            normal_trials=len(fold.query_indices),
            normal_alerts=0,
            normal_wilson_upper=0.013388564788679745,
            per_type_auc={
                "level_shift": 0.95,
                "gradual_drift": 0.94,
                "amplitude_scaling": 0.96,
                "period_change": 0.93,
            },
            pooled_auc=0.945,
            passed=True,
        )

        for field, value in (
            ("dataset", "ECG5000_normal_1"),
            ("dataset", ExactStrSubclassAntigravity(DATASET_ANTIGRAVITY)),
            ("seed", True),
            ("seed", 20260722),
            ("fold_index", True),
            ("fold_index", 5),
            ("normal_trials", True),
            ("normal_trials", 282),
            ("normal_alerts", False),
            ("normal_alerts", 284),
            ("normal_wilson_upper", True),
            ("normal_wilson_upper", float("nan")),
            ("pooled_auc", True),
            ("pooled_auc", 0.1),
            ("passed", 1),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                FoldEvidenceAntigravity(**{**valid, field: value})

        invalid_arrays = (
            dict(query_indices=np.arange(283, dtype=np.int64), fit_indices=fold.fit_indices),
            dict(query_indices=fold.query_indices, fit_indices=np.arange(1130, dtype=np.int64)),
        )
        for overrides in invalid_arrays:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                FoldEvidenceAntigravity(**{**valid, **overrides})

        invalid_per_type = (
            {name: 0.95 for name in PRIMARY_TYPES_ANTIGRAVITY[:-1]},
            {**valid["per_type_auc"], "bonus": 0.9},
            {name: 0.95 for name in reversed(PRIMARY_TYPES_ANTIGRAVITY)},
            {**valid["per_type_auc"], "level_shift": np.float64(0.95)},
            {**valid["per_type_auc"], "level_shift": float("inf")},
        )
        for per_type_auc in invalid_per_type:
            with self.subTest(per_type_auc=per_type_auc), self.assertRaises(ValueError):
                FoldEvidenceAntigravity(**{**valid, "per_type_auc": per_type_auc})

    def test_fold_evidence_summary_qualification_and_decision_recompute_semantics_antigravity(
        self,
    ) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            BlockedQualificationAntigravity,
            ValidationDecisionAntigravity,
        )

        first = seed_summary_antigravity(seed=SEEDS_ANTIGRAVITY[0])
        second = seed_summary_antigravity(seed=SEEDS_ANTIGRAVITY[1], failing_fold_index=4)
        evidence = first.fold_records + second.fold_records

        qualification = BlockedQualificationAntigravity(
            dataset=DATASET_ANTIGRAVITY,
            fold_evidence=evidence,
            seed_summaries=(first, second),
            passed=False,
        )
        self.assertFalse(qualification.passed)
        self.assertEqual(
            [(record.seed, record.fold_index) for record in qualification.fold_evidence],
            [(SEEDS_ANTIGRAVITY[0], index) for index in range(5)]
            + [(SEEDS_ANTIGRAVITY[1], index) for index in range(5)],
        )

        decision = ValidationDecisionAntigravity(
            dataset=DATASET_ANTIGRAVITY,
            qualification=qualification,
            status="unstable",
            passed=False,
        )
        self.assertEqual(decision.status, "unstable")

        with self.assertRaises(ValueError):
            ValidationDecisionAntigravity(
                dataset=DATASET_ANTIGRAVITY,
                qualification=qualification,
                status="stable",
                passed=False,
            )
        with self.assertRaises(ValueError):
            BlockedQualificationAntigravity(
                dataset=DATASET_ANTIGRAVITY,
                fold_evidence=evidence[1:] + evidence[:1],
                seed_summaries=(first, second),
                passed=False,
            )


if __name__ == "__main__":
    unittest.main()
