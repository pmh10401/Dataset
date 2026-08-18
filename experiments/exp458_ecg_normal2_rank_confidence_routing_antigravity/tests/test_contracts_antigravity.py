from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
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


def seed_evidence_antigravity(
    *,
    seed: int = SEEDS_ANTIGRAVITY[0],
    per_type: tuple[float, float, float, float] = (0.85, 0.90, 0.90, 0.95),
    wilson: float = 0.05,
) -> object:
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
        RankSeedEvidenceAntigravity,
    )

    return RankSeedEvidenceAntigravity(
        DATASET_ANTIGRAVITY,
        seed,
        100,
        0,
        wilson,
        {name: value for name, value in zip(PRIMARY_TYPES_ANTIGRAVITY, per_type, strict=True)},
        0.90,
        True,
    )


class ContractsAntigravityTest(unittest.TestCase):
    def test_contract_fields_are_exact_and_instances_are_immutable_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankQualificationAntigravity,
            RankSeedEvidenceAntigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(RankSeedEvidenceAntigravity)),
            (
                "dataset",
                "seed",
                "normal_trials",
                "normal_alerts",
                "normal_wilson_upper",
                "per_type_auc",
                "pooled_auc",
                "passed",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(RankQualificationAntigravity)),
            ("dataset", "seed_evidence", "passed"),
        )

        evidence = seed_evidence_antigravity()
        self.assertIsInstance(evidence.per_type_auc, MappingProxyType)
        self.assertEqual(tuple(evidence.per_type_auc), PRIMARY_TYPES_ANTIGRAVITY)
        with self.assertRaises(TypeError):
            evidence.per_type_auc["level_shift"] = 0.0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            evidence.passed = False  # type: ignore[misc]

    def test_seed_evidence_rejects_wrong_scalar_types_vectors_and_bounds_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankSeedEvidenceAntigravity,
        )

        class ExactStrSubclassAntigravity(str):
            pass

        valid = dict(
            dataset=DATASET_ANTIGRAVITY,
            seed=SEEDS_ANTIGRAVITY[0],
            normal_trials=100,
            normal_alerts=0,
            normal_wilson_upper=0.05,
            per_type_auc={name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY},
            pooled_auc=0.90,
            passed=True,
        )
        for field, value in (
            ("dataset", "ECG5000_normal_1"),
            ("dataset", ExactStrSubclassAntigravity(DATASET_ANTIGRAVITY)),
            ("seed", True),
            ("seed", 20260722),
            ("normal_trials", True),
            ("normal_trials", 0),
            ("normal_alerts", False),
            ("normal_alerts", -1),
            ("normal_alerts", 101),
            ("normal_wilson_upper", True),
            ("normal_wilson_upper", 0.051),
            ("normal_wilson_upper", float("nan")),
            ("pooled_auc", True),
            ("pooled_auc", 0.89),
            ("passed", 1),
        ):
            case = {**valid, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                RankSeedEvidenceAntigravity(**case)

        for per_type_auc in (
            {name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY[:-1]},
            {**{name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY}, "bonus": 0.9},
            {name: 0.90 for name in reversed(PRIMARY_TYPES_ANTIGRAVITY)},
            {**{name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY}, "level_shift": True},
            {**{name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY}, "level_shift": np.float64(0.9)},
            {**{name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY}, "level_shift": float("inf")},
        ):
            with self.subTest(per_type_auc=per_type_auc), self.assertRaises(ValueError):
                RankSeedEvidenceAntigravity(**{**valid, "per_type_auc": per_type_auc})

    def test_seed_validator_recomputes_pooled_and_passed_from_numeric_evidence_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankSeedEvidenceAntigravity,
            validate_rank_seed_semantics_antigravity,
        )

        evidence = seed_evidence_antigravity()
        self.assertIs(validate_rank_seed_semantics_antigravity(evidence), evidence)

        for case in (
            dict(
                dataset=DATASET_ANTIGRAVITY,
                seed=SEEDS_ANTIGRAVITY[0],
                normal_trials=100,
                normal_alerts=0,
                normal_wilson_upper=0.05,
                per_type_auc={
                    "level_shift": 0.85,
                    "gradual_drift": 0.90,
                    "amplitude_scaling": 0.90,
                    "period_change": 0.95,
                },
                pooled_auc=0.91,
                passed=True,
            ),
            dict(
                dataset=DATASET_ANTIGRAVITY,
                seed=SEEDS_ANTIGRAVITY[0],
                normal_trials=100,
                normal_alerts=0,
                normal_wilson_upper=0.05,
                per_type_auc={
                    "level_shift": 0.85,
                    "gradual_drift": 0.90,
                    "amplitude_scaling": 0.90,
                    "period_change": 0.95,
                },
                pooled_auc=0.90,
                passed=False,
            ),
        ):
            with self.subTest(case=case), self.assertRaises(ValueError):
                RankSeedEvidenceAntigravity(**case)

    def test_seed_evidence_uses_inclusive_boundaries_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankSeedEvidenceAntigravity,
        )

        evidence = RankSeedEvidenceAntigravity(
            DATASET_ANTIGRAVITY,
            SEEDS_ANTIGRAVITY[0],
            100,
            0,
            0.05,
            {
                "level_shift": 0.85,
                "gradual_drift": 0.90,
                "amplitude_scaling": 0.90,
                "period_change": 0.95,
            },
            0.90,
            True,
        )
        self.assertTrue(evidence.passed)

    def test_qualification_requires_both_ordered_matching_seed_records_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankQualificationAntigravity,
            RankSeedEvidenceAntigravity,
        )

        first = seed_evidence_antigravity(seed=SEEDS_ANTIGRAVITY[0])
        second = seed_evidence_antigravity(seed=SEEDS_ANTIGRAVITY[1])
        qualification = RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), True)
        self.assertEqual(qualification.seed_evidence, (first, second))

        for evidence in (
            [first, second],
            (first,),
            (second, first),
            (first, first),
        ):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                RankQualificationAntigravity(DATASET_ANTIGRAVITY, evidence, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), 1)  # type: ignore[arg-type]

    def test_qualification_recomputes_both_seed_gate_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankQualificationAntigravity,
            validate_rank_qualification_semantics_antigravity,
        )

        first = seed_evidence_antigravity(seed=SEEDS_ANTIGRAVITY[0])
        second = seed_evidence_antigravity(seed=SEEDS_ANTIGRAVITY[1])
        qualification = RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), True)
        self.assertIs(validate_rank_qualification_semantics_antigravity(qualification), qualification)

        with self.assertRaises(ValueError):
            RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), False)


if __name__ == "__main__":
    unittest.main()
