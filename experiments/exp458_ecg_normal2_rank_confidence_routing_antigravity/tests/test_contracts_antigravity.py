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
TASK3_AUC_BY_SEED_ANTIGRAVITY = {
    20260721: (
        0.9796866524522819,
        0.9052489545815847,
        0.9951992643379718,
        0.9104679076956519,
    ),
    20261721: (
        0.9761215364958586,
        0.895775202359648,
        0.9943593234193259,
        0.9278161686372973,
    ),
}


def seed_evidence_antigravity(
    *,
    seed: int = SEEDS_ANTIGRAVITY[0],
    per_type: tuple[float, float, float, float] | None = None,
    pooled: float | None = None,
    trials: int = 1413,
    alerts: int = 6,
    wilson: float = 0.00923330730695328,
    passed: bool = True,
) -> object:
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
        RankSeedEvidenceAntigravity,
    )

    values = TASK3_AUC_BY_SEED_ANTIGRAVITY[seed] if per_type is None else per_type
    pooled_value = float(sum(values) / len(values)) if pooled is None else pooled
    return RankSeedEvidenceAntigravity(
        DATASET_ANTIGRAVITY,
        seed,
        trials,
        alerts,
        wilson,
        {name: value for name, value in zip(PRIMARY_TYPES_ANTIGRAVITY, values, strict=True)},
        pooled_value,
        passed,
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
            normal_trials=1413,
            normal_alerts=6,
            normal_wilson_upper=0.00923330730695328,
            per_type_auc={
                "level_shift": 0.9796866524522819,
                "gradual_drift": 0.9052489545815847,
                "amplitude_scaling": 0.9951992643379718,
                "period_change": 0.9104679076956519,
            },
            pooled_auc=0.9476506947668726,
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
            ("normal_alerts", 1414),
            ("normal_wilson_upper", True),
            ("normal_wilson_upper", float("nan")),
            ("pooled_auc", True),
            ("passed", 1),
        ):
            case = {**valid, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                RankSeedEvidenceAntigravity(**case)

        for per_type_auc in (
            {name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY[:-1]},
            {**{name: 0.90 for name in PRIMARY_TYPES_ANTIGRAVITY}, "bonus": 0.9},
            {name: 0.90 for name in reversed(PRIMARY_TYPES_ANTIGRAVITY)},
            {**valid["per_type_auc"], "level_shift": True},
            {**valid["per_type_auc"], "level_shift": np.float64(0.9)},
            {**valid["per_type_auc"], "level_shift": float("inf")},
        ):
            with self.subTest(per_type_auc=per_type_auc), self.assertRaises(ValueError):
                RankSeedEvidenceAntigravity(**{**valid, "per_type_auc": per_type_auc})

    def test_seed_validator_recomputes_wilson_pooled_and_passed_without_exact_task3_values_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankSeedEvidenceAntigravity,
            validate_rank_seed_semantics_antigravity,
        )

        evidence = seed_evidence_antigravity()
        self.assertIs(validate_rank_seed_semantics_antigravity(evidence), evidence)
        high_auc = RankSeedEvidenceAntigravity(
            DATASET_ANTIGRAVITY,
            SEEDS_ANTIGRAVITY[0],
            1413,
            6,
            0.00923330730695328,
            {name: 0.99 for name in PRIMARY_TYPES_ANTIGRAVITY},
            0.99,
            True,
        )
        self.assertIs(validate_rank_seed_semantics_antigravity(high_auc), high_auc)
        failing = RankSeedEvidenceAntigravity(
            DATASET_ANTIGRAVITY,
            SEEDS_ANTIGRAVITY[0],
            1413,
            6,
            0.00923330730695328,
            {
                "level_shift": 0.84,
                "gradual_drift": 0.95,
                "amplitude_scaling": 0.95,
                "period_change": 0.95,
            },
            0.9225,
            False,
        )
        self.assertFalse(failing.passed)
        self.assertIs(validate_rank_seed_semantics_antigravity(failing), failing)

        for case in (
            dict(
                dataset=DATASET_ANTIGRAVITY,
                seed=SEEDS_ANTIGRAVITY[0],
                normal_trials=1413,
                normal_alerts=6,
                normal_wilson_upper=0.0,
                per_type_auc={
                    "level_shift": 0.99,
                    "gradual_drift": 0.99,
                    "amplitude_scaling": 0.99,
                    "period_change": 0.99,
                },
                pooled_auc=0.99,
                passed=True,
            ),
            dict(
                dataset=DATASET_ANTIGRAVITY,
                seed=SEEDS_ANTIGRAVITY[0],
                normal_trials=1413,
                normal_alerts=6,
                normal_wilson_upper=0.00923330730695328,
                per_type_auc={
                    "level_shift": 0.9796866524522819,
                    "gradual_drift": 0.9052489545815847,
                    "amplitude_scaling": 0.9951992643379718,
                    "period_change": 0.9104679076956519,
                },
                pooled_auc=0.1,
                passed=True,
            ),
        ):
            with self.subTest(case=case), self.assertRaises(ValueError):
                RankSeedEvidenceAntigravity(**case)

    def test_seed_validator_revalidates_exact_fields_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            validate_rank_seed_semantics_antigravity,
        )

        evidence = seed_evidence_antigravity()

        object.__setattr__(evidence, "dataset", "ECG5000_normal_1")
        with self.assertRaises(ValueError):
            validate_rank_seed_semantics_antigravity(evidence)

        object.__setattr__(evidence, "dataset", DATASET_ANTIGRAVITY)
        object.__setattr__(evidence, "seed", True)
        with self.assertRaises(ValueError):
            validate_rank_seed_semantics_antigravity(evidence)

        object.__setattr__(evidence, "seed", SEEDS_ANTIGRAVITY[0])
        object.__setattr__(evidence, "normal_trials", True)
        with self.assertRaises(ValueError):
            validate_rank_seed_semantics_antigravity(evidence)

    def test_seed_evidence_uses_inclusive_boundaries_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankSeedEvidenceAntigravity,
        )

        evidence = RankSeedEvidenceAntigravity(
            DATASET_ANTIGRAVITY,
            SEEDS_ANTIGRAVITY[0],
            1413,
            6,
            0.00923330730695328,
            {
                "level_shift": 0.9796866524522819,
                "gradual_drift": 0.9052489545815847,
                "amplitude_scaling": 0.9951992643379718,
                "period_change": 0.9104679076956519,
            },
            0.9476506947668726,
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
        second = seed_evidence_antigravity(
            seed=SEEDS_ANTIGRAVITY[1],
            per_type=(0.84, 0.95, 0.95, 0.95),
            passed=False,
        )
        failing = RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), False)
        self.assertIs(validate_rank_qualification_semantics_antigravity(failing), failing)
        second = seed_evidence_antigravity(seed=SEEDS_ANTIGRAVITY[1])
        qualification = RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), True)
        self.assertIs(validate_rank_qualification_semantics_antigravity(qualification), qualification)

        with self.assertRaises(ValueError):
            RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), False)

    def test_qualification_validator_revalidates_seed_order_and_nested_semantics_antigravity(self) -> None:
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
            RankQualificationAntigravity,
            validate_rank_qualification_semantics_antigravity,
        )

        first = seed_evidence_antigravity(seed=SEEDS_ANTIGRAVITY[0])
        second = seed_evidence_antigravity(seed=SEEDS_ANTIGRAVITY[1])
        qualification = RankQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), True)

        object.__setattr__(qualification, "seed_evidence", (second, first))
        with self.assertRaises(ValueError):
            validate_rank_qualification_semantics_antigravity(qualification)

        object.__setattr__(qualification, "seed_evidence", (first, second))
        object.__setattr__(second, "pooled_auc", 0.1)
        with self.assertRaises(ValueError):
            validate_rank_qualification_semantics_antigravity(qualification)


if __name__ == "__main__":
    unittest.main()
