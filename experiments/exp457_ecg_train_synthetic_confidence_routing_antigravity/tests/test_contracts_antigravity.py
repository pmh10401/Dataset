from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType
import unittest


PRIMARY_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
DATASET_ANTIGRAVITY = "ECG5000_normal_1"
SEEDS_ANTIGRAVITY = (20260721, 20261721)


def seed_evidence_antigravity(seed: int = SEEDS_ANTIGRAVITY[0]):
    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
        TrainSeedEvidenceAntigravity,
    )

    return TrainSeedEvidenceAntigravity(
        DATASET_ANTIGRAVITY,
        seed,
        100,
        0,
        0.03699349820698568,
        0.75,
        {name: 0.75 for name in PRIMARY_TYPES_ANTIGRAVITY},
        True,
    )


class ContractsAntigravityTest(unittest.TestCase):
    def test_contract_fields_are_exact_and_instances_are_immutable_antigravity(self) -> None:
        """Break caught: later tasks receive an invented field or mutable result record."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
            RouteDecisionAntigravity,
            TrainQualificationAntigravity,
            TrainSeedEvidenceAntigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(TrainSeedEvidenceAntigravity)),
            (
                "dataset",
                "seed",
                "normal_trials",
                "normal_alerts",
                "normal_wilson_upper",
                "pooled_sensitivity",
                "per_type_sensitivity",
                "passed",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(TrainQualificationAntigravity)),
            ("dataset", "seed_evidence", "passed"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(RouteDecisionAntigravity)),
            ("dataset", "train_gate", "structure_gate", "route"),
        )

        evidence = seed_evidence_antigravity()
        self.assertIsInstance(evidence.per_type_sensitivity, MappingProxyType)
        self.assertEqual(tuple(evidence.per_type_sensitivity), PRIMARY_TYPES_ANTIGRAVITY)
        with self.assertRaises(TypeError):
            evidence.per_type_sensitivity["level_shift"] = 0.0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            evidence.passed = False  # type: ignore[misc]

    def test_seed_evidence_rejects_wrong_dataset_seed_and_integer_contracts_antigravity(self) -> None:
        """Break caught: evidence from another dataset/seed or bool-as-int enters Gate 1."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
            TrainSeedEvidenceAntigravity,
        )

        valid = dict(
            dataset=DATASET_ANTIGRAVITY,
            seed=SEEDS_ANTIGRAVITY[0],
            normal_trials=100,
            normal_alerts=2,
            normal_wilson_upper=0.049,
            pooled_sensitivity=0.75,
            per_type_sensitivity={name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY},
            passed=True,
        )
        for field, value in (
            ("dataset", "ChlorineConcentration_normal_1"),
            ("seed", 1),
            ("seed", True),
            ("normal_trials", True),
            ("normal_trials", 100.0),
            ("normal_trials", 0),
            ("normal_alerts", False),
            ("normal_alerts", -1),
            ("normal_alerts", 101),
            ("passed", 1),
        ):
            case = {**valid, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                TrainSeedEvidenceAntigravity(**case)

    def test_seed_evidence_rejects_nonfinite_out_of_range_and_wrong_numeric_types_antigravity(self) -> None:
        """Break caught: invalid floating evidence can silently satisfy a threshold comparison."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
            TrainSeedEvidenceAntigravity,
        )

        valid = dict(
            dataset=DATASET_ANTIGRAVITY,
            seed=SEEDS_ANTIGRAVITY[0],
            normal_trials=100,
            normal_alerts=2,
            normal_wilson_upper=0.049,
            pooled_sensitivity=0.75,
            per_type_sensitivity={name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY},
            passed=True,
        )
        for field, value in (
            ("normal_wilson_upper", True),
            ("normal_wilson_upper", 0),
            ("normal_wilson_upper", float("nan")),
            ("normal_wilson_upper", 1.01),
            ("pooled_sensitivity", False),
            ("pooled_sensitivity", float("inf")),
            ("pooled_sensitivity", -0.01),
        ):
            case = {**valid, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                TrainSeedEvidenceAntigravity(**case)

        for sensitivities in (
            {name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY[:-1]},
            {**{name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY}, "secondary_spike": 0.9},
            {name: 0.60 for name in reversed(PRIMARY_TYPES_ANTIGRAVITY)},
            {**{name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY}, "level_shift": True},
            {**{name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY}, "level_shift": float("nan")},
            {**{name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY}, "level_shift": 1.01},
        ):
            with self.subTest(sensitivities=sensitivities), self.assertRaises(ValueError):
                TrainSeedEvidenceAntigravity(**{**valid, "per_type_sensitivity": sensitivities})

    def test_qualification_requires_both_ordered_matching_seed_records_antigravity(self) -> None:
        """Break caught: Gate 1 evidence is incomplete, reordered, duplicated, or cross-dataset."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
            TrainQualificationAntigravity,
            TrainSeedEvidenceAntigravity,
        )

        first = seed_evidence_antigravity(SEEDS_ANTIGRAVITY[0])
        second = seed_evidence_antigravity(SEEDS_ANTIGRAVITY[1])
        qualification = TrainQualificationAntigravity(
            DATASET_ANTIGRAVITY, (first, second), True
        )
        self.assertEqual(qualification.seed_evidence, (first, second))

        cross_dataset = TrainSeedEvidenceAntigravity(
            "ECG5000_normal_2",
            SEEDS_ANTIGRAVITY[1],
            100,
            2,
            0.049,
            0.75,
            {name: 0.60 for name in PRIMARY_TYPES_ANTIGRAVITY},
            True,
        )
        for evidence in (
            [first, second],
            (first,),
            (second, first),
            (first, first),
            (first, cross_dataset),
        ):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                TrainQualificationAntigravity(DATASET_ANTIGRAVITY, evidence, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TrainQualificationAntigravity(DATASET_ANTIGRAVITY, (first, second), 1)  # type: ignore[arg-type]

    def test_qualification_recomputes_numeric_seed_and_aggregate_gate_semantics_antigravity(self) -> None:
        """Break caught: stored Wilson, pooled, seed, or aggregate flags override Gate 1 math."""
        from dataclasses import replace
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
            TrainQualificationAntigravity,
        )

        first = seed_evidence_antigravity(SEEDS_ANTIGRAVITY[0])
        second = seed_evidence_antigravity(SEEDS_ANTIGRAVITY[1])
        broken = (
            (replace(first, normal_wilson_upper=0.04), second, True),
            (replace(first, pooled_sensitivity=0.74), second, True),
            (replace(first, passed=False), second, True),
            (first, second, False),
        )
        for left, right, aggregate in broken:
            with self.subTest(left=left, aggregate=aggregate), self.assertRaises(ValueError):
                TrainQualificationAntigravity(
                    DATASET_ANTIGRAVITY, (left, right), aggregate
                )

    def test_route_decision_rejects_wrong_dataset_bool_flags_and_route_names_antigravity(self) -> None:
        """Break caught: downstream results carry an out-of-scope route or non-boolean gate."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
            RouteDecisionAntigravity,
        )

        candidate = RouteDecisionAntigravity(DATASET_ANTIGRAVITY, True, True, "candidate")
        baseline = RouteDecisionAntigravity(DATASET_ANTIGRAVITY, False, True, "baseline")
        self.assertEqual((candidate.route, baseline.route), ("candidate", "baseline"))

        for args in (
            ("other", True, True, "candidate"),
            (DATASET_ANTIGRAVITY, 1, True, "candidate"),
            (DATASET_ANTIGRAVITY, True, 0, "baseline"),
            (DATASET_ANTIGRAVITY, True, True, "hydra_local_gap"),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                RouteDecisionAntigravity(*args)

    def test_task4_result_contracts_are_exact_frozen_and_type_strict_antigravity(self) -> None:
        """Break caught: publication/evaluation records gain mutable or ambiguous scalar fields."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
            CriterionOutcomeAntigravity,
            EvaluationResultAntigravity,
            Exp456ResultAuthorityAntigravity,
            FamilyMetricAntigravity,
            PublicationEvidenceAntigravity,
            PublishedResultAntigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(CriterionOutcomeAntigravity)),
            ("metric", "operator", "threshold", "actual", "passed"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(FamilyMetricAntigravity)),
            ("family", "fn_delta", "fp_delta"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(EvaluationResultAntigravity)),
            ("comparisons", "family_metrics", "decisions", "criteria", "passed", "status"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp456ResultAuthorityAntigravity)),
            (
                "authority",
                "features",
                "predecessor_comparisons",
                "artifact_map_sha256",
                "comparison_sha256",
                "decision_sha256",
                "label_query_count",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(PublicationEvidenceAntigravity)),
            ("exp456_authority", "qualifications", "features", "result"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(PublishedResultAntigravity)),
            ("root", "leaf_sha256", "artifact_map_sha256"),
        )
        criterion = CriterionOutcomeAntigravity("final_fn", "<=", 834, 829, True)
        family = FamilyMetricAntigravity("ECG5000", -8, 0)
        with self.assertRaises(FrozenInstanceError):
            criterion.passed = False  # type: ignore[misc]
        for args in (
            ("final_fn", "<=", 834, True, True),
            ("final_f1", ">", 0.5, 1, True),
            ("unknown", "<=", 834, 829, True),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                CriterionOutcomeAntigravity(*args)
        with self.assertRaises(ValueError):
            FamilyMetricAntigravity("Other", -8, 0)
        with self.assertRaises(ValueError):
            FamilyMetricAntigravity("ECG5000", True, 0)


if __name__ == "__main__":
    unittest.main()
