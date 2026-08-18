from __future__ import annotations

from fractions import Fraction
import unittest

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_contracts_antigravity import (
    DATASET_ANTIGRAVITY,
    SEEDS_ANTIGRAVITY,
    seed_summary_antigravity,
)


def qualification_antigravity(*, failing_seed: int | None = None, failing_fold: int = 4):
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
        BlockedQualificationAntigravity,
    )

    summaries = tuple(
        seed_summary_antigravity(
            seed=seed,
            failing_fold_index=failing_fold if seed == failing_seed else None,
        )
        for seed in SEEDS_ANTIGRAVITY
    )
    return BlockedQualificationAntigravity(
        dataset=DATASET_ANTIGRAVITY,
        fold_evidence=summaries[0].fold_records + summaries[1].fold_records,
        seed_summaries=summaries,
        passed=failing_seed is None,
    )


class EvaluationAntigravityTest(unittest.TestCase):
    def test_stable_requires_all_ten_ordered_fold_seed_records_antigravity(self) -> None:
        """Break caught: evaluator sorts, truncates, or treats fewer than 10 passes as stable."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            evaluate_blocked_qualification_antigravity,
        )

        stable = evaluate_blocked_qualification_antigravity(qualification_antigravity())
        self.assertEqual(stable.status, "stable")
        self.assertTrue(stable.passed)
        self.assertEqual(
            [(record.seed, record.fold_index) for record in stable.qualification.fold_evidence],
            [(seed, fold_index) for seed in SEEDS_ANTIGRAVITY for fold_index in range(5)],
        )

        unstable = evaluate_blocked_qualification_antigravity(
            qualification_antigravity(failing_seed=SEEDS_ANTIGRAVITY[1], failing_fold=4)
        )
        self.assertEqual(unstable.status, "unstable")
        self.assertFalse(unstable.passed)

    def test_exposes_exp458_metrics_as_predecessor_context_only_antigravity(self) -> None:
        """Break caught: Exp459 recomputes TEST metrics or changes the frozen Exp458 context."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY,
            evaluate_blocked_qualification_antigravity,
        )

        decision = evaluate_blocked_qualification_antigravity(qualification_antigravity())
        self.assertEqual(decision.status, "stable")
        self.assertEqual(
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY,
            {
                "experiment": "Exp458",
                "tp": 617,
                "fp": 66,
                "fn": 829,
                "tn": 70788,
                "f1": 0.5796148426491311,
            },
        )
        self.assertEqual(
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY["f1"],
            float(Fraction(2 * 617, 2 * 617 + 66 + 829)),
        )
        self.assertNotIn("confusion_matrix_antigravity", repr(decision).lower())

    def test_rejects_forged_or_mutated_qualification_at_evaluation_boundary_antigravity(self) -> None:
        """Break caught: caller-mutated evidence bypasses the exact qualification contract."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            evaluate_blocked_qualification_antigravity,
        )

        forged = qualification_antigravity()
        object.__setattr__(forged, "passed", False)
        with self.assertRaises(ValueError):
            evaluate_blocked_qualification_antigravity(forged)

        valid = qualification_antigravity()
        object.__setattr__(
            valid,
            "fold_evidence",
            valid.fold_evidence[1:] + valid.fold_evidence[:1],
        )
        with self.assertRaises(ValueError):
            evaluate_blocked_qualification_antigravity(valid)

    def test_rejects_bool_int_container_and_per_type_post_construction_mutation_antigravity(self) -> None:
        """Break caught: post-construction mutation changes exact types while preserving equal values."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
            evaluate_blocked_qualification_antigravity,
        )

        bool_as_int = qualification_antigravity()
        object.__setattr__(bool_as_int, "passed", 1)
        with self.assertRaises(ValueError):
            evaluate_blocked_qualification_antigravity(bool_as_int)

        list_container = qualification_antigravity()
        object.__setattr__(list_container, "fold_evidence", list(list_container.fold_evidence))
        with self.assertRaises(ValueError):
            evaluate_blocked_qualification_antigravity(list_container)

        extra_auc = qualification_antigravity()
        record = extra_auc.fold_evidence[0]
        object.__setattr__(record, "per_type_auc", {**dict(record.per_type_auc), "bonus": 0.9})
        with self.assertRaises(ValueError):
            evaluate_blocked_qualification_antigravity(extra_auc)


if __name__ == "__main__":
    unittest.main()
