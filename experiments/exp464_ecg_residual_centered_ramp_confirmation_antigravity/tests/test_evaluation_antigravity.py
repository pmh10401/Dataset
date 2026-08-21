from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity import (
    qualification_antigravity,
)


class EvaluationAntigravityTest(unittest.TestCase):
    def test_decision_is_fixed_c6_only_and_blocks_before_routing_research_antigravity(self) -> None:
        """Break caught: Task 4 promotes another method or opens routing before review."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        decision = evaluate_qualification_antigravity(qualification, authority)
        self.assertEqual(decision.primary_candidate, "template_residual_centered_ramp_confirmed_max")
        self.assertEqual(decision.primary_required_count, 10)
        self.assertEqual(decision.primary_passed_count, 10)
        self.assertEqual(decision.failure_reasons, ())
        self.assertTrue(decision.eligible_for_routing_research)
        self.assertFalse(decision.routing_applied)
        self.assertFalse(decision.test_labels_opened)
        self.assertEqual(decision.label_query_count, 0)
        self.assertEqual(
            decision.status,
            "qualification_passed_ready_for_task5_reproducibility",
        )

    def test_decision_recomputes_c6_gates_from_score_evidence_antigravity(self) -> None:
        """Break caught: caller summaries or booleans bypass the frozen C6 gates."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        expected_decision = evaluate_qualification_antigravity(qualification, authority)
        rows = list(qualification.fold_metrics)
        index = next(
            i
            for i, row in enumerate(rows)
            if row.method == "C6" and row.seed == 20261721 and row.fold_index == 4
        )
        row = rows[index]
        rows[index] = replace(
            row,
            passed=True,
            failure_reasons=(),
        )
        decision = evaluate_qualification_antigravity(
            replace(qualification, fold_metrics=tuple(rows)),
            authority,
        )
        self.assertEqual(decision, expected_decision)


if __name__ == "__main__":
    unittest.main()
