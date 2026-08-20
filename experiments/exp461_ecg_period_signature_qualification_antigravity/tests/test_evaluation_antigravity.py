from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.exp461_ecg_period_signature_qualification_antigravity.tests.test_qualification_antigravity import (
    qualification_antigravity,
)


class EvaluationAntigravityTest(unittest.TestCase):
    def test_decision_recomputes_primary_summary_without_routing_or_labels_antigravity(self) -> None:
        """Break caught: evaluator trusts summary text, applies routing, or allows label access."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp460_period_authority_antigravity,
        )

        authority = capture_exp460_period_authority_antigravity()
        qualification = qualification_antigravity()
        decision = evaluate_qualification_antigravity(authority, qualification)

        self.assertTrue(decision.eligible_for_routing_research)
        self.assertEqual(decision.primary_passed_count, 10)
        self.assertEqual(decision.primary_required_count, 10)
        self.assertEqual(decision.label_query_count, 0)
        self.assertFalse(decision.routing_applied)
        self.assertFalse(decision.test_labels_opened)
        self.assertEqual(decision.failure_reasons, ())

    def test_decision_rejects_summary_forgery_and_primary_fold_failure_antigravity(self) -> None:
        """Break caught: evaluator accepts forged summaries or a failed primary fold."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp460_period_authority_antigravity,
        )

        authority = capture_exp460_period_authority_antigravity()
        qualification = qualification_antigravity()
        rows = list(qualification.fold_metrics)
        for index, row in enumerate(rows):
            if row.method == "hydra_period_rank_max" and row.seed == 20261721 and row.fold_index == 4:
                per_type_auc = {**row.per_type_auc, "period_change": 0.8004124541019063}
                rows[index] = replace(
                    row,
                    per_type_auc=per_type_auc,
                    pooled_auc=sum(per_type_auc.values()) / len(per_type_auc),
                    passed=False,
                    failure_reasons=("period_change_auc_below_min",),
                )
                break
        forged = replace(qualification, fold_metrics=tuple(rows), summaries=qualification.summaries)

        decision = evaluate_qualification_antigravity(authority, forged)

        self.assertFalse(decision.eligible_for_routing_research)
        self.assertEqual(decision.primary_passed_count, 9)
        self.assertIn("primary_candidate_not_10_of_10", decision.failure_reasons)
        self.assertIn(
            "historical_period_change_not_strictly_improved:20261721:4",
            decision.failure_reasons,
        )


if __name__ == "__main__":
    unittest.main()
