from __future__ import annotations

import unittest

from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_qualification_antigravity import (
    qualification_antigravity,
)


class EvaluationAntigravityTest(unittest.TestCase):
    def test_evaluation_is_blocked_without_routing_or_test_labels_antigravity(self) -> None:
        """Break caught: evaluation opens TEST labels or treats diagnostics as M2."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        decision = evaluate_qualification_antigravity(
            capture_exp462_authority_antigravity(), qualification_antigravity()
        )
        self.assertEqual(decision.status, "qualification_failed_blocked_before_routing_research")
        self.assertFalse(decision.eligible_for_routing_research)
        self.assertEqual(decision.primary_required_count, 10)
        self.assertFalse(decision.routing_applied)
        self.assertFalse(decision.test_labels_opened)
        self.assertEqual(decision.label_query_count, 0)
        self.assertIn("m0_control_not_reproduced", decision.failure_reasons)
        self.assertIn("m6_control_not_reproduced", decision.failure_reasons)

    def test_decision_recomputes_all_m2_gate_reasons_from_score_evidence_antigravity(self) -> None:
        """Break caught: summaries or caller booleans replace the frozen M2 gates."""
        from dataclasses import replace

        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        qualification = qualification_antigravity()
        rows = list(qualification.fold_metrics)
        index = next(i for i, row in enumerate(rows) if row.method == "M2")
        row = rows[index]
        rows[index] = replace(
            row,
            primary_scores={**row.primary_scores, "gradual_drift": row.normal_scores},
            passed=True,
            failure_reasons=(),
        )
        decision = evaluate_qualification_antigravity(
            capture_exp462_authority_antigravity(), replace(qualification, fold_metrics=tuple(rows))
        )
        self.assertFalse(decision.eligible_for_routing_research)
        self.assertIn("gradual_drift_auc_below_min", decision.failure_reasons)


if __name__ == "__main__":
    unittest.main()
