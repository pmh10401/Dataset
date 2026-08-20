from __future__ import annotations

import unittest

from experiments.exp462_ecg_gradual_signature_qualification_antigravity.tests.test_qualification_antigravity import (
    qualification_antigravity,
)


class EvaluationAntigravityTest(unittest.TestCase):
    def test_decision_recomputes_primary_ten_of_ten_without_routing_or_labels_antigravity(self) -> None:
        """Break caught: decision trusts summaries or opens TEST labels."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.predecessor_antigravity import (
            capture_exp461_gradual_authority_antigravity,
        )

        decision = evaluate_qualification_antigravity(
            capture_exp461_gradual_authority_antigravity(),
            qualification_antigravity(),
        )
        self.assertTrue(decision.eligible_for_routing_research)
        self.assertEqual(decision.primary_passed_count, 10)
        self.assertEqual(decision.primary_required_count, 10)
        self.assertEqual(decision.failure_reasons, ())
        self.assertFalse(decision.routing_applied)
        self.assertFalse(decision.test_labels_opened)
        self.assertEqual(decision.label_query_count, 0)


if __name__ == "__main__":
    unittest.main()
