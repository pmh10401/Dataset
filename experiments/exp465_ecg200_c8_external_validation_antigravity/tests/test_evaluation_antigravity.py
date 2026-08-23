from __future__ import annotations
import unittest
from experiments.exp465_ecg200_c8_external_validation_antigravity.tests.test_qualification_antigravity import passing_scores_antigravity

class EvaluationAntigravityTest(unittest.TestCase):
    def test_c8_only_decision_requires_zero_confirmation_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.evaluation_antigravity import evaluate_qualification_antigravity
        decision = evaluate_qualification_antigravity(qualify_methods_antigravity(passing_scores_antigravity()))
        self.assertFalse(decision.eligible_for_routing_research)
        self.assertEqual(decision.primary_candidate, "C8")
