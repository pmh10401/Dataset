from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from .test_qualification_antigravity import passing_scores_antigravity

class ReplayAntigravityTest(unittest.TestCase):
    def test_semantic_replay_rejects_tampered_decision_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.evaluation_antigravity import evaluate_qualification_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.artifacts_antigravity import publish_result_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.replay_antigravity import inspect_replay_at_antigravity
        qualification = qualify_methods_antigravity(passing_scores_antigravity())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "result"; publish_result_antigravity(root, qualification, evaluate_qualification_antigravity(qualification))
            self.assertTrue(inspect_replay_at_antigravity(root).decision.eligible_for_routing_research)
            (root / "decision_antigravity.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError): inspect_replay_at_antigravity(root)
