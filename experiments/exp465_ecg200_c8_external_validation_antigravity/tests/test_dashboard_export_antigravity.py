from __future__ import annotations
import unittest
from .test_qualification_antigravity import passing_scores_antigravity

class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_uses_simple_korean_and_c8_summary_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.evaluation_antigravity import evaluate_qualification_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.dashboard_export_antigravity import export_dashboard_antigravity
        qualification = qualify_methods_antigravity(passing_scores_antigravity())
        snapshot = export_dashboard_antigravity(qualification, evaluate_qualification_antigravity(qualification))
        self.assertEqual(snapshot["후보"], "C8")
        self.assertLess(snapshot["통과_기록"], 10)
