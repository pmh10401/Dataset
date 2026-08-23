from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from .test_qualification_antigravity import passing_scores_antigravity

class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publish_creates_exact_seven_leaves_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.evaluation_antigravity import evaluate_qualification_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.artifacts_antigravity import _LEAVES_ANTIGRAVITY, publish_result_antigravity
        qualification = qualify_methods_antigravity(passing_scores_antigravity())
        with tempfile.TemporaryDirectory() as temp:
            published = publish_result_antigravity(Path(temp) / "result", qualification, evaluate_qualification_antigravity(qualification))
            self.assertEqual(tuple(name for name, _ in published.leaf_sha256), _LEAVES_ANTIGRAVITY)
