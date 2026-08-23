from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from hashlib import sha256
import json
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

    def test_replay_rejects_rank_preserving_c8_score_tamper_with_refreshed_hashes_antigravity(self):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.qualification_antigravity import qualify_methods_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.evaluation_antigravity import evaluate_qualification_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity, publish_result_antigravity
        from experiments.exp465_ecg200_c8_external_validation_antigravity.replay_antigravity import inspect_replay_at_antigravity
        qualification = qualify_methods_antigravity(passing_scores_antigravity())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "result"; publish_result_antigravity(root, qualification, evaluate_qualification_antigravity(qualification))
            rows = [json.loads(line) for line in (root / "method_fold_metrics_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]
            row = next(item for item in rows if item["method"] == "C8")
            row["primary_scores"]["level_shift"] = [value * 0.9 for value in row["primary_scores"]["level_shift"]]
            metrics = b"".join(canonical_json_bytes_antigravity(item) for item in rows); (root / "method_fold_metrics_antigravity.jsonl").write_bytes(metrics)
            context = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8")); context["method_fold_metrics_sha256"] = sha256(metrics).hexdigest(); (root / "context_antigravity.json").write_bytes(canonical_json_bytes_antigravity(context))
            leaves = [path for path in root.iterdir() if path.name != "artifact_sha256_antigravity.json"]; (root / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity({path.name: sha256(path.read_bytes()).hexdigest() for path in leaves}))
            with self.assertRaises(ValueError): inspect_replay_at_antigravity(root)
