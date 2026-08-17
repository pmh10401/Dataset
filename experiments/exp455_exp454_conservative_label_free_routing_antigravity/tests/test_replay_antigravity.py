from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class ReplayAntigravityTests(unittest.TestCase):
    def test_semantic_replay_recomputes_and_rejects_tamper_even_with_rehashed_map_antigravity(self) -> None:
        """Break caught: replay trusts stored metrics after a semantic change and rehashed map."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.artifacts_antigravity import publish_evaluation_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.replay_antigravity import inspect_replay_at_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            expected = evaluate_routing_antigravity(*RoutingEvaluationAntigravityTests._inputs_antigravity())
            publish_evaluation_antigravity(root, expected)
            self.assertEqual(inspect_replay_at_antigravity(root), expected)
            context_path = root / "context_antigravity.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["rows"][0]["baseline_confusion"]["fn"] += 1
            context_path.write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            artifact_map = {entry.name: hashlib.sha256(entry.read_bytes()).hexdigest() for entry in root.iterdir() if entry.name != "artifact_sha256_antigravity.json"}
            (root / "artifact_sha256_antigravity.json").write_bytes(json.dumps(artifact_map, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            with self.assertRaises(ValueError):
                inspect_replay_at_antigravity(root)


if __name__ == "__main__":
    unittest.main()
