"""Red-first exact-order replay fixture for the Exp452 controlled fork."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


class ReplayAntigravityTests(unittest.TestCase):
    def test_exact_replay_contract_requires_ordered_integer_lists_antigravity(self) -> None:
        """Break caught: set-equivalent reordered evidence is not a valid frozen prediction."""
        from experiments.exp452_exp449_index_order_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp452_exp449_index_order_repair_antigravity.replay_antigravity import inspect_replay_at_antigravity
        from experiments.exp452_exp449_index_order_repair_antigravity.tests.test_evaluation_antigravity import EvaluationAntigravityTests

        with tempfile.TemporaryDirectory(prefix="exp452_replay_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"
            dependencies = EvaluationAntigravityTests()._dependencies_antigravity(root)
            _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            payload = json.loads((root / "evaluation_antigravity.json").read_text())
            candidate = next(iter(payload["dataset_evidence"][0]["candidates"].values()))
            candidate["indices"] = list(reversed(candidate["indices"]))
            from experiments.exp452_exp449_index_order_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
            import hashlib
            path = root / "evaluation_antigravity.json"
            path.write_bytes(canonical_json_bytes_antigravity(payload))
            hashes = json.loads((root / "artifact_sha256_antigravity.json").read_text())
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity(hashes))
            with self.assertRaisesRegex(ValueError, "frozen prediction indices differ from evidence"):
                inspect_replay_at_antigravity(root, expected_prediction_authority=dependencies.capture_task5_prediction_antigravity())


if __name__ == "__main__":
    unittest.main()
