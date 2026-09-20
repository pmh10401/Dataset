from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class ArtifactPublicationAntigravityTests(unittest.TestCase):
    @staticmethod
    def _evaluation_antigravity():
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        return evaluate_routing_antigravity(*RoutingEvaluationAntigravityTests._inputs_antigravity())

    def test_publishes_exact_canonical_eight_leaf_inventory_antigravity(self) -> None:
        """Break caught: publication omits a leaf, writes a noncanonical record, or overwrites a root."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import publish_evaluation_antigravity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            frozen = publish_evaluation_antigravity(root, self._evaluation_antigravity())
            self.assertEqual({entry.name for entry in root.iterdir()}, {"context_antigravity.json", "features_antigravity.jsonl", "fold_decisions_antigravity.jsonl", "comparison_metrics_antigravity.json", "family_metrics_antigravity.jsonl", "decision_antigravity.json", "dashboard_snapshot_antigravity.json", "artifact_sha256_antigravity.json"})
            self.assertEqual(len(frozen.leaf_sha256), 8)
            self.assertTrue(all((root / name).read_bytes().endswith(b"\n") for name in ("features_antigravity.jsonl", "fold_decisions_antigravity.jsonl", "family_metrics_antigravity.jsonl")))
            with self.assertRaises(ValueError):
                publish_evaluation_antigravity(root, self._evaluation_antigravity())

    def test_rejects_symlink_and_partial_destination_antigravity(self) -> None:
        """Break caught: a pre-existing symlink or partial root is accepted as a frozen publication."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import publish_evaluation_antigravity

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            partial = base / "partial"
            partial.mkdir()
            (partial / "old").write_text("old", encoding="utf-8")
            with self.assertRaises(ValueError):
                publish_evaluation_antigravity(partial, self._evaluation_antigravity())
            target, linked = base / "target", base / "linked"
            target.mkdir()
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                publish_evaluation_antigravity(linked, self._evaluation_antigravity())


if __name__ == "__main__":
    unittest.main()

