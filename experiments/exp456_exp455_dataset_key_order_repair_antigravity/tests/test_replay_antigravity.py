from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class ReplayAntigravityTests(unittest.TestCase):
    @staticmethod
    def _inputs_antigravity():
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests
        return RoutingEvaluationAntigravityTests._inputs_antigravity()

    @staticmethod
    def _rehash_antigravity(root: Path) -> None:
        artifact_map = {entry.name: hashlib.sha256(entry.read_bytes()).hexdigest() for entry in root.iterdir() if entry.name != "artifact_sha256_antigravity.json"}
        (root / "artifact_sha256_antigravity.json").write_bytes(json.dumps(artifact_map, sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def test_semantic_replay_recomputes_and_rejects_tamper_even_with_rehashed_map_antigravity(self) -> None:
        """Break caught: replay trusts stored metrics after a semantic change and rehashed map."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import publish_evaluation_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.replay_antigravity import _inspect_replay_with_dependencies_antigravity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            authority, features, decisions = self._inputs_antigravity()
            expected = evaluate_routing_antigravity(authority, features, decisions)
            publish_evaluation_antigravity(root, expected)
            self.assertEqual(_inspect_replay_with_dependencies_antigravity(root, authority, features), expected)
            context_path = root / "context_antigravity.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["rows"][0]["baseline_confusion"]["fn"] += 1
            context_path.write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            self._rehash_antigravity(root)
            with self.assertRaises(ValueError):
                _inspect_replay_with_dependencies_antigravity(root, authority, features)

    def test_replay_rejects_self_consistent_rehash_with_mutated_frozen_authority_binding_antigravity(self) -> None:
        """Break caught: a rewritten context, decisions, semantic leaves, and map can replace Exp454 authority evidence."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import _payloads_antigravity, publish_evaluation_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.replay_antigravity import _inspect_replay_with_dependencies_antigravity, _reconstruct_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import leave_one_dataset_out_antigravity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            expected_authority, expected_features, expected_decisions = self._inputs_antigravity()
            publish_evaluation_antigravity(root, evaluate_routing_antigravity(expected_authority, expected_features, expected_decisions))
            context_path = root / "context_antigravity.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["rows"][0]["baseline_confusion"]["fn"] += 1
            context_path.write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            leaves = {entry.name: entry.read_bytes() for entry in root.iterdir()}
            authority, features, _decisions = _reconstruct_antigravity(leaves)
            outcomes = {row.dataset: (row.baseline_confusion, row.candidate_confusion) for row in authority.rows if row.candidate_confusion is not None}
            evaluation = evaluate_routing_antigravity(authority, features, leave_one_dataset_out_antigravity(features, outcomes))
            for name, payload in _payloads_antigravity(evaluation).items():
                (root / name).write_bytes(payload)
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["exp454_leaf_sha256"][0][1] = "b" * 64
            context_path.write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            self._rehash_antigravity(root)
            with self.assertRaises(ValueError):
                _inspect_replay_with_dependencies_antigravity(root, expected_authority, expected_features)

    def test_injected_replay_rejects_self_consistent_context_mutation_with_original_binding_antigravity(self) -> None:
        """Break caught: new context outcomes, rules, every semantic leaf, and map replace the captured 48-row authority."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import _payloads_antigravity, publish_evaluation_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.replay_antigravity import _inspect_replay_with_dependencies_antigravity, _reconstruct_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import leave_one_dataset_out_antigravity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            expected_authority, expected_features, expected_decisions = self._inputs_antigravity()
            publish_evaluation_antigravity(root, evaluate_routing_antigravity(expected_authority, expected_features, expected_decisions))
            context_path = root / "context_antigravity.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["rows"][0]["baseline_confusion"]["fn"] += 1
            context_path.write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            leaves = {entry.name: entry.read_bytes() for entry in root.iterdir()}
            authority, features, _decisions = _reconstruct_antigravity(leaves)
            outcomes = {row.dataset: (row.baseline_confusion, row.candidate_confusion) for row in authority.rows if row.candidate_confusion is not None}
            evaluation = evaluate_routing_antigravity(authority, features, leave_one_dataset_out_antigravity(features, outcomes))
            for name, payload in _payloads_antigravity(evaluation).items():
                (root / name).write_bytes(payload)
            self._rehash_antigravity(root)
            with self.assertRaises(ValueError):
                _inspect_replay_with_dependencies_antigravity(root, expected_authority, expected_features)

    def test_injected_replay_rejects_self_consistent_feature_mutation_antigravity(self) -> None:
        """Break caught: rewritten label-free feature rows, decisions, semantic leaves, and map replace captured features."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import _payloads_antigravity, publish_evaluation_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.replay_antigravity import _inspect_replay_with_dependencies_antigravity, _reconstruct_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import leave_one_dataset_out_antigravity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            expected_authority, expected_features, expected_decisions = self._inputs_antigravity()
            publish_evaluation_antigravity(root, evaluate_routing_antigravity(expected_authority, expected_features, expected_decisions))
            feature_path = root / "features_antigravity.jsonl"
            rows = [json.loads(line) for line in feature_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["baseline_count"] += 100
            feature_path.write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows))
            leaves = {entry.name: entry.read_bytes() for entry in root.iterdir()}
            authority, features, _decisions = _reconstruct_antigravity(leaves)
            outcomes = {row.dataset: (row.baseline_confusion, row.candidate_confusion) for row in authority.rows if row.candidate_confusion is not None}
            evaluation = evaluate_routing_antigravity(authority, features, leave_one_dataset_out_antigravity(features, outcomes))
            for name, payload in _payloads_antigravity(evaluation).items():
                (root / name).write_bytes(payload)
            self._rehash_antigravity(root)
            with self.assertRaises(ValueError):
                _inspect_replay_with_dependencies_antigravity(root, expected_authority, expected_features)


if __name__ == "__main__":
    unittest.main()

