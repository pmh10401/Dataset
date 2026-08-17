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

    def test_public_replay_uses_one_injected_capture_and_feature_build_antigravity(self) -> None:
        """Break caught: public replay touches a real source or captures/builds frozen inputs more than once."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import publish_evaluation_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity import replay_antigravity

        authority, features, decisions = self._inputs_antigravity()
        expected = evaluate_routing_antigravity(authority, features, decisions)
        calls = {"capture": 0, "build": 0}
        original_capture, original_build = replay_antigravity.capture_routing_authority_antigravity, replay_antigravity.build_label_free_features_antigravity
        def capture_antigravity():
            calls["capture"] += 1
            return authority
        def build_antigravity(received):
            calls["build"] += 1
            self.assertIs(received, authority)
            return features
        try:
            replay_antigravity.capture_routing_authority_antigravity = capture_antigravity
            replay_antigravity.build_label_free_features_antigravity = build_antigravity
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result"
                publish_evaluation_antigravity(root, expected)
                self.assertEqual(replay_antigravity.inspect_replay_at_antigravity(root), expected)
        finally:
            replay_antigravity.capture_routing_authority_antigravity = original_capture
            replay_antigravity.build_label_free_features_antigravity = original_build
        self.assertEqual(calls, {"capture": 1, "build": 1})

    def test_replay_rejects_rehashed_dataset_identity_changes_antigravity(self) -> None:
        """Break caught: a fully regenerated semantic artifact can replace a bound dataset key, family, method, or length."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import _payloads_antigravity, publish_evaluation_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.replay_antigravity import _inspect_replay_with_dependencies_antigravity, _reconstruct_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import leave_one_dataset_out_antigravity

        def rewrite_antigravity(root: Path) -> None:
            leaves = {entry.name: entry.read_bytes() for entry in root.iterdir()}
            authority, features, _decisions = _reconstruct_antigravity(leaves)
            outcomes = {row.dataset: (row.baseline_confusion, row.candidate_confusion) for row in authority.rows if row.candidate_confusion is not None}
            for name, payload in _payloads_antigravity(evaluate_routing_antigravity(authority, features, leave_one_dataset_out_antigravity(features, outcomes))).items():
                (root / name).write_bytes(payload)
            self._rehash_antigravity(root)

        def mutate_dataset_antigravity(root: Path) -> None:
            context = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8")); features = [json.loads(line) for line in (root / "features_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]; decisions = [json.loads(line) for line in (root / "fold_decisions_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]
            old, new = features[0]["dataset"], "bound_dataset_replacement"
            next(row for row in context["rows"] if row["dataset"] == old)["dataset"] = new; features[0]["dataset"] = new; next(row for row in decisions if row["dataset"] == old)["dataset"] = new
            (root / "context_antigravity.json").write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n"); (root / "features_antigravity.jsonl").write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in features)); (root / "fold_decisions_antigravity.jsonl").write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in decisions))

        def mutate_family_antigravity(root: Path) -> None:
            context = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8")); features = [json.loads(line) for line in (root / "features_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]; decisions = [json.loads(line) for line in (root / "fold_decisions_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]
            first, second = features[0]["dataset"], features[-1]["dataset"]
            for rows in (context["rows"], features, decisions):
                left, right = next(row for row in rows if row["dataset"] == first), next(row for row in rows if row["dataset"] == second)
                left["family"], right["family"] = right["family"], left["family"]
            for rows in (context["rows"], decisions):
                left, right = next(row for row in rows if row["dataset"] == first), next(row for row in rows if row["dataset"] == second)
                left["method"], right["method"] = right["method"], left["method"]
            (root / "context_antigravity.json").write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n"); (root / "features_antigravity.jsonl").write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in features)); (root / "fold_decisions_antigravity.jsonl").write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in decisions))

        def mutate_method_antigravity(root: Path) -> None:
            context = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8")); decisions = [json.loads(line) for line in (root / "fold_decisions_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]
            dataset = decisions[0]["dataset"]
            next(row for row in context["rows"] if row["dataset"] == dataset)["method"] = "stored_method_replacement"
            decisions[0]["method"] = "stored_method_replacement"
            (root / "context_antigravity.json").write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n"); (root / "fold_decisions_antigravity.jsonl").write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in decisions))

        def mutate_length_antigravity(root: Path) -> None:
            context = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8")); features = [json.loads(line) for line in (root / "features_antigravity.jsonl").read_text(encoding="utf-8").splitlines()]; dataset = features[0]["dataset"]
            next(row for row in context["rows"] if row["dataset"] == dataset)["target_length"] += 1; features[0]["target_length"] += 1
            (root / "context_antigravity.json").write_bytes(json.dumps(context, sort_keys=True, separators=(",", ":")).encode() + b"\n"); (root / "features_antigravity.jsonl").write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in features))

        expected_authority, expected_features, expected_decisions = self._inputs_antigravity()
        mutations = (("dataset", mutate_dataset_antigravity, True), ("family", mutate_family_antigravity, True), ("method", mutate_method_antigravity, False), ("target_length", mutate_length_antigravity, True))
        for name, mutate, rebuild in mutations:
            with self.subTest(change=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result"
                publish_evaluation_antigravity(root, evaluate_routing_antigravity(expected_authority, expected_features, expected_decisions))
                mutate(root)
                if rebuild:
                    rewrite_antigravity(root)
                else:
                    self._rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    _inspect_replay_with_dependencies_antigravity(root, expected_authority, expected_features)


if __name__ == "__main__":
    unittest.main()
