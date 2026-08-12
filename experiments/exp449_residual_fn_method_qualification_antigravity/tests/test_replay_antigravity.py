"""Synthetic-only source-free replay boundaries for Task 6."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity, canonical_jsonl_bytes_antigravity


class ReplayAntigravityTests(unittest.TestCase):
    def _publication_antigravity(self, directory: str):
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity
        dependencies = _DependenciesAntigravity(Path(directory) / "publication_antigravity")
        return _evaluate_with_dependencies_antigravity("a" * 40, dependencies), dependencies.root

    def test_replay_matches_evaluation_without_source_modules_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.replay_antigravity import inspect_replay_at_antigravity

        with tempfile.TemporaryDirectory(prefix="exp449_replay_antigravity_") as temporary:
            expected, root = self._publication_antigravity(temporary)
            self.assertEqual(inspect_replay_at_antigravity(root), expected)

    def test_replay_rejects_rehashed_semantic_and_canonical_attacks_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.replay_antigravity import inspect_replay_at_antigravity

        for attack in ("duplicate", "noncanonical", "reordered", "bool_order", "wrong_baseline", "forged_delta", "forged_global"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory(prefix="exp449_replay_attack_antigravity_") as temporary:
                _expected, root = self._publication_antigravity(temporary)
                path = root / "candidate_metrics_antigravity.jsonl"
                if attack == "duplicate":
                    path.write_bytes(b'{"family":"Crop","family":"Crop"}\n')
                elif attack == "noncanonical":
                    path.write_bytes(path.read_bytes().replace(b'{', b'{ ', 1))
                else:
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if attack == "reordered": rows[:2] = rows[1::-1]
                    elif attack == "bool_order": rows[0]["dataset_order"] = True
                    elif attack == "wrong_baseline": rows[0]["target_confusion"]["fn"] += 1
                    elif attack == "forged_delta": rows[0]["target_fn_delta"] = -999
                    else: rows[0]["global_confusion"]["fp"] += 1
                    path.write_bytes(canonical_jsonl_bytes_antigravity(rows))
                hashes = json.loads((root / "artifact_sha256_antigravity.json").read_text())
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity(hashes))
                with self.assertRaises(ValueError): inspect_replay_at_antigravity(root)

    def test_replay_rejects_extra_and_symlink_leaf_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.replay_antigravity import inspect_replay_at_antigravity

        with tempfile.TemporaryDirectory(prefix="exp449_replay_namespace_antigravity_") as temporary:
            _expected, root = self._publication_antigravity(temporary)
            (root / "extra_antigravity.json").write_bytes(b"{}\n")
            with self.assertRaises(ValueError): inspect_replay_at_antigravity(root)

    def test_replay_rejects_rehashed_prediction_and_bootstrap_content_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.replay_antigravity import inspect_replay_at_antigravity

        for leaf, mutate in (("frozen_predictions_antigravity.jsonl", "prediction"), ("bootstrap_antigravity.json", "bootstrap")):
            with self.subTest(leaf=leaf), tempfile.TemporaryDirectory(prefix="exp449_replay_bound_content_antigravity_") as temporary:
                _expected, root = self._publication_antigravity(temporary)
                path = root / leaf
                if mutate == "prediction":
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["candidate_name"] = "forged_antigravity"
                    path.write_bytes(canonical_jsonl_bytes_antigravity(rows))
                    manifest_path = root / "prediction_manifest_antigravity.json"
                    manifest = json.loads(manifest_path.read_text()); manifest["frozen_prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    manifest_path.write_bytes(canonical_json_bytes_antigravity(manifest))
                else:
                    bootstrap = json.loads(path.read_text()); bootstrap["comparisons"][0]["fn_delta"] = 999
                    path.write_bytes(canonical_json_bytes_antigravity(bootstrap))
                hashes = json.loads((root / "artifact_sha256_antigravity.json").read_text())
                for changed in (path, *( (root / "prediction_manifest_antigravity.json",) if mutate == "prediction" else ())):
                    hashes[changed.name] = hashlib.sha256(changed.read_bytes()).hexdigest()
                (root / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity(hashes))
                with self.assertRaises(ValueError): inspect_replay_at_antigravity(root)
        with tempfile.TemporaryDirectory(prefix="exp449_replay_symlink_antigravity_") as temporary:
            _expected, root = self._publication_antigravity(temporary)
            target = root / "context_antigravity.json"; replacement = root / "replacement_antigravity.json"
            replacement.write_bytes(target.read_bytes()); target.unlink(); target.symlink_to(replacement.name)
            with self.assertRaises(ValueError): inspect_replay_at_antigravity(root)


if __name__ == "__main__":
    unittest.main()
