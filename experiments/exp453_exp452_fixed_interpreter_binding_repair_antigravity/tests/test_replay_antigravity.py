"""Red-first exact-order replay fixture for the Exp453 controlled fork."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReplayAntigravityTests(unittest.TestCase):
    def _publication_antigravity(self, temporary: str):
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.tests.test_evaluation_antigravity import EvaluationAntigravityTests
        root = Path(temporary) / "publication_antigravity"
        dependencies = EvaluationAntigravityTests()._dependencies_antigravity(root)
        return _evaluate_with_dependencies_antigravity("a" * 40, dependencies), root, dependencies

    def _rehash_antigravity(self, root: Path, *names: str) -> None:
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        hashes = json.loads((root / "artifact_sha256_antigravity.json").read_text())
        for name in names:
            hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        (root / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity(hashes))

    def test_exact_replay_contract_requires_ordered_integer_lists_antigravity(self) -> None:
        """Break caught: set-equivalent reordered evidence is not a valid frozen prediction."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.replay_antigravity import inspect_replay_at_antigravity
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.tests.test_evaluation_antigravity import EvaluationAntigravityTests

        with tempfile.TemporaryDirectory(prefix="exp453_replay_antigravity_") as temporary:
            root = Path(temporary) / "publication_antigravity"
            dependencies = EvaluationAntigravityTests()._dependencies_antigravity(root)
            _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            payload = json.loads((root / "evaluation_antigravity.json").read_text())
            candidate = next(iter(payload["dataset_evidence"][0]["candidates"].values()))
            candidate["indices"] = list(reversed(candidate["indices"]))
            from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
            path = root / "evaluation_antigravity.json"
            path.write_bytes(canonical_json_bytes_antigravity(payload))
            hashes = json.loads((root / "artifact_sha256_antigravity.json").read_text())
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "artifact_sha256_antigravity.json").write_bytes(canonical_json_bytes_antigravity(hashes))
            with self.assertRaisesRegex(ValueError, "frozen prediction indices differ from evidence"):
                inspect_replay_at_antigravity(root, expected_prediction_authority=dependencies.capture_task5_prediction_antigravity(), expected_reviewed_commit="a" * 40)

    def test_rehashed_index_evidence_rejects_all_exact_list_violations_antigravity(self) -> None:
        """Break caught: replay accepts set-equivalent or malformed evidence after artifact-map rehash."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.replay_antigravity import inspect_replay_at_antigravity

        for case in ("missing", "extra", "duplicate", "reordered", "bool", "bool_expected", "float_expected", "wrong_type", "out_of_range"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp453_rehashed_indices_antigravity_") as temporary:
                _expected, root, dependencies = self._publication_antigravity(temporary)
                path = root / "evaluation_antigravity.json"
                payload = json.loads(path.read_text())
                candidate = next(iter(payload["dataset_evidence"][0]["candidates"].values()))
                indices = list(candidate["indices"])
                if case == "missing":
                    candidate["indices"] = indices[1:]
                elif case == "extra":
                    candidate["indices"] = [*indices, 2]
                elif case == "duplicate":
                    candidate["indices"] = [indices[0], indices[0]]
                elif case == "reordered":
                    candidate["indices"] = list(reversed(indices))
                elif case == "bool":
                    candidate["indices"] = [True, indices[1]]
                elif case == "bool_expected":
                    candidate["indices"] = [indices[0], True]
                elif case == "float_expected":
                    candidate["indices"] = [indices[0], 1.0]
                elif case == "wrong_type":
                    candidate["indices"] = ["0", indices[1]]
                else:
                    candidate["indices"] = [payload["dataset_evidence"][0]["target_length"], indices[1]]
                path.write_bytes(canonical_json_bytes_antigravity(payload))
                self._rehash_antigravity(root, path.name)
                with self.assertRaises(ValueError):
                    inspect_replay_at_antigravity(root, expected_prediction_authority=dependencies.capture_task5_prediction_antigravity(), expected_reviewed_commit="a" * 40)

    def test_replay_rejects_rehashed_confusion_bootstrap_decision_and_partial_root_antigravity(self) -> None:
        """Break caught: a self-consistent semantic leaf, bootstrap, decision, or partial root can be forged."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.replay_antigravity import inspect_replay_at_antigravity

        for case in ("confusion", "bootstrap", "decision", "artifact_map", "partial"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp453_replay_tamper_antigravity_") as temporary:
                _expected, root, dependencies = self._publication_antigravity(temporary)
                capture = dependencies.capture_task5_prediction_antigravity()
                if case == "confusion":
                    path = root / "candidate_metrics_antigravity.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows[0]["target_confusion"]["fn"] += 1
                    path.write_bytes(b"".join(canonical_json_bytes_antigravity(row) for row in rows))
                    self._rehash_antigravity(root, path.name)
                elif case == "bootstrap":
                    path = root / "bootstrap_antigravity.json"
                    payload = json.loads(path.read_text())
                    payload["draw_sha256"] = "0" * 64
                    path.write_bytes(canonical_json_bytes_antigravity(payload))
                    self._rehash_antigravity(root, path.name)
                elif case == "decision":
                    path = root / "decision_antigravity.json"
                    payload = json.loads(path.read_text())
                    payload["qualified_count"] += 1
                    path.write_bytes(canonical_json_bytes_antigravity(payload))
                    self._rehash_antigravity(root, path.name)
                elif case == "artifact_map":
                    path = root / "artifact_sha256_antigravity.json"
                    payload = json.loads(path.read_text())
                    payload["context_antigravity.json"] = "0" * 64
                    path.write_bytes(canonical_json_bytes_antigravity(payload))
                else:
                    (root / "decision_antigravity.json").unlink()
                with self.assertRaises(ValueError):
                    inspect_replay_at_antigravity(root, expected_prediction_authority=capture, expected_reviewed_commit="a" * 40)

    def test_replay_requires_exact_exp453_evaluation_schema_antigravity(self) -> None:
        """Break caught: a rehashed predecessor schema is accepted as an Exp453 evaluation."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.replay_antigravity import inspect_replay_at_antigravity
        with tempfile.TemporaryDirectory(prefix="exp453_schema_antigravity_") as temporary:
            _expected, root, dependencies = self._publication_antigravity(temporary)
            path = root / "evaluation_antigravity.json"
            payload = json.loads(path.read_text())
            payload["schema"] = "exp452"
            path.write_bytes(canonical_json_bytes_antigravity(payload))
            self._rehash_antigravity(root, path.name)
            with self.assertRaisesRegex(ValueError, "evaluation schema differs"):
                inspect_replay_at_antigravity(root, expected_prediction_authority=dependencies.capture_task5_prediction_antigravity(), expected_reviewed_commit="a" * 40)

    def test_source_free_replay_import_firewall_subprocess_antigravity(self) -> None:
        """Break caught: importing replay reaches labels, SQLite, network, or dashboard publication."""
        code = (
            "import sys; import experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.replay_antigravity; "
            "blocked=('sqlite3','experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.sources_antigravity','requests','urllib','httpx'); "
            "assert not [name for name in blocked if name in sys.modules]"
        )
        completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
        self.assertEqual((completed.returncode, completed.stdout, completed.stderr), (0, "", ""))


if __name__ == "__main__":
    unittest.main()
