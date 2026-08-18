from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_artifacts_antigravity import (
    decision_antigravity,
)


def canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def rehash_antigravity(root: Path) -> None:
    names = (
        "context_antigravity.json",
        "fold_plan_antigravity.jsonl",
        "fold_evidence_antigravity.jsonl",
        "seed_summary_antigravity.jsonl",
        "decision_antigravity.json",
        "dashboard_snapshot_antigravity.json",
    )
    artifact_map = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    (root / "artifact_sha256_antigravity.json").write_bytes(canonical_antigravity(artifact_map))


class ReplayAntigravityTest(unittest.TestCase):
    def test_source_free_replay_recomputes_plan_evidence_decision_and_dashboard_antigravity(self) -> None:
        """Break caught: replay trusts stored leaves instead of rebuilding the result from contracts."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.replay_antigravity import (
            inspect_result_at_antigravity,
        )

        decision = decision_antigravity(unstable=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result_antigravity"
            publish_result_antigravity(root, decision)
            replayed = inspect_result_at_antigravity(root)
            self.assertEqual(replayed.dataset, decision.dataset)
            self.assertEqual(replayed.status, decision.status)
            self.assertEqual(replayed.passed, decision.passed)
            self.assertEqual(
                [(record.seed, record.fold_index) for record in replayed.qualification.fold_evidence],
                [(record.seed, record.fold_index) for record in decision.qualification.fold_evidence],
            )
            self.assertEqual(replayed.status, "unstable")

    def test_rejects_inventory_hash_canonical_duplicate_bool_and_reordered_rows_antigravity(self) -> None:
        """Break caught: byte validation misses extra/missing leaves, duplicate keys, bool/int drift, or row reorder."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.replay_antigravity import (
            inspect_result_at_antigravity,
        )

        mutations = (
            lambda root: (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8"),
            lambda root: (root / "fold_plan_antigravity.jsonl").unlink(),
            lambda root: (root / "decision_antigravity.json").write_text('{"passed":true,"passed":false,"status":"stable"}\n', encoding="utf-8"),
            lambda root: (root / "decision_antigravity.json").write_text('{"dataset":"ECG5000_normal_2","passed":1,"status":"stable"}\n', encoding="utf-8"),
            lambda root: self._reverse_jsonl_antigravity(root, "fold_plan_antigravity.jsonl"),
            lambda root: self._reverse_jsonl_antigravity(root, "fold_evidence_antigravity.jsonl"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result_antigravity"
                publish_result_antigravity(root, decision_antigravity())
                mutate(root)
                if (root / "artifact_sha256_antigravity.json").exists() and not (root / "extra_antigravity.json").exists():
                    try:
                        rehash_antigravity(root)
                    except FileNotFoundError:
                        pass
                with self.assertRaises(ValueError):
                    inspect_result_at_antigravity(root)

    def test_rejects_rehashed_semantic_tamper_across_result_leaves_antigravity(self) -> None:
        """Break caught: replay accepts rehashed plan, AUC, Wilson, pass, dashboard, or map tamper."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.replay_antigravity import (
            inspect_result_at_antigravity,
        )

        mutations = (
            lambda root: self._mutate_jsonl_antigravity(root, "fold_plan_antigravity.jsonl", lambda rows: rows[0].__setitem__("query_count", 284)),
            lambda root: self._mutate_jsonl_antigravity(root, "fold_evidence_antigravity.jsonl", lambda rows: rows[0]["per_type_auc"].__setitem__("level_shift", 0.86)),
            lambda root: self._mutate_jsonl_antigravity(root, "fold_evidence_antigravity.jsonl", lambda rows: rows[0].__setitem__("pooled_auc", 0.91)),
            lambda root: self._mutate_jsonl_antigravity(root, "fold_evidence_antigravity.jsonl", lambda rows: rows[0].__setitem__("normal_wilson_upper", 0.0)),
            lambda root: self._mutate_jsonl_antigravity(root, "fold_evidence_antigravity.jsonl", lambda rows: rows[0].__setitem__("passed", False)),
            lambda root: self._mutate_jsonl_antigravity(root, "seed_summary_antigravity.jsonl", lambda rows: rows[0].__setitem__("passed", False)),
            lambda root: self._mutate_json_antigravity(root, "decision_antigravity.json", lambda value: value.__setitem__("status", "unstable")),
            lambda root: self._mutate_json_antigravity(root, "dashboard_snapshot_antigravity.json", lambda value: value.__setitem__("label_query_count", 1)),
            lambda root: self._mutate_json_antigravity(root, "artifact_sha256_antigravity.json", lambda value: value.__setitem__("context_antigravity.json", "0" * 64)),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result_antigravity"
                publish_result_antigravity(root, decision_antigravity())
                mutate(root)
                if index != len(mutations) - 1:
                    rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    inspect_result_at_antigravity(root)

    @staticmethod
    def _reverse_jsonl_antigravity(root: Path, name: str) -> None:
        path = root / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_bytes(b"".join(canonical_antigravity(row) for row in reversed(rows)))

    @staticmethod
    def _mutate_json_antigravity(root: Path, name: str, mutate) -> None:
        path = root / name
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_bytes(canonical_antigravity(value))

    @staticmethod
    def _mutate_jsonl_antigravity(root: Path, name: str, mutate) -> None:
        path = root / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        path.write_bytes(b"".join(canonical_antigravity(row) for row in rows))


if __name__ == "__main__":
    unittest.main()
