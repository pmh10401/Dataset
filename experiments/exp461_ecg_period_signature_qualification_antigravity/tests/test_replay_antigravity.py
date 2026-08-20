from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp461_ecg_period_signature_qualification_antigravity.tests.test_artifacts_antigravity import (
    LEAVES_ANTIGRAVITY,
    publication_inputs_antigravity,
)


def canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def rehash_antigravity(root: Path) -> None:
    artifact_map = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in LEAVES_ANTIGRAVITY[:-1]
    }
    (root / "artifact_sha256_antigravity.json").write_bytes(canonical_antigravity(artifact_map))


class ReplayAntigravityTest(unittest.TestCase):
    def test_inspect_recomputes_qualification_decision_dashboard_and_hashes_antigravity(self) -> None:
        """Break caught: replay trusts stored summaries, decisions, dashboards, or hash maps."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp461_task4_antigravity"
            publication = publish_result_antigravity(root, authority, qualification, decision)
            replay = inspect_replay_at_antigravity(root)

            self.assertEqual(replay.qualification, qualification)
            self.assertEqual(replay.decision, decision)
            self.assertEqual(replay.artifact_map_sha256, publication.artifact_map_sha256)

    def test_replay_rejects_inventory_canonical_and_rehashed_semantic_tamper_antigravity(self) -> None:
        """Break caught: replay accepts missing leaves, noncanonical JSON, or rehashed result tampering."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        def mutate_json(root: Path, name: str, mutate) -> None:
            path = root / name
            value = json.loads(path.read_text(encoding="utf-8"))
            mutate(value)
            path.write_bytes(canonical_antigravity(value))

        def mutate_jsonl(root: Path, name: str, mutate) -> None:
            path = root / name
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            mutate(rows)
            path.write_bytes(b"".join(canonical_antigravity(row) for row in rows))

        mutations = (
            lambda root: (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8"),
            lambda root: (root / "method_summary_antigravity.json").write_text('{"summary": true}\n', encoding="utf-8"),
            lambda root: mutate_jsonl(
                root,
                "method_fold_metrics_antigravity.jsonl",
                lambda rows: rows[0].__setitem__("method", "hydra_period_rank_max"),
            ),
            lambda root: mutate_jsonl(
                root,
                "method_fold_metrics_antigravity.jsonl",
                lambda rows: rows[-1]["per_type_auc"].__setitem__("period_change", 0.5),
            ),
            lambda root: mutate_json(
                root,
                "decision_antigravity.json",
                lambda value: value.__setitem__("eligible_for_routing_research", False),
            ),
            lambda root: mutate_json(
                root,
                "dashboard_snapshot_antigravity.json",
                lambda value: value.__setitem__("primary_status", "tampered"),
            ),
        )
        authority, qualification, decision = publication_inputs_antigravity()
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "exp461_task4_antigravity"
                publish_result_antigravity(root, authority, qualification, decision)
                mutate(root)
                if (root / "artifact_sha256_antigravity.json").exists() and not (root / "extra_antigravity.json").exists():
                    rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    inspect_replay_at_antigravity(root)


if __name__ == "__main__":
    unittest.main()
