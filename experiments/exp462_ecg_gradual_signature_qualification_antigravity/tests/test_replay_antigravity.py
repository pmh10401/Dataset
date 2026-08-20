from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp462_ecg_gradual_signature_qualification_antigravity.tests.test_artifacts_antigravity import (
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
    (root / LEAVES_ANTIGRAVITY[-1]).write_bytes(canonical_antigravity(artifact_map))


class ReplayAntigravityTest(unittest.TestCase):
    def test_inspect_recomputes_semantics_and_dashboard_without_sources_antigravity(self) -> None:
        """Break caught: replay trusts stored summaries, decision, or dashboard."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp462_task4_antigravity"
            publication = publish_result_antigravity(root, authority, qualification, decision)
            replay = inspect_replay_at_antigravity(root)
            self.assertEqual(replay.qualification, qualification)
            self.assertEqual(replay.decision, decision)
            self.assertEqual(replay.artifact_map_sha256, publication.artifact_map_sha256)

    def test_replay_rejects_inventory_noncanonical_and_rehashed_metric_tamper_antigravity(self) -> None:
        """Break caught: inventory or coherently rehashed metric tamper bypasses replay."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.replay_antigravity import (
            inspect_replay_at_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp462_task4_antigravity"
            publish_result_antigravity(root, authority, qualification, decision)
            rows = [
                json.loads(line)
                for line in (root / "method_fold_metrics_antigravity.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            rows[0]["per_type_auc"]["period_change"] = 0.5
            rows[0]["pooled_auc"] = sum(rows[0]["per_type_auc"].values()) / 4.0
            (root / "method_fold_metrics_antigravity.jsonl").write_bytes(
                b"".join(canonical_antigravity(row) for row in rows)
            )
            rehash_antigravity(root)
            with self.assertRaises(ValueError):
                inspect_replay_at_antigravity(root)

            (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                inspect_replay_at_antigravity(root)


if __name__ == "__main__":
    unittest.main()
