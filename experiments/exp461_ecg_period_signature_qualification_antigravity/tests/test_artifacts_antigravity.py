from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp461_ecg_period_signature_qualification_antigravity.tests.test_qualification_antigravity import (
    qualification_antigravity,
)


LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "method_fold_metrics_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.json",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


def publication_inputs_antigravity():
    from experiments.exp461_ecg_period_signature_qualification_antigravity.evaluation_antigravity import (
        evaluate_qualification_antigravity,
    )
    from experiments.exp461_ecg_period_signature_qualification_antigravity.predecessor_antigravity import (
        capture_exp460_period_authority_antigravity,
    )

    authority = capture_exp460_period_authority_antigravity()
    qualification = qualification_antigravity()
    decision = evaluate_qualification_antigravity(authority, qualification)
    return authority, qualification, decision


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publish_writes_exact_seven_leaf_create_only_canonical_inventory_antigravity(self) -> None:
        """Break caught: publication changes leaf count/order, hashes itself, or overwrites a root."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp461_task4_antigravity"
            publication = publish_result_antigravity(root, authority, qualification, decision)

            self.assertEqual(tuple(path.name for path in sorted(root.iterdir())), tuple(sorted(LEAVES_ANTIGRAVITY)))
            self.assertEqual(tuple(name for name, _digest in publication.leaf_sha256), LEAVES_ANTIGRAVITY)
            artifact_map = json.loads((root / "artifact_sha256_antigravity.json").read_text(encoding="utf-8"))
            self.assertEqual(set(artifact_map), set(LEAVES_ANTIGRAVITY[:-1]))
            self.assertNotIn("artifact_sha256_antigravity.json", artifact_map)
            for name in LEAVES_ANTIGRAVITY[:-1]:
                self.assertEqual(artifact_map[name], hashlib.sha256((root / name).read_bytes()).hexdigest())
            self.assertEqual(
                publication.artifact_map_sha256,
                hashlib.sha256((root / "artifact_sha256_antigravity.json").read_bytes()).hexdigest(),
            )
            with self.assertRaises(ValueError):
                publish_result_antigravity(root, authority, qualification, decision)


if __name__ == "__main__":
    unittest.main()
