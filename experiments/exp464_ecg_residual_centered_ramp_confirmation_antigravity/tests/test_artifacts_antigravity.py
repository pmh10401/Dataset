from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity import (
    qualification_antigravity,
)


LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "method_fold_metrics_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


def publication_inputs_antigravity():
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
        evaluate_qualification_antigravity,
    )
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
        capture_exp463_authority_antigravity,
    )

    authority = capture_exp463_authority_antigravity()
    qualification = qualification_antigravity()
    decision = evaluate_qualification_antigravity(qualification, authority)
    return authority, qualification, decision


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publish_is_exact_seven_leaf_canonical_create_only_and_self_excluding_antigravity(self) -> None:
        """Break caught: publication changes inventory, order, or self-hash rules."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exp464_task4_antigravity"
            publication = publish_result_antigravity(root, qualification, decision, authority)
            self.assertEqual(tuple(path.name for path in sorted(root.iterdir())), tuple(sorted(LEAVES_ANTIGRAVITY)))
            self.assertEqual(tuple(name for name, _digest in publication.leaf_sha256), LEAVES_ANTIGRAVITY)
            artifact_map = json.loads((root / LEAVES_ANTIGRAVITY[-1]).read_text(encoding="utf-8"))
            self.assertEqual(set(artifact_map), set(LEAVES_ANTIGRAVITY[:-1]))
            for name in LEAVES_ANTIGRAVITY[:-1]:
                self.assertEqual(artifact_map[name], hashlib.sha256((root / name).read_bytes()).hexdigest())
            self.assertEqual(
                len((root / "method_fold_metrics_antigravity.jsonl").read_text(encoding="utf-8").splitlines()),
                90,
            )
            with self.assertRaises(ValueError):
                publish_result_antigravity(root, qualification, decision, authority)


if __name__ == "__main__":
    unittest.main()
