from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_evaluation_antigravity import (
    qualification_antigravity,
)


def decision_antigravity(*, unstable: bool = False):
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
        evaluate_blocked_qualification_antigravity,
    )

    return evaluate_blocked_qualification_antigravity(
        qualification_antigravity(failing_seed=20261721 if unstable else None)
    )


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publishes_exact_canonical_seven_leaf_inventory_create_only_antigravity(self) -> None:
        """Break caught: publication writes the wrong leaves, noncanonical bytes, or overwrites a root."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import (
            _LEAVES_ANTIGRAVITY,
            publish_result_antigravity,
        )

        expected = (
            "context_antigravity.json",
            "fold_plan_antigravity.jsonl",
            "fold_evidence_antigravity.jsonl",
            "seed_summary_antigravity.jsonl",
            "decision_antigravity.json",
            "dashboard_snapshot_antigravity.json",
            "artifact_sha256_antigravity.json",
        )
        self.assertEqual(_LEAVES_ANTIGRAVITY, expected)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result_antigravity"
            published = publish_result_antigravity(root, decision_antigravity())
            self.assertEqual(tuple(name for name, _digest in published.leaf_sha256), expected)
            self.assertEqual({entry.name for entry in root.iterdir()}, set(expected))
            self.assertTrue(all((root / name).read_bytes().endswith(b"\n") for name in expected))

            artifact_map = json.loads((root / "artifact_sha256_antigravity.json").read_text(encoding="utf-8"))
            self.assertEqual(tuple(artifact_map), tuple(sorted(expected[:-1])))
            self.assertNotIn("artifact_sha256_antigravity.json", artifact_map)
            self.assertEqual(len((root / "fold_plan_antigravity.jsonl").read_text(encoding="utf-8").splitlines()), 5)
            self.assertEqual(len((root / "fold_evidence_antigravity.jsonl").read_text(encoding="utf-8").splitlines()), 10)
            self.assertEqual(len((root / "seed_summary_antigravity.jsonl").read_text(encoding="utf-8").splitlines()), 2)

            context = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8"))
            self.assertEqual(context["experiment_id"], "Exp459")
            self.assertEqual(context["label_query_count"], 0)
            self.assertEqual(
                context["exp458_predecessor"],
                {"experiment": "Exp458", "f1": 0.5796148426491311, "fn": 829, "fp": 66, "tn": 70788, "tp": 617},
            )
            self.assertNotIn("/Users/mac/Documents/Dataset/univariate_ts.db", repr(context))

            with self.assertRaises(ValueError):
                publish_result_antigravity(root, decision_antigravity())

    def test_rejects_symlink_destination_and_forged_decision_without_shared_evaluator_antigravity(self) -> None:
        """Break caught: publication trusts unsafe paths or caller-minted decisions."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            ValidationDecisionAntigravity,
        )

        decision = decision_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target_antigravity"
            target.mkdir()
            link = Path(temporary) / "link_antigravity"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                publish_result_antigravity(link, decision)

        forged = ValidationDecisionAntigravity(
            dataset=decision.dataset,
            qualification=decision.qualification,
            status=decision.status,
            passed=decision.passed,
        )
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            publish_result_antigravity(Path(temporary) / "result_antigravity", forged)


if __name__ == "__main__":
    unittest.main()
