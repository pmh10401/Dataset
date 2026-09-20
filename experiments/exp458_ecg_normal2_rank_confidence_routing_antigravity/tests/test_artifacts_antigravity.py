from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_evaluation_antigravity import (
    DATASET_ANTIGRAVITY,
    exp456_authority_antigravity,
    normal2_decision_antigravity,
)
from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_router_antigravity import (
    qualification_antigravity,
)


def publication_evidence_antigravity():
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.contracts_antigravity import (
        PublicationEvidenceAntigravity,
    )
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.evaluation_antigravity import (
        evaluate_routes_antigravity,
    )

    authority = exp456_authority_antigravity()
    feature = {item.dataset: item for item in authority.features}[DATASET_ANTIGRAVITY]
    qualification = qualification_antigravity(True)
    decision = normal2_decision_antigravity(authority)
    result = evaluate_routes_antigravity(authority, decision)
    return PublicationEvidenceAntigravity(authority, qualification, feature, result)


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publishes_exact_canonical_nine_leaf_inventory_create_only_antigravity(self) -> None:
        """Break caught: publication inventory, canonical bytes, or create-only root safety drifts."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            _LEAVES_ANTIGRAVITY,
            publish_result_antigravity,
        )

        expected = (
            "context_antigravity.json",
            "train_qualification_antigravity.jsonl",
            "features_antigravity.jsonl",
            "route_decisions_antigravity.jsonl",
            "comparison_metrics_antigravity.json",
            "family_metrics_antigravity.json",
            "dashboard_snapshot_antigravity.json",
            "decision_antigravity.json",
            "artifact_sha256_antigravity.json",
        )
        self.assertEqual(_LEAVES_ANTIGRAVITY, expected)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            published = publish_result_antigravity(root, publication_evidence_antigravity())
            self.assertEqual(tuple(name for name, _digest in published.leaf_sha256), expected)
            self.assertEqual({entry.name for entry in root.iterdir()}, set(expected))
            artifact_map = json.loads((root / "artifact_sha256_antigravity.json").read_text(encoding="utf-8"))
            self.assertEqual(tuple(artifact_map), tuple(sorted(expected[:-1])))
            self.assertNotIn("artifact_sha256_antigravity.json", artifact_map)
            context = json.loads((root / "context_antigravity.json").read_text(encoding="utf-8"))
            self.assertEqual(
                context["runtime"],
                {
                    "aeon_version": "1.4.0",
                    "executable": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
                    "numpy_version": "2.3.5",
                    "prefix": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314",
                    "python_version": "3.14.6",
                },
            )
            self.assertTrue(all((root / name).read_bytes().endswith(b"\n") for name in expected))
            with self.assertRaises(ValueError):
                publish_result_antigravity(root, publication_evidence_antigravity())

    def test_rejects_symlink_root_wrong_feature_or_qualification_antigravity(self) -> None:
        """Break caught: publication accepts unsafe destinations or evidence not replayed from gates."""
        from dataclasses import replace
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )

        evidence = publication_evidence_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            target.mkdir()
            link = Path(temporary) / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                publish_result_antigravity(link, evidence)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            publish_result_antigravity(
                Path(temporary) / "result",
                replace(evidence, feature=replace(evidence.feature, added_count=99)),
            )
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            publish_result_antigravity(
                Path(temporary) / "result",
                replace(evidence, qualification=qualification_antigravity(False)),
            )


if __name__ == "__main__":
    unittest.main()
