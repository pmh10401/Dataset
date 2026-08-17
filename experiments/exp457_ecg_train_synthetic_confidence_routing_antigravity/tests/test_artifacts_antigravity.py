from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.tests.test_evaluation_antigravity import (
    decisions_antigravity,
    exp456_authority_antigravity,
)
from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.tests.test_router_antigravity import (
    qualification_antigravity,
)


def publication_evidence_antigravity():
    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.contracts_antigravity import (
        PublicationEvidenceAntigravity,
    )
    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
        evaluate_routes_antigravity,
    )

    authority = exp456_authority_antigravity()
    datasets = tuple(f"ECG5000_normal_{index}" for index in range(1, 6))
    features_by_dataset = {feature.dataset: feature for feature in authority.features}
    features = tuple(features_by_dataset[dataset] for dataset in datasets)
    qualifications = tuple(
        qualification_antigravity(dataset, dataset == "ECG5000_normal_2")
        for dataset in datasets
    )
    decisions = decisions_antigravity(authority)
    result = evaluate_routes_antigravity(authority, decisions)
    return PublicationEvidenceAntigravity(
        authority, qualifications, features, result
    )


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publishes_exact_ordered_canonical_nine_leaf_inventory_create_only_antigravity(self) -> None:
        """Break caught: publication omits evidence, changes order/encoding, or overwrites a root."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.artifacts_antigravity import (
            _LEAVES_ANTIGRAVITY,
            publish_result_antigravity,
        )

        expected = (
            "context_antigravity.json",
            "train_qualification_antigravity.json",
            "features_antigravity.jsonl",
            "route_decisions_antigravity.jsonl",
            "comparison_metrics_antigravity.json",
            "family_metrics_antigravity.jsonl",
            "dashboard_snapshot_antigravity.json",
            "decision_antigravity.json",
            "artifact_sha256_antigravity.json",
        )
        self.assertEqual(_LEAVES_ANTIGRAVITY, expected)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            published = publish_result_antigravity(root, publication_evidence_antigravity())
            self.assertEqual({entry.name for entry in root.iterdir()}, set(expected))
            self.assertEqual(tuple(name for name, _digest in published.leaf_sha256), expected)
            artifact_map = json.loads(
                (root / "artifact_sha256_antigravity.json").read_text(encoding="utf-8")
            )
            self.assertEqual(tuple(artifact_map), tuple(sorted(expected[:-1])))
            self.assertNotIn("artifact_sha256_antigravity.json", artifact_map)
            self.assertTrue(
                all(
                    (root / name).read_bytes().endswith(b"\n")
                    for name in expected
                )
            )
            with self.assertRaises(ValueError):
                publish_result_antigravity(root, publication_evidence_antigravity())

    def test_rejects_evidence_not_recomputed_from_qualification_and_feature_antigravity(self) -> None:
        """Break caught: stored routes are accepted without rerunning both gates."""
        from dataclasses import replace
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )

        evidence = publication_evidence_antigravity()
        forged = replace(
            evidence,
            qualifications=(replace(evidence.qualifications[0], passed=True),)
            + evidence.qualifications[1:],
        )
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            publish_result_antigravity(Path(temporary) / "result", forged)


if __name__ == "__main__":
    unittest.main()
