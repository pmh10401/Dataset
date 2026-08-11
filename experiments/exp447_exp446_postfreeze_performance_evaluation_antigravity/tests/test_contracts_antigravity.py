"""Immutable, exact-type Exp447 contracts."""
from __future__ import annotations

from types import MappingProxyType
import unittest


class ContractAntigravityTests(unittest.TestCase):
    def test_confusion_rejects_bool_and_computes_project_metrics_antigravity(self) -> None:
        # Break caught: bool is counted as an integer or F1 uses a non-project denominator.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
        )

        confusion = ConfusionAntigravity(tp=1, fp=1, fn=1, tn=7)
        self.assertEqual(confusion.precision, 0.5)
        self.assertEqual(confusion.recall, 0.5)
        self.assertEqual(confusion.f1, 0.5)
        self.assertEqual(confusion.false_positive_rate, 0.125)
        self.assertEqual(confusion.to_payload_antigravity(), {"fn": 1, "fp": 1, "tn": 7, "tp": 1})
        with self.assertRaises(ValueError):
            ConfusionAntigravity(tp=True, fp=0, fn=0, tn=0)

    def test_metric_summary_requires_exact_finite_mapping_antigravity(self) -> None:
        # Break caught: a malformed aggregate metric is serialised as scientific evidence.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
            MetricSummaryAntigravity,
        )

        summary = MetricSummaryAntigravity(
            comparison_name="hard_baseline",
            confusion=ConfusionAntigravity(tp=1, fp=0, fn=1, tn=8),
            macro_f1=0.5,
        )
        self.assertEqual(
            summary.to_payload_antigravity(),
            {
                "comparison_name": "hard_baseline", "tp": 1, "fp": 0, "fn": 1, "tn": 8,
                "precision": 1.0, "recall": 0.5, "f1": 2.0 / 3.0,
                "false_positive_rate": 0.0, "macro_f1": 0.5,
            },
        )
        with self.assertRaises(ValueError):
            MetricSummaryAntigravity("hard_baseline", ConfusionAntigravity(0, 0, 0, 0), float("nan"))

    def test_publication_identity_deep_freezes_and_conserves_lanes_antigravity(self) -> None:
        # Break caught: a source leaf or lane partition can be modified after validation.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            PublicationIdentityAntigravity,
        )

        digest = "a" * 64
        leaves = {"prediction_antigravity.json": digest}
        identity = PublicationIdentityAntigravity(
            root_device=1,
            root_inode=2,
            leaf_sha256=leaves,
            manifest_sha256=digest,
            prediction_sha256=digest,
            artifact_map_sha256=digest,
            dataset_order=("A", "B"),
            source_coordinates=(("A", 1), ("B", 2), ("B", 3)),
            stable_coordinates=(("A", 1), ("B", 2)),
            unstable_coordinates=(("B", 3),),
            label_access_count=0,
        )
        leaves["forged"] = digest
        self.assertIsInstance(identity.leaf_sha256, MappingProxyType)
        self.assertEqual(tuple(identity.leaf_sha256), ("prediction_antigravity.json",))
        self.assertEqual(identity.source_count, 3)
        self.assertEqual(identity.stable_count + identity.unstable_count, identity.source_count)
        with self.assertRaises(TypeError):
            identity.leaf_sha256["forged"] = digest  # type: ignore[index]
        with self.assertRaises(ValueError):
            PublicationIdentityAntigravity(
                1, 2, {"leaf": digest}, digest, digest, digest, ("A",),
                (("A", 1),), (("A", 1),), (("A", 1),), 0,
            )

    def test_prelabel_payload_recursively_rejects_label_like_keys_antigravity(self) -> None:
        # Break caught: pre-label source evidence carries a hidden TEST-label field.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            freeze_prelabel_payload_antigravity,
        )

        frozen = freeze_prelabel_payload_antigravity({"source": {"candidate_count": 30}, "lanes": ["stable_review"]})
        self.assertIsInstance(frozen, MappingProxyType)
        with self.assertRaises(ValueError):
            freeze_prelabel_payload_antigravity({"source": {"test_label": 1}})

    def test_dataset_rows_reject_duplicate_names_antigravity(self) -> None:
        # Break caught: bootstrap could pair two rows from one dataset.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            normalize_dataset_confusion_rows_antigravity,
        )

        with self.assertRaises(ValueError):
            normalize_dataset_confusion_rows_antigravity((
                {"dataset": "A", "tp": 1, "fp": 0, "fn": 0, "tn": 1},
                {"dataset": "A", "tp": 0, "fp": 1, "fn": 0, "tn": 1},
            ))


if __name__ == "__main__":
    unittest.main()
