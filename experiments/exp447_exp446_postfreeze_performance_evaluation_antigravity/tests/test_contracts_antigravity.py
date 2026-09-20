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
        datasets = tuple(f"dataset-{index:04d}" for index in range(1117))
        source = tuple((datasets[index], index) for index in range(30))
        identity = PublicationIdentityAntigravity(
            root_device=1,
            root_inode=2,
            leaf_sha256=leaves,
            manifest_sha256=digest,
            prediction_sha256=digest,
            artifact_map_sha256=digest,
            dataset_order=datasets,
            source_coordinates=source,
            stable_coordinates=source[:26],
            unstable_coordinates=source[26:],
            label_access_count=0,
        )
        leaves["forged"] = digest
        self.assertIsInstance(identity.leaf_sha256, MappingProxyType)
        self.assertEqual(tuple(identity.leaf_sha256), ("prediction_antigravity.json",))
        self.assertEqual(identity.source_count, 30)
        self.assertEqual(identity.stable_count + identity.unstable_count, identity.source_count)
        with self.assertRaises(TypeError):
            identity.leaf_sha256["forged"] = digest  # type: ignore[index]
        with self.assertRaises(ValueError):
            PublicationIdentityAntigravity(
                1, 2, {"leaf": digest}, digest, digest, digest, datasets,
                source, source[:26], source[25:29], 0,
            )

    def test_prelabel_payload_recursively_rejects_label_like_keys_antigravity(self) -> None:
        # Break caught: pre-label source evidence carries a hidden TEST-label field.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            freeze_prelabel_payload_antigravity,
        )

        frozen = freeze_prelabel_payload_antigravity({"source": {"candidate_count": 30}, "lanes": ["stable_review"]})
        self.assertIsInstance(frozen, MappingProxyType)
        for payload in (
            {"source": {"test_label": 1}},
            {"source": {"class-id": 1}},
            {"source": {"outlier_score": 1}},
            {"source": {"nested": {"TEST_y": 1}}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                freeze_prelabel_payload_antigravity(payload)

    def test_publication_identity_rejects_wrong_population_or_lane_order_antigravity(self) -> None:
        # Break caught: an inspected publication has right set membership but a
        # wrong population or a lane order that cannot bind frozen source order.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            PublicationIdentityAntigravity,
        )

        digest = "b" * 64
        datasets = tuple(f"dataset-{index:04d}" for index in range(1117))
        source = tuple((datasets[index], index) for index in range(30))
        with self.assertRaises(ValueError):
            PublicationIdentityAntigravity(1, 2, {"leaf": digest}, digest, digest, digest, datasets[:-1], source, source[:26], source[26:], 0)
        with self.assertRaises(ValueError):
            PublicationIdentityAntigravity(
                1, 2, {"leaf": digest}, digest, digest, digest, datasets, source,
                (source[1], source[0], *source[2:26]), source[26:], 0,
            )

    def test_publication_identity_preserves_real_interleaved_lane_order_antigravity(self) -> None:
        # Break caught: the genuine Exp446 source rank order is rewritten into
        # stable-then-unstable order even though the lane sets conserve it.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.contracts_antigravity import (
            PublicationIdentityAntigravity,
        )

        digest = "c" * 64
        datasets = tuple(f"dataset-{index:04d}" for index in range(1117))
        source = tuple((datasets[index], index) for index in range(30))
        stable_ranks = (0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29)
        unstable_ranks = (8, 10, 14, 24)
        identity = PublicationIdentityAntigravity(
            1, 2, {"leaf": digest}, digest, digest, digest, datasets, source,
            tuple(source[index] for index in stable_ranks),
            tuple(source[index] for index in unstable_ranks), 0,
        )
        self.assertEqual(identity.source_coordinates, source)
        self.assertEqual(identity.stable_coordinates, tuple(source[index] for index in stable_ranks))

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
