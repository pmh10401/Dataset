import unittest

from experiments.exp422_test_eval.audit import compute_confusion_audit
from experiments.exp422_test_eval.contracts import (
    FrozenLabels,
    FrozenPrediction,
    LabelRow,
)


class AuditTests(unittest.TestCase):
    def test_micro_and_dataset_counts_are_deterministic(self) -> None:
        prediction = FrozenPrediction(
            coordinates=(("A", 0), ("B", 1)),
            dataset_scope=("A", "B"),
            prediction_dataset_count=2,
            identity_sha256="a" * 64,
            prediction_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )
        labels = FrozenLabels(
            rows=(
                LabelRow("A", 0, 1),
                LabelRow("A", 1, 0),
                LabelRow("B", 0, 1),
                LabelRow("B", 1, 0),
            ),
            dataset_scope=("A", "B"),
            labels_sha256="d" * 64,
        )
        result = compute_confusion_audit(prediction, labels)
        self.assertEqual(result.micro_counts.as_nested(), [[1, 1], [1, 1]])
        self.assertEqual(result.dataset_rows[0].dataset, "A")
        self.assertEqual(result.dataset_rows[1].dataset, "B")
        self.assertAlmostEqual(result.micro_metrics.f1, 0.5)

    def test_prediction_outside_label_universe_is_rejected(self) -> None:
        prediction = FrozenPrediction(
            coordinates=(("A", 2),),
            dataset_scope=("A",),
            prediction_dataset_count=1,
            identity_sha256="a" * 64,
            prediction_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )
        labels = FrozenLabels(
            rows=(LabelRow("A", 0, 0),),
            dataset_scope=("A",),
            labels_sha256="d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            compute_confusion_audit(prediction, labels)


if __name__ == "__main__":
    unittest.main()
