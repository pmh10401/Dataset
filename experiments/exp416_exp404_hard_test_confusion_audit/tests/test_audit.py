import unittest

from experiments.exp416_exp404_hard_test_confusion_audit.audit import (
    compute_confusion_audit,
    metrics_from_counts,
)
from experiments.exp416_exp404_hard_test_confusion_audit.contracts import (
    ConfusionCounts,
    FrozenLabels,
    FrozenPrediction,
    LabelRow,
)


def frozen_prediction(
    *, coordinates: tuple[tuple[str, int], ...], dataset_scope: tuple[str, ...]
) -> FrozenPrediction:
    return FrozenPrediction(
        coordinates=coordinates,
        dataset_scope=dataset_scope,
        prediction_dataset_count=len({dataset for dataset, _ in coordinates}),
        identity_sha256="synthetic-identity",
        prediction_sha256="synthetic-prediction",
        manifest_sha256="synthetic-manifest",
    )


def prediction_with_no_coordinates(dataset_scope: tuple[str, ...]) -> FrozenPrediction:
    return frozen_prediction(coordinates=(), dataset_scope=dataset_scope)


def frozen_labels(
    rows: tuple[LabelRow, ...], *, dataset_scope: tuple[str, ...]
) -> FrozenLabels:
    return FrozenLabels(
        rows=rows,
        dataset_scope=dataset_scope,
        labels_sha256="synthetic-labels",
    )


class ConfusionAuditTests(unittest.TestCase):
    def test_computes_exact_micro_confusion_matrix(self) -> None:
        prediction = frozen_prediction(
            coordinates=(("A", 0), ("A", 2), ("B", 1)),
            dataset_scope=("A", "B"),
        )
        labels = frozen_labels(
            (
                LabelRow("A", 0, 1),
                LabelRow("A", 1, 1),
                LabelRow("A", 2, 0),
                LabelRow("B", 0, 0),
                LabelRow("B", 1, 1),
                LabelRow("B", 2, 0),
            ),
            dataset_scope=("A", "B"),
        )

        result = compute_confusion_audit(prediction, labels)

        self.assertEqual(result.micro_counts, ConfusionCounts(tp=2, fp=1, fn=1, tn=2))
        self.assertAlmostEqual(result.micro_metrics.precision, 2 / 3)
        self.assertAlmostEqual(result.micro_metrics.recall, 2 / 3)
        self.assertAlmostEqual(result.micro_metrics.specificity, 2 / 3)
        self.assertAlmostEqual(result.micro_metrics.f1, 2 / 3)

    def test_undefined_metrics_remain_none(self) -> None:
        metrics = metrics_from_counts(ConfusionCounts(tp=0, fp=0, fn=0, tn=5))

        self.assertIsNone(metrics.precision)
        self.assertIsNone(metrics.recall)
        self.assertEqual(metrics.specificity, 1.0)
        self.assertEqual(metrics.false_positive_rate, 0.0)
        self.assertIsNone(metrics.false_negative_rate)
        self.assertIsNone(metrics.f1)

    def test_macro_uses_only_defined_dataset_values(self) -> None:
        result = compute_confusion_audit(
            prediction_with_no_coordinates(("A", "B")),
            frozen_labels(
                (
                    LabelRow("A", 0, 1),
                    LabelRow("B", 0, 0),
                ),
                dataset_scope=("A", "B"),
            ),
        )

        self.assertEqual(result.macro_metrics["recall"].defined_dataset_count, 1)
        self.assertEqual(result.macro_metrics["recall"].value, 0.0)
        self.assertEqual(result.macro_metrics["specificity"].defined_dataset_count, 1)
        self.assertEqual(result.macro_metrics["specificity"].value, 1.0)

    def test_rejects_mismatched_dataset_scopes(self) -> None:
        prediction = prediction_with_no_coordinates(("A",))
        labels = frozen_labels((LabelRow("B", 0, 0),), dataset_scope=("B",))

        with self.assertRaisesRegex(ValueError, "scope"):
            compute_confusion_audit(prediction, labels)

    def test_rejects_prediction_coordinate_absent_from_labels(self) -> None:
        prediction = frozen_prediction(coordinates=(("A", 1),), dataset_scope=("A",))
        labels = frozen_labels((LabelRow("A", 0, 0),), dataset_scope=("A",))

        with self.assertRaisesRegex(ValueError, "outside label universe"):
            compute_confusion_audit(prediction, labels)

    def test_sorts_dataset_rows_and_conserves_counts(self) -> None:
        result = compute_confusion_audit(
            frozen_prediction(coordinates=(("A", 0),), dataset_scope=("A", "B")),
            frozen_labels(
                (
                    LabelRow("A", 0, 1),
                    LabelRow("B", 0, 0),
                ),
                dataset_scope=("A", "B"),
            ),
        )

        self.assertEqual(tuple(row.dataset for row in result.dataset_rows), ("A", "B"))
        self.assertEqual(
            result.micro_counts,
            ConfusionCounts(
                tp=sum(row.counts.tp for row in result.dataset_rows),
                fp=sum(row.counts.fp for row in result.dataset_rows),
                fn=sum(row.counts.fn for row in result.dataset_rows),
                tn=sum(row.counts.tn for row in result.dataset_rows),
            ),
        )
        self.assertEqual(result.micro_counts.total, result.label_rows)

