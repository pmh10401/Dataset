import math
import unittest

from experiments.exp416_exp404_hard_test_confusion_audit.contracts import (
    AuditResult,
    ConfusionCounts,
    ConfusionMetrics,
    DatasetAuditRow,
    FrozenLabels,
    FrozenPrediction,
    LabelRow,
    MacroMetric,
)


class ContractTests(unittest.TestCase):
    def test_label_requires_exact_binary_integer(self) -> None:
        self.assertEqual(LabelRow("d", 1, 1).is_true_anomaly, 1)
        for invalid in (True, False, 0.0, 1.0, "1", -1, 2):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    LabelRow("d", 1, invalid)  # type: ignore[arg-type]

    def test_counts_require_exact_nonnegative_integers(self) -> None:
        self.assertEqual(ConfusionCounts(tp=1, fp=2, fn=3, tn=4).total, 10)
        with self.assertRaises(ValueError):
            ConfusionCounts(tp=True, fp=0, fn=0, tn=0)
        with self.assertRaises(ValueError):
            ConfusionCounts(tp=-1, fp=0, fn=0, tn=0)

    def test_coordinate_collections_reject_unsorted_duplicates_and_bad_members(self) -> None:
        with self.assertRaises(ValueError):
            FrozenPrediction(
                coordinates=(("b", 1), ("a", 1)),
                dataset_scope=("a", "b"),
                prediction_dataset_count=2,
                identity_sha256="identity",
                prediction_sha256="prediction",
                manifest_sha256="manifest",
            )
        with self.assertRaises(ValueError):
            FrozenPrediction(
                coordinates=(("a", 1), ("a", 1)),
                dataset_scope=("a",),
                prediction_dataset_count=1,
                identity_sha256="identity",
                prediction_sha256="prediction",
                manifest_sha256="manifest",
            )
        with self.assertRaises(ValueError):
            LabelRow("", 0, 0)
        with self.assertRaises(ValueError):
            LabelRow("a", -1, 0)
        with self.assertRaises(ValueError):
            FrozenLabels(
                rows=(LabelRow("a", 1, 0), LabelRow("a", 0, 1)),
                dataset_scope=("a",),
                labels_sha256="labels",
            )

    def test_metrics_reject_nonfinite_values_and_bool_counts(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ConfusionMetrics(invalid, 1.0, 1.0, 0.0, 0.0, 1.0)
                with self.assertRaises(ValueError):
                    MacroMetric(invalid, 1)
        with self.assertRaises(ValueError):
            MacroMetric(1.0, True)

    def test_macro_metric_requires_value_and_defined_count_to_agree(self) -> None:
        for value, defined_dataset_count in ((None, 1), (0.5, 0)):
            with self.subTest(
                value=value,
                defined_dataset_count=defined_dataset_count,
            ):
                with self.assertRaises(ValueError):
                    MacroMetric(value, defined_dataset_count)

    def test_dataset_metrics_must_be_derived_from_counts(self) -> None:
        counts = ConfusionCounts(tp=1, fp=1, fn=0, tn=2)
        tampered_metrics = ConfusionMetrics(0.75, 1.0, 2 / 3, 1 / 3, 0.0, 2 / 3)
        with self.assertRaises(ValueError):
            DatasetAuditRow("a", counts, tampered_metrics)

    def test_audit_micro_metrics_must_be_derived_from_counts(self) -> None:
        counts = ConfusionCounts(tp=1, fp=1, fn=0, tn=2)
        row_metrics = ConfusionMetrics(0.5, 1.0, 2 / 3, 1 / 3, 0.0, 2 / 3)
        tampered_micro_metrics = ConfusionMetrics(0.75, 1.0, 2 / 3, 1 / 3, 0.0, 2 / 3)
        with self.assertRaises(ValueError):
            AuditResult(
                micro_counts=counts,
                micro_metrics=tampered_micro_metrics,
                dataset_rows=(DatasetAuditRow("a", counts, row_metrics),),
                macro_metrics={"precision": MacroMetric(0.5, 1)},
                label_rows=4,
                prediction_coordinates=2,
            )

    def test_audit_rejects_prediction_coordinate_count_mismatch(self) -> None:
        counts = ConfusionCounts(tp=1, fp=1, fn=0, tn=2)
        metrics = ConfusionMetrics(0.5, 1.0, 2 / 3, 1 / 3, 0.0, 2 / 3)
        with self.assertRaises(ValueError):
            AuditResult(
                micro_counts=counts,
                micro_metrics=metrics,
                dataset_rows=(DatasetAuditRow("a", counts, metrics),),
                macro_metrics={"precision": MacroMetric(0.5, 1)},
                label_rows=4,
                prediction_coordinates=1,
            )

    def test_zero_denominator_metrics_are_none(self) -> None:
        zero_counts = ConfusionCounts(tp=0, fp=0, fn=0, tn=0)
        undefined_metrics = ConfusionMetrics(None, None, None, None, None, None)
        result = AuditResult(
            micro_counts=zero_counts,
            micro_metrics=undefined_metrics,
            dataset_rows=(DatasetAuditRow("a", zero_counts, undefined_metrics),),
            macro_metrics={"precision": MacroMetric(None, 0)},
            label_rows=0,
            prediction_coordinates=0,
        )
        self.assertIsNone(result.micro_metrics.f1)

    def test_audit_rejects_inconsistent_count_totals(self) -> None:
        counts = ConfusionCounts(tp=1, fp=1, fn=0, tn=2)
        metrics = ConfusionMetrics(0.5, 1.0, 2 / 3, 1 / 3, 0.0, 2 / 3)
        row = DatasetAuditRow("a", counts, metrics)
        with self.assertRaises(ValueError):
            AuditResult(
                micro_counts=counts,
                micro_metrics=metrics,
                dataset_rows=(row,),
                macro_metrics={"precision": MacroMetric(0.5, 1)},
                label_rows=3,
                prediction_coordinates=2,
            )

    def test_valid_audit_contract_preserves_total(self) -> None:
        counts = ConfusionCounts(tp=1, fp=1, fn=0, tn=2)
        metrics = ConfusionMetrics(0.5, 1.0, 2 / 3, 1 / 3, 0.0, 2 / 3)
        result = AuditResult(
            micro_counts=counts,
            micro_metrics=metrics,
            dataset_rows=(DatasetAuditRow("a", counts, metrics),),
            macro_metrics={"precision": MacroMetric(0.5, 1)},
            label_rows=4,
            prediction_coordinates=2,
        )
        self.assertEqual(result.micro_counts.total, result.label_rows)
