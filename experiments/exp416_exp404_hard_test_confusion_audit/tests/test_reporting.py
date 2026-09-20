from contextlib import contextmanager
import csv
from dataclasses import replace
from io import StringIO
import json
import math
from typing import Iterator
import unittest
from unittest import mock

from experiments.exp416_exp404_hard_test_confusion_audit.audit import (
    compute_confusion_audit,
)
from experiments.exp416_exp404_hard_test_confusion_audit.config import (
    Exp416FrozenConfig,
    FROZEN_CONFIG,
)
from experiments.exp416_exp404_hard_test_confusion_audit.contracts import (
    AuditResult,
    FrozenLabels,
    FrozenPrediction,
    LabelRow,
)
from experiments.exp416_exp404_hard_test_confusion_audit.reporting import (
    DATASET_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    render_payloads,
    verify_rendered_payloads,
)


DATASET_FIELDS = (
    "dataset",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "f1",
)
METRIC_NAMES = (
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "f1",
)


@contextmanager
def fixture_config_validation() -> Iterator[None]:
    with mock.patch.object(
        Exp416FrozenConfig,
        "validate",
        autospec=True,
        return_value=None,
    ):
        yield


def hand_config() -> Exp416FrozenConfig:
    return replace(
        FROZEN_CONFIG,
        expected_datasets=2,
        expected_prediction_datasets=2,
        expected_prediction_coordinates=3,
        expected_label_rows=6,
        prediction_sha256="1" * 64,
        manifest_sha256="2" * 64,
        labels_sha256="3" * 64,
        database_sha256="4" * 64,
    )


def hand_frozen_prediction() -> FrozenPrediction:
    return FrozenPrediction(
        coordinates=(("A", 0), ("A", 2), ("B", 1)),
        dataset_scope=("A", "B"),
        prediction_dataset_count=2,
        identity_sha256="5" * 64,
        prediction_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )


def hand_frozen_labels() -> FrozenLabels:
    return FrozenLabels(
        rows=(
            LabelRow("A", 0, 1),
            LabelRow("A", 1, 1),
            LabelRow("A", 2, 0),
            LabelRow("B", 0, 0),
            LabelRow("B", 1, 1),
            LabelRow("B", 2, 0),
        ),
        dataset_scope=("A", "B"),
        labels_sha256="3" * 64,
    )


def hand_audit_result() -> AuditResult:
    return compute_confusion_audit(
        hand_frozen_prediction(),
        hand_frozen_labels(),
    )


def valid_payloads(
    config: Exp416FrozenConfig | None = None,
) -> dict[str, bytes]:
    selected = hand_config() if config is None else config
    with fixture_config_validation():
        return dict(
            render_payloads(
                hand_audit_result(),
                hand_frozen_prediction(),
                hand_frozen_labels(),
                selected.database_sha256,
                "a" * 40,
                config=selected,
            )
        )


def mutate_summary(
    payloads: dict[str, bytes],
    mutation: object,
) -> dict[str, bytes]:
    summary = json.loads(payloads[SUMMARY_FILE])
    assert callable(mutation)
    mutation(summary)
    payloads[SUMMARY_FILE] = (
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=True,
        )
        + "\n"
    ).encode("utf-8")
    return payloads


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = hand_config()

    def verify(self, payloads: dict[str, bytes]) -> None:
        with fixture_config_validation():
            verify_rendered_payloads(payloads, config=self.config)

    def test_renders_exact_payloads_and_governance_flags(self) -> None:
        with fixture_config_validation():
            payloads = render_payloads(
                hand_audit_result(),
                hand_frozen_prediction(),
                hand_frozen_labels(),
                self.config.database_sha256,
                "a" * 40,
                config=self.config,
            )

        self.assertEqual(
            set(payloads),
            {
                "exp416_confusion_matrix.json",
                "exp416_dataset_metrics.csv",
                "EXP416_RESULT_REPORT_KO.md",
            },
        )
        summary = json.loads(payloads[SUMMARY_FILE])
        self.assertEqual(
            summary["micro"]["counts"],
            {"tp": 2, "fp": 1, "fn": 1, "tn": 2},
        )
        self.assertFalse(summary["prediction_modified"])
        self.assertTrue(
            summary["test_labels_loaded_after_prediction_freeze"]
        )
        self.assertFalse(summary["test_informed_reselection"])
        self.assertFalse(summary["operational_policy_changed"])
        self.assertFalse(summary["exp415_prediction_created"])

    def test_csv_has_fixed_header_sorted_rows_and_empty_undefined_cells(
        self,
    ) -> None:
        labels = FrozenLabels(
            rows=(LabelRow("A", 0, 1), LabelRow("B", 0, 0)),
            dataset_scope=("A", "B"),
            labels_sha256="3" * 64,
        )
        prediction = FrozenPrediction(
            coordinates=(),
            dataset_scope=("A", "B"),
            prediction_dataset_count=0,
            identity_sha256="5" * 64,
            prediction_sha256="1" * 64,
            manifest_sha256="2" * 64,
        )
        config = replace(
            self.config,
            expected_prediction_datasets=0,
            expected_prediction_coordinates=0,
            expected_label_rows=2,
        )
        with fixture_config_validation():
            payloads = render_payloads(
                compute_confusion_audit(prediction, labels),
                prediction,
                labels,
                config.database_sha256,
                "a" * 40,
                config=config,
            )

        reader = csv.DictReader(
            StringIO(payloads[DATASET_FILE].decode("utf-8"))
        )
        rows = tuple(reader)
        self.assertEqual(tuple(reader.fieldnames or ()), DATASET_FIELDS)
        self.assertEqual([row["dataset"] for row in rows], ["A", "B"])
        self.assertEqual(rows[0]["precision"], "")
        self.assertEqual(rows[0]["specificity"], "")
        self.assertEqual(rows[1]["recall"], "")
        self.assertEqual(rows[1]["f1"], "")
        self.assertTrue(payloads[DATASET_FILE].endswith(b"\n"))
        self.assertNotIn(b"\r\n", payloads[DATASET_FILE])

    def test_markdown_contains_integer_matrix_metrics_and_no_reselection_boundary(
        self,
    ) -> None:
        report = valid_payloads(self.config)[REPORT_FILE].decode("utf-8")

        self.assertIn("| 실제 양성 | 2 | 1 |", report)
        self.assertIn("| 실제 음성 | 1 | 2 |", report)
        self.assertIn("precision: `0.6666666666666666`", report)
        self.assertIn("dataset-balanced macro", report)
        self.assertIn("TEST 결과로 재선택하지 않는다", report)
        self.assertIn("운영 정책을 변경하지 않는다", report)

    def test_verifier_accepts_rendered_fixture(self) -> None:
        self.verify(valid_payloads(self.config))

    def test_rejects_count_changed_without_metric_change(self) -> None:
        payloads = mutate_summary(
            valid_payloads(self.config),
            lambda value: value["micro"]["counts"].__setitem__("tp", 3),
        )
        with self.assertRaisesRegex(ValueError, "micro|count|metric"):
            self.verify(payloads)

    def test_rejects_total_not_equal_to_frozen_label_rows(self) -> None:
        payloads = mutate_summary(
            valid_payloads(self.config),
            lambda value: value.__setitem__("frozen_label_row_count", 7),
        )
        with self.assertRaisesRegex(ValueError, "label|total|count"):
            self.verify(payloads)

    def test_rejects_macro_defined_count_above_dataset_count(self) -> None:
        payloads = mutate_summary(
            valid_payloads(self.config),
            lambda value: value["macro"]["precision"].__setitem__(
                "defined_dataset_count",
                3,
            ),
        )
        with self.assertRaisesRegex(ValueError, "macro|defined"):
            self.verify(payloads)

    def test_rejects_nonfinite_json_numbers(self) -> None:
        for name, number in (("nan", math.nan), ("infinity", math.inf)):
            with self.subTest(name=name):
                payloads = mutate_summary(
                    valid_payloads(self.config),
                    lambda value, number=number: value["micro"][
                        "metrics"
                    ].__setitem__("precision", number),
                )
                with self.assertRaisesRegex(ValueError, "JSON|finite|metric"):
                    self.verify(payloads)

    def test_rejects_numeric_string_in_json(self) -> None:
        payloads = mutate_summary(
            valid_payloads(self.config),
            lambda value: value["micro"]["metrics"].__setitem__(
                "precision",
                "0.6666666666666666",
            ),
        )
        with self.assertRaisesRegex(ValueError, "metric|number|type"):
            self.verify(payloads)

    def test_rejects_missing_or_extra_payload(self) -> None:
        for name, payloads in (
            (
                "missing",
                {
                    key: value
                    for key, value in valid_payloads(self.config).items()
                    if key != REPORT_FILE
                },
            ),
            (
                "extra",
                {
                    **valid_payloads(self.config),
                    "foreign.bin": b"foreign\n",
                },
            ),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                "coverage|payload|files",
            ):
                self.verify(payloads)

    def test_rejects_governance_flag_changed_to_true(self) -> None:
        payloads = mutate_summary(
            valid_payloads(self.config),
            lambda value: value.__setitem__("test_informed_reselection", True),
        )
        with self.assertRaisesRegex(ValueError, "governance|reselection"):
            self.verify(payloads)

    def test_rejects_csv_dataset_order_change(self) -> None:
        payloads = valid_payloads(self.config)
        lines = payloads[DATASET_FILE].splitlines(keepends=True)
        payloads[DATASET_FILE] = b"".join((lines[0], lines[2], lines[1]))

        with self.assertRaisesRegex(ValueError, "CSV|order|dataset"):
            self.verify(payloads)

    def test_rejects_csv_count_mismatch(self) -> None:
        payloads = valid_payloads(self.config)
        rows = list(
            csv.DictReader(
                StringIO(payloads[DATASET_FILE].decode("utf-8"))
            )
        )
        rows[0]["tp"] = "2"
        buffer = StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=DATASET_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        payloads[DATASET_FILE] = buffer.getvalue().encode("utf-8")

        with self.assertRaisesRegex(ValueError, "CSV|count|micro"):
            self.verify(payloads)

    def test_rejects_extra_json_or_csv_field(self) -> None:
        json_payloads = mutate_summary(
            valid_payloads(self.config),
            lambda value: value.__setitem__("foreign", False),
        )
        csv_payloads = valid_payloads(self.config)
        csv_payloads[DATASET_FILE] = csv_payloads[DATASET_FILE].replace(
            b"dataset,tp,",
            b"dataset,foreign,tp,",
            1,
        ).replace(b"A,1,", b"A,x,1,", 1).replace(
            b"B,1,", b"B,x,1,", 1
        )

        for name, payloads in (
            ("json", json_payloads),
            ("csv", csv_payloads),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                "schema|header",
            ):
                self.verify(payloads)

    def test_rejects_csv_metric_tampering_and_recomputed_macro_tampering(
        self,
    ) -> None:
        csv_payloads = valid_payloads(self.config)
        csv_payloads[DATASET_FILE] = csv_payloads[DATASET_FILE].replace(
            b"0.5,0.5,0.0,1.0,0.5,0.5",
            b"0.6,0.5,0.0,1.0,0.5,0.5",
            1,
        )
        macro_payloads = mutate_summary(
            valid_payloads(self.config),
            lambda value: value["macro"]["f1"].__setitem__("value", 0.5),
        )

        for name, payloads in (
            ("csv_metric", csv_payloads),
            ("macro", macro_payloads),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                "metric|macro",
            ):
                self.verify(payloads)


if __name__ == "__main__":
    unittest.main()
