"""Task 3 label-firewall regression tests; label rows are synthetic only."""
from __future__ import annotations

import unittest

import numpy as np


class RecordingConnectionAntigravity:
    def __init__(self) -> None:
        self.sql: list[str] = []
        self.closed = False

    def execute(self, sql: str, parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
        self.sql.append(sql)
        if "values_blob" in sql:
            return (("TRAIN", 0, np.asarray([1.0, 2.0], dtype="<f4").tobytes()), ("TRAIN", 1, np.asarray([2.0, 3.0], dtype="<f4").tobytes()), ("TEST", 0, np.asarray([3.0, 4.0], dtype="<f4").tobytes()))
        return ()

    def close(self) -> None:
        self.closed = True


class SourcesAntigravityTests(unittest.TestCase):
    def test_values_bundle_never_selects_label_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import load_target_bundle_antigravity

        connection = RecordingConnectionAntigravity()
        bundle = load_target_bundle_antigravity("Crop_normal_1", connection_factory=lambda _: connection)
        self.assertEqual(bundle.dataset, "Crop_normal_1")
        self.assertEqual(bundle.train_normal.shape, (2, 2))
        self.assertEqual(bundle.test_values.shape, (1, 2))
        self.assertFalse(any("label" in sql.casefold() for sql in connection.sql))
        self.assertTrue(connection.closed)

    def test_forged_prediction_authority_is_rejected_before_database_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import open_test_labels_once_antigravity

        with self.assertRaisesRegex(ValueError, "prediction authority is invalid"):
            open_test_labels_once_antigravity(object())

    def test_exact_text_label_converter_and_contiguous_rows_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import convert_test_label_row_antigravity

        self.assertEqual(convert_test_label_row_antigravity(0, "0"), (0, 0))
        self.assertEqual(convert_test_label_row_antigravity(1, "1"), (1, 1))
        for bad in (0, 1, b"0", "2", None):
            with self.assertRaises(ValueError):
                convert_test_label_row_antigravity(0, bad)

    def test_label_session_close_is_truthful_after_incomplete_order_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        class FrozenPredictionAuthorityAntigravity:
            family_method_pairs = 16
            dataset_units = 96
            label_access_count = 0
            target_dataset_order = ("Crop_normal_1",)
        FrozenPredictionAuthorityAntigravity.__module__ = "experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity"
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (-1, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            with self.assertRaisesRegex(ValueError, "did not complete"):
                sources.open_test_labels_once_antigravity(FrozenPredictionAuthorityAntigravity(), connection_factory=lambda _: RecordingConnectionAntigravity()).close_antigravity()
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
