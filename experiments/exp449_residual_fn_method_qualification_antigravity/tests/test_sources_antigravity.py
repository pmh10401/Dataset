"""Task 3 label-firewall regression tests; label rows are synthetic only."""
from __future__ import annotations

import unittest
import os
import sys
import types

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
    def _install_prediction_module_antigravity(self, order: tuple[str, ...] | None = None) -> tuple[types.ModuleType, object]:
        name = "experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity"
        module = types.ModuleType(name)
        order = ("Crop_normal_1", *(f"synthetic_target_{index}_antigravity" for index in range(47))) if order is None else order
        class FrozenPredictionAuthorityAntigravity:
            family_method_pairs = 16
            dataset_units = 96
            label_access_count = 0
            target_dataset_order = order
        FrozenPredictionAuthorityAntigravity.__module__ = name
        authority = FrozenPredictionAuthorityAntigravity()
        module.FrozenPredictionAuthorityAntigravity = FrozenPredictionAuthorityAntigravity
        module.validate_frozen_prediction_authority_antigravity = lambda value: order
        self.addCleanup(lambda: sys.modules.pop(name, None))
        sys.modules[name] = module
        return module, authority
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

    def test_module_qualified_authority_requires_its_fixed_validator_before_database_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        module, authority = self._install_prediction_module_antigravity()
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (_ for _ in ()).throw(AssertionError("database reached"))  # type: ignore[assignment]
            module.validate_frozen_prediction_authority_antigravity = lambda value: (_ for _ in ()).throw(ValueError("stale manifest"))
            with self.assertRaisesRegex(ValueError, "prediction authority is invalid"):
                sources.open_test_labels_once_antigravity(authority)
            self.assertIs(type(authority), module.FrozenPredictionAuthorityAntigravity)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_exact_text_label_converter_and_contiguous_rows_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity import convert_test_label_row_antigravity

        self.assertEqual(convert_test_label_row_antigravity(0, "0"), (0, 0))
        self.assertEqual(convert_test_label_row_antigravity(1, "1"), (1, 1))
        for bad in (0, 1, b"0", "2", None):
            with self.assertRaises(ValueError):
                convert_test_label_row_antigravity(0, bad)

    def test_label_session_close_is_truthful_after_incomplete_order_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (-1, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            with self.assertRaisesRegex(ValueError, "did not complete"):
                sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: RecordingConnectionAntigravity()).close_antigravity()
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_query_failure_is_terminal_and_closes_resources_once_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        class FailingConnectionAntigravity:
            def __init__(self) -> None:
                self.close_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
                raise RuntimeError("query boom")
            def close(self) -> None:
                self.close_calls += 1
        connection = FailingConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaisesRegex(RuntimeError, "query boom"):
                session.labels_for_dataset_antigravity("Crop_normal_1")
            self.assertEqual(connection.close_calls, 1)
            with self.assertRaisesRegex(RuntimeError, "query boom"):
                session.labels_for_dataset_antigravity("Crop_normal_1")
            with self.assertRaisesRegex(RuntimeError, "query boom"):
                session.close_antigravity()
            self.assertEqual(connection.close_calls, 1)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_connection_close_failure_is_terminal_close_failed_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        order = tuple(f"close_target_{index}_antigravity" for index in range(48))
        _, authority = self._install_prediction_module_antigravity(order)
        class CloseFailingConnectionAntigravity:
            def __init__(self) -> None:
                self.close_calls = 0
                self.query_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
                self.query_calls += 1
                size = 1518 if self.query_calls == 48 else 1506
                return tuple((index, "0") for index in range(size))
            def close(self) -> None:
                self.close_calls += 1
                raise OSError("close boom")
        connection = CloseFailingConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            for dataset in order:
                session.labels_for_dataset_antigravity(dataset)
            with self.assertRaisesRegex(OSError, "close boom"):
                session.close_antigravity()
            self.assertEqual(session.close_state_antigravity(), "close_failed")
            self.assertEqual(connection.close_calls, 1)
            with self.assertRaisesRegex(OSError, "close boom"):
                session.close_antigravity()
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_query_oserror_with_clean_cleanup_is_terminal_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        class QueryOSErrorConnectionAntigravity:
            def __init__(self) -> None: self.close_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()): raise OSError("query io")  # type: ignore[no-untyped-def]
            def close(self) -> None: self.close_calls += 1
        connection = QueryOSErrorConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaisesRegex(OSError, "query io"): session.labels_for_dataset_antigravity("Crop_normal_1")
            self.assertEqual(session.close_state_antigravity(), "terminal")
            self.assertIsNone(session.close_error_antigravity())
            self.assertIsInstance(session.terminal_error_antigravity(), OSError)
            with self.assertRaisesRegex(OSError, "query io"): session.close_antigravity()
            self.assertEqual(connection.close_calls, 1)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]

    def test_row_failure_and_cleanup_failure_are_recorded_separately_antigravity(self) -> None:
        from experiments.exp449_residual_fn_method_qualification_antigravity import sources_antigravity as sources

        _, authority = self._install_prediction_module_antigravity()
        class BadRowCloseFailingConnectionAntigravity:
            def __init__(self) -> None: self.close_calls = 0
            def execute(self, sql: str, parameters: tuple[object, ...] = ()): return ((0, "bad"),)  # type: ignore[no-untyped-def]
            def close(self) -> None:
                self.close_calls += 1
                raise OSError("cleanup io")
        connection = BadRowCloseFailingConnectionAntigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        original = sources._verify_database_antigravity
        try:
            sources._verify_database_antigravity = lambda: (descriptor, (0, 0, 0, 1, 0, 0, 0))  # type: ignore[assignment]
            session = sources.open_test_labels_once_antigravity(authority, connection_factory=lambda _: connection)
            with self.assertRaisesRegex(ValueError, "TEST label row is invalid"): session.labels_for_dataset_antigravity("Crop_normal_1")
            self.assertEqual(session.close_state_antigravity(), "close_failed")
            self.assertIsInstance(session.terminal_error_antigravity(), ValueError)
            self.assertIsInstance(session.close_error_antigravity(), OSError)
            with self.assertRaisesRegex(ValueError, "TEST label row is invalid"): session.close_antigravity()
            self.assertEqual(connection.close_calls, 1)
        finally:
            sources._verify_database_antigravity = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
