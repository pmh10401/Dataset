"""Tests for Exp447's no-label, no-mutation readiness firewall."""
from __future__ import annotations

import ast
import inspect
import os
import unittest
from unittest.mock import patch


class ReadinessAntigravityTests(unittest.TestCase):
    def test_readiness_rejects_non_commit_before_source_access_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.readiness_antigravity import (
            readiness_antigravity,
        )

        with self.assertRaises(ValueError):
            readiness_antigravity("not-a-40-hex-commit")

    def test_readiness_module_has_no_label_session_or_sqlite_import_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import readiness_antigravity

        source = inspect.getsource(readiness_antigravity)
        tree = ast.parse(source)
        imported_names = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        } | {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("sqlite3", imported_names)
        self.assertNotIn("open_test_labels_antigravity", source)

    def test_readiness_verifies_and_closes_database_pin_without_labels_antigravity(self) -> None:
        # Break caught: readiness checks Exp446 text bytes but silently skips the
        # frozen DB SHA/inode/single-link gate before later evaluation is enabled.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import readiness_antigravity as readiness
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.sources_antigravity import (
            HardPredictionsAntigravity,
            capture_exp446_publication_antigravity,
        )

        identity = capture_exp446_publication_antigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        with (
            patch.object(readiness, "_require_clean_reviewed_head_antigravity"),
            patch.object(readiness, "_require_document_hashes_antigravity"),
            patch.object(readiness, "_require_absent_run_root_antigravity"),
            patch.object(readiness, "capture_exp446_publication_antigravity", return_value=identity),
            patch.object(readiness, "capture_hard_predictions_antigravity", return_value=HardPredictionsAntigravity("a" * 64, frozenset())),
            patch.object(readiness, "_verify_database_antigravity", return_value=(descriptor, (1, 2, 3, 1, 4, 5, 6))) as verify,
        ):
            result = readiness.readiness_antigravity("a" * 40)
        verify.assert_called_once_with()
        self.assertEqual(result["label_access_count"], 0)
        with self.assertRaises(OSError):
            os.fstat(descriptor)


if __name__ == "__main__":
    unittest.main()
