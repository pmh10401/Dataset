"""Synthetic no-label readiness tests for Exp448."""
from __future__ import annotations

import ast
import inspect
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class ReadinessAntigravityTests(unittest.TestCase):
    def test_schema_metadata_reads_only_table_info_and_requires_text_label_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import readiness_antigravity as readiness

        descriptor = os.open("/dev/null", os.O_RDONLY)
        calls: list[str] = []
        class ConnectionAntigravity:
            def set_authorizer(self, _authorizer): pass
            def execute(self, query):
                calls.append(query)
                return ((0, "instance_index", "INTEGER", 0, None, 0), (1, "label", "TEXT", 0, None, 0))
            def close(self): pass
        sqlite = SimpleNamespace(SQLITE_OK=0, SQLITE_DENY=1, SQLITE_PRAGMA=19, connect=lambda *_a, **_k: ConnectionAntigravity())
        with patch.dict("sys.modules", {"sqlite3": sqlite}):
            readiness.verify_instances_schema_metadata_antigravity(descriptor)
        self.assertEqual(calls, ["PRAGMA table_info(instances)"])
        self.assertFalse(any("select" in query.casefold() or "count" in query.casefold() for query in calls))
        os.close(descriptor)

    def test_readiness_rechecks_head_and_root_after_sources_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import readiness_antigravity as readiness

        descriptor = os.open("/dev/null", os.O_RDONLY)
        identity = SimpleNamespace(dataset_order=("synthetic",))
        hard = SimpleNamespace(coordinates=frozenset())
        with (
            patch.object(readiness, "_require_clean_reviewed_head_antigravity", side_effect=(None, ValueError("dirty after scan"))) as clean,
            patch.object(readiness, "_require_document_hashes_antigravity"),
            patch.object(readiness, "_require_absent_run_root_antigravity") as absent,
            patch.object(readiness, "_require_frozen_runtime_antigravity"),
            patch.object(readiness, "verify_exp447_failure_authority_antigravity"),
            patch.object(readiness, "require_absent_exp447_root_antigravity"),
            patch.object(readiness, "_verify_database_antigravity", return_value=(descriptor, (1, 2, 3, 1, 4, 5, 6))),
            patch.object(readiness, "verify_instances_schema_metadata_antigravity"),
            patch.object(readiness, "capture_exp446_publication_antigravity", return_value=identity),
            patch.object(readiness, "capture_hard_predictions_antigravity", return_value=hard),
            self.assertRaisesRegex(ValueError, "dirty after scan"),
        ):
            readiness.readiness_antigravity("a" * 40)
        self.assertEqual(clean.call_count, 2)
        self.assertEqual(absent.call_count, 1)
        with self.assertRaises(OSError): os.fstat(descriptor)

    def test_readiness_has_no_label_session_import_or_root_creation_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import readiness_antigravity

        source = inspect.getsource(readiness_antigravity)
        tree = ast.parse(source)
        imported = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("open_test_labels_antigravity", source)
        self.assertNotIn("mkdir(", source)


if __name__ == "__main__":
    unittest.main()
