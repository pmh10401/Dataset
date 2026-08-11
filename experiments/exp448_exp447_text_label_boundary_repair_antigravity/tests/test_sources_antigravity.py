"""Synthetic-only tests for the predecessor audit and TEXT converter."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def _audit_payload_antigravity() -> bytes:
    return b'{"destination_state":"absent","error":"TEST label row is invalid","error_type":"ValueError","label_rows":0,"label_session_closed":true,"label_session_state":"closed","phase":"label_snapshot","query_attempts":1,"query_completions":0,"schema":"exp447_evaluation_audit"}\n'


class SourcesAntigravityTests(unittest.TestCase):
    def test_failure_audit_requires_exact_identity_and_facts_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp448_audit_antigravity_") as temporary:
            audit = Path(temporary) / "audit_antigravity.json"
            payload = _audit_payload_antigravity()
            audit.write_bytes(payload)
            stat_result = audit.stat()
            constants = {
                "_EXP447_FAILURE_AUDIT_ANTIGRAVITY": audit,
                "_EXP447_FAILURE_AUDIT_SHA256_ANTIGRAVITY": hashlib.sha256(payload).hexdigest(),
                "_EXP447_FAILURE_AUDIT_INODE_ANTIGRAVITY": stat_result.st_ino,
                "_EXP447_FAILURE_AUDIT_NLINK_ANTIGRAVITY": 1,
            }
            with patch.multiple(sources, **constants):
                sources.verify_exp447_failure_authority_antigravity()
                for key, value in (("schema", b"bad_schema"), ("phase", b"bad_phase"), ("error", b"bad_error")):
                    audit.write_bytes(payload.replace(value, value))
                    altered = payload.replace(b"exp447_evaluation_audit" if key == "schema" else (b"label_snapshot" if key == "phase" else b"TEST label row is invalid"), value)
                    audit.write_bytes(altered)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        sources.verify_exp447_failure_authority_antigravity()
                    audit.write_bytes(payload)
                with patch.object(sources, "_EXP447_FAILURE_AUDIT_SHA256_ANTIGRAVITY", "0" * 64), self.assertRaises(ValueError):
                    sources.verify_exp447_failure_authority_antigravity()

    def test_audit_and_predecessor_root_reject_symlinks_or_presence_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp448_root_antigravity_") as temporary:
            root = Path(temporary)
            target = root / "target_antigravity.json"
            target.write_bytes(_audit_payload_antigravity())
            audit_link = root / "audit_link_antigravity.json"
            audit_link.symlink_to(target)
            with patch.object(sources, "_EXP447_FAILURE_AUDIT_ANTIGRAVITY", audit_link), self.assertRaises(ValueError):
                sources.verify_exp447_failure_authority_antigravity()
            present = root / "present_antigravity"
            present.mkdir()
            with patch.object(sources, "_EXP447_RUN_ROOT_ANTIGRAVITY", present), self.assertRaises(ValueError):
                sources.require_absent_exp447_root_antigravity()
            link = root / "root_link_antigravity"
            link.symlink_to(present)
            with patch.object(sources, "_EXP447_RUN_ROOT_ANTIGRAVITY", link), self.assertRaises(ValueError):
                sources.require_absent_exp447_root_antigravity()

    def test_exact_text_converter_accepts_only_contract_pair_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.sources_antigravity import convert_test_label_row_antigravity

        self.assertEqual(convert_test_label_row_antigravity(0, "0"), (0, 0))
        self.assertEqual(convert_test_label_row_antigravity(1, "1"), (1, 1))
        for raw in (0, True, 0.0, b"0", " 0", "0 ", None, "2", ""):
            with self.subTest(raw=repr(raw)), self.assertRaises(ValueError):
                convert_test_label_row_antigravity(0, raw)
        with self.assertRaises(ValueError):
            convert_test_label_row_antigravity(True, "0")

    def test_label_capability_fails_before_database_for_bad_predecessor_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import sources_antigravity as sources

        with patch.object(sources, "_verify_database_antigravity") as database, self.assertRaises(ValueError):
            sources.open_test_labels_antigravity(object(), object(), object())
        database.assert_not_called()

    def test_sources_have_no_eager_sqlite_or_runtime_import_antigravity(self) -> None:
        import ast
        import inspect
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import sources_antigravity

        tree = ast.parse(inspect.getsource(sources_antigravity))
        imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        self.assertNotIn("sqlite3", imports)
        self.assertNotIn("evaluation_antigravity", imports)


if __name__ == "__main__":
    unittest.main()
