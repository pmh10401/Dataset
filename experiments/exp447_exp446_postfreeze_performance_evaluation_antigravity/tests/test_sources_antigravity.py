"""Tests for Exp447's immutable pre-label input authority."""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from dataclasses import replace
from unittest.mock import patch


class SourcesAntigravityTests(unittest.TestCase):
    def test_capture_exp446_publication_has_the_frozen_population_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.sources_antigravity import (
            capture_exp446_publication_antigravity,
        )

        identity = capture_exp446_publication_antigravity()

        self.assertEqual(len(identity.dataset_order), 1117)
        self.assertEqual((identity.source_count, identity.stable_count, identity.unstable_count), (30, 26, 4))
        self.assertEqual(identity.label_access_count, 0)
        self.assertEqual(
            identity.prediction_sha256,
            "602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100",
        )

    def test_capture_hard_predictions_has_the_pinned_digest_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.sources_antigravity import (
            capture_exp446_publication_antigravity,
            capture_hard_predictions_antigravity,
        )

        hard = capture_hard_predictions_antigravity(capture_exp446_publication_antigravity())

        self.assertEqual(hard.sha256, "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70")
        self.assertEqual(len(hard.coordinates), 2218)
        self.assertTrue(all(dataset for dataset, _ in hard.coordinates))

    def test_label_session_is_unreachable_before_two_equal_captures_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.sources_antigravity import (
            capture_exp446_publication_antigravity,
            open_test_labels_antigravity,
        )

        first = capture_exp446_publication_antigravity()
        with self.assertRaises(ValueError):
            open_test_labels_antigravity(first, independently_confirmed=None)

    def test_label_session_requires_two_fresh_internal_captures_antigravity(self) -> None:
        # Break caught: an evaluator could pass two equal public value objects
        # while the source changed after its outer inspection.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        authority = sources.capture_exp446_publication_antigravity()
        changed = replace(authority, root_inode=authority.root_inode + 1)
        with (
            patch.object(sources, "capture_exp446_publication_antigravity", side_effect=(authority, changed)) as capture,
            patch.object(sources, "_verify_database_antigravity") as verify,
            self.assertRaises(ValueError),
        ):
            sources.open_test_labels_antigravity(authority, replace(authority))
        self.assertEqual(capture.call_count, 2)
        verify.assert_not_called()

    def test_pinned_leaf_rejects_post_read_namespace_replacement_antigravity(self) -> None:
        # Break caught: an attacker replaces a same-name leaf after its bytes
        # were read but before the immutable authority returns.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp447_leaf_swap_antigravity_") as temporary:
            root = Path(temporary)
            target = root / "target_antigravity.json"
            target.write_bytes(b"first\n")
            directory_fd = os.open(root, sources._DIRECTORY_FLAGS_ANTIGRAVITY)
            try:
                pinned = sources._pin_leaf_antigravity(directory_fd, target.name)
                try:
                    self.assertEqual(sources._read_pinned_leaf_antigravity(pinned), b"first\n")
                    replacement = root / "replacement_antigravity.json"
                    replacement.write_bytes(b"second\n")
                    os.replace(replacement, target)
                    with self.assertRaises(ValueError):
                        sources._assert_pinned_leaf_antigravity(directory_fd, pinned)
                finally:
                    os.close(pinned.descriptor)
            finally:
                os.close(directory_fd)

    def test_database_verifier_rejects_digest_inode_and_hardlink_antigravity(self) -> None:
        # Break caught: a name-stable or linked database reaches the label
        # session without an exact byte/inode/single-link verification.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp447_db_gate_antigravity_") as temporary:
            database = Path(temporary) / "database_antigravity.db"
            database.write_bytes(b"immutable fixture\n")
            digest = __import__("hashlib").sha256(database.read_bytes()).hexdigest()
            inode = database.stat().st_ino
            descriptor, _identity = sources._verify_database_path_antigravity(database, digest, inode)
            os.close(descriptor)
            with self.assertRaises(ValueError):
                sources._verify_database_path_antigravity(database, "0" * 64, inode)
            with self.assertRaises(ValueError):
                sources._verify_database_path_antigravity(database, digest, inode + 1)
            alias = Path(temporary) / "alias_antigravity.db"
            os.link(database, alias)
            try:
                with self.assertRaises(ValueError):
                    sources._verify_database_path_antigravity(database, digest, inode)
            finally:
                alias.unlink()

    def test_label_session_close_rehashes_database_after_connection_close_antigravity(self) -> None:
        # Break caught: a database changed during an already-open label session
        # is checked only by inode metadata, not its exact bytes.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        authority = sources.capture_exp446_publication_antigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        expected = sources._identity_antigravity(os.fstat(descriptor))

        class ConnectionAntigravity:
            def set_authorizer(self, _authorizer: object) -> None:
                pass

            def close(self) -> None:
                pass

        sqlite = SimpleNamespace(
            SQLITE_SELECT=1, SQLITE_READ=2, SQLITE_OK=0, SQLITE_DENY=1,
            connect=lambda *_args, **_kwargs: ConnectionAntigravity(),
        )
        with (
            patch.object(sources, "capture_exp446_publication_antigravity", side_effect=(replace(authority), replace(authority))),
            patch.object(sources, "_verify_database_antigravity", side_effect=((descriptor, expected), ValueError("digest changed"))) as verify,
            patch.dict(sys.modules, {"sqlite3": sqlite}),
        ):
            session = sources.open_test_labels_antigravity(authority, replace(authority))
            with self.assertRaisesRegex(ValueError, "digest changed"):
                session.close_antigravity()
        self.assertEqual(verify.call_count, 2)
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_label_session_close_failure_is_terminal_and_releases_descriptor_antigravity(self) -> None:
        """A failed connection close cannot be relabelled as a successful retry."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        authority = sources.capture_exp446_publication_antigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        expected = sources._identity_antigravity(os.fstat(descriptor))
        close_calls = 0

        class ConnectionAntigravity:
            def set_authorizer(self, _authorizer: object) -> None:
                pass

            def close(self) -> None:
                nonlocal close_calls
                close_calls += 1
                raise OSError("connection close failed")

        sqlite = SimpleNamespace(
            SQLITE_SELECT=1, SQLITE_READ=2, SQLITE_OK=0, SQLITE_DENY=1,
            connect=lambda *_args, **_kwargs: ConnectionAntigravity(),
        )
        with (
            patch.object(sources, "capture_exp446_publication_antigravity", side_effect=(replace(authority), replace(authority))),
            patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, expected)),
            patch.dict(sys.modules, {"sqlite3": sqlite}),
        ):
            session = sources.open_test_labels_antigravity(authority, replace(authority))
            with self.assertRaisesRegex(OSError, "connection close failed"):
                session.close_antigravity()
            with self.assertRaisesRegex(OSError, "connection close failed"):
                session.close_antigravity()
        self.assertEqual(close_calls, 1)
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_label_setup_authorizer_failure_closes_connection_and_descriptor_antigravity(self) -> None:
        """A failing authorizer setup releases every already-open local resource."""
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        authority = sources.capture_exp446_publication_antigravity()
        descriptor = os.open("/dev/null", os.O_RDONLY)
        expected = sources._identity_antigravity(os.fstat(descriptor))
        close_calls = 0

        class ConnectionAntigravity:
            def set_authorizer(self, _authorizer: object) -> None:
                raise OSError("authorizer setup failed")

            def close(self) -> None:
                nonlocal close_calls
                close_calls += 1

        sqlite = SimpleNamespace(
            SQLITE_SELECT=1, SQLITE_READ=2, SQLITE_OK=0, SQLITE_DENY=1,
            connect=lambda *_args, **_kwargs: ConnectionAntigravity(),
        )
        with (
            patch.object(sources, "capture_exp446_publication_antigravity", side_effect=(replace(authority), replace(authority))),
            patch.object(sources, "_verify_database_antigravity", return_value=(descriptor, expected)),
            patch.dict(sys.modules, {"sqlite3": sqlite}),
            self.assertRaisesRegex(OSError, "authorizer setup failed"),
        ):
            sources.open_test_labels_antigravity(authority, replace(authority))
        self.assertEqual(close_calls, 1)
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_exp404_capture_rejects_tampered_bytes_before_schema_antigravity(self) -> None:
        # Break caught: Exp404's filename and later row validation are trusted
        # while its frozen byte digest no longer matches.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        authority = sources.capture_exp446_publication_antigravity()
        with patch.object(sources, "_read_pinned_leaf_antigravity", return_value=b"{}\n"):
            with self.assertRaisesRegex(ValueError, "digest differs"):
                sources.capture_hard_predictions_antigravity(authority)

    def test_source_module_has_no_eager_sqlite_import_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity

        tree = ast.parse(inspect.getsource(sources_antigravity))
        eager = [
            node for node in tree.body
            if (isinstance(node, ast.Import) and any(alias.name == "sqlite3" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "sqlite3")
        ]
        self.assertEqual(eager, [])

    def test_publication_identity_module_has_no_label_authority_import_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import publication_identity_antigravity

        tree = ast.parse(inspect.getsource(publication_identity_antigravity))
        imported = {
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        } | {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("sources_antigravity", imported)
        self.assertNotIn("open_test_labels_antigravity", inspect.getsource(publication_identity_antigravity))

    def test_descriptor_helpers_reject_extra_or_symlinked_leaf_antigravity(self) -> None:
        # Break caught: a source inspector accepts a namespace addition or a
        # symlink that can be replaced after the initial path check.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp447_sources_antigravity_") as temporary:
            root = Path(temporary)
            (root / "safe_antigravity.json").write_bytes(b"{}\n")
            descriptor = os.open(root, sources._DIRECTORY_FLAGS_ANTIGRAVITY)
            try:
                sources._exact_names_antigravity(descriptor, ("safe_antigravity.json",))
                (root / "extra_antigravity.json").write_bytes(b"{}\n")
                with self.assertRaises(ValueError):
                    sources._exact_names_antigravity(descriptor, ("safe_antigravity.json",))
            finally:
                os.close(descriptor)
            (root / "extra_antigravity.json").unlink()
            (root / "link_antigravity.json").symlink_to(root / "safe_antigravity.json")
            descriptor = os.open(root, sources._DIRECTORY_FLAGS_ANTIGRAVITY)
            try:
                with self.assertRaises(ValueError):
                    sources._read_leaf_antigravity(descriptor, "link_antigravity.json")
            finally:
                os.close(descriptor)

    def test_descriptor_helpers_detect_replaced_nested_directory_antigravity(self) -> None:
        # Break caught: a verified nested publication directory is swapped in
        # its parent namespace before the capture returns.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity as sources

        with tempfile.TemporaryDirectory(prefix="exp447_nested_antigravity_") as temporary:
            root = Path(temporary)
            child = root / "publication_antigravity"
            child.mkdir()
            parent_fd = os.open(root, sources._DIRECTORY_FLAGS_ANTIGRAVITY)
            try:
                child_fd, identity = sources._open_directory_antigravity(parent_fd, child.name)
                try:
                    child.rename(root / "retired_antigravity")
                    child.mkdir()
                    with self.assertRaises(ValueError):
                        sources._assert_directory_antigravity(parent_fd, child.name, child_fd, identity)
                finally:
                    os.close(child_fd)
            finally:
                os.close(parent_fd)


if __name__ == "__main__":
    unittest.main()
