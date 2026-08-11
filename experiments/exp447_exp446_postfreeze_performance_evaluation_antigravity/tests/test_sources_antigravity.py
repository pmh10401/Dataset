"""Tests for Exp447's immutable pre-label input authority."""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import tempfile
import unittest


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

    def test_source_module_has_no_eager_sqlite_import_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import sources_antigravity

        tree = ast.parse(inspect.getsource(sources_antigravity))
        eager = [
            node for node in tree.body
            if (isinstance(node, ast.Import) and any(alias.name == "sqlite3" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "sqlite3")
        ]
        self.assertEqual(eager, [])

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
