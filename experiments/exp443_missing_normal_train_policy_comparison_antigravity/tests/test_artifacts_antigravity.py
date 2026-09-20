"""RED tests for Exp443's narrow artifact wrappers."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    exclusive_rename_antigravity,
    fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)


class Exp443ArtifactsAntigravityTests(unittest.TestCase):
    def test_canonical_json_and_jsonl_round_trip_without_whitespace_or_key_drift(self) -> None:
        self.assertEqual(canonical_json_bytes_antigravity({"b": 1, "a": 2}), b'{"a":2,"b":1}\n')
        self.assertEqual(canonical_jsonl_bytes_antigravity(({"b": 1}, {"a": 2})), b'{"b":1}\n{"a":2}\n')
        self.assertEqual(parse_canonical_json_bytes_antigravity(b'{"a":2,"b":1}\n'), {"a": 2, "b": 1})
        with self.assertRaises(ValueError):
            parse_canonical_json_bytes_antigravity(b'{"a":1,"a":2}\n')

    def test_pinned_regular_read_rejects_symlink_and_returns_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source_antigravity.json"
            source.write_bytes(b"payload")
            link = root / "link_antigravity.json"
            link.symlink_to(source)
            with self.assertRaises((OSError, ValueError)):
                read_pinned_regular_bytes_antigravity(link)
            self.assertEqual(read_pinned_regular_bytes_antigravity(source), (b"payload", hashlib.sha256(b"payload").hexdigest()))

    def test_create_only_write_is_private_fsyncing_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result_antigravity.json"
            write_create_only_antigravity(destination, b"payload")
            self.assertEqual(destination.read_bytes(), b"payload")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_create_only_antigravity(destination, b"other")

    def test_directory_fsync_and_exclusive_rename_never_replace_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fsync_directory_antigravity(root)
            source = root / "source_antigravity"
            destination = root / "destination_antigravity"
            source.mkdir()
            destination.mkdir()
            with self.assertRaises(OSError):
                exclusive_rename_antigravity(source, destination)
            self.assertTrue(source.exists())
            self.assertTrue(destination.exists())


if __name__ == "__main__":
    unittest.main()
