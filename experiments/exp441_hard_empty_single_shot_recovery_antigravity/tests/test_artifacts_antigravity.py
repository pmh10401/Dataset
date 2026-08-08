"""Safe create-only artifact boundary behavior for Exp441."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    canonical_jsonl_bytes_antigravity,
    exclusive_rename_antigravity,
    fsync_directory_antigravity,
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)


class Exp441ArtifactsAntigravityTests(unittest.TestCase):
    def test_canonical_parser_rejects_duplicate_keys_and_noncanonical_bytes(self) -> None:
        """A malformed or differently encoded source must not share a frozen JSON identity."""
        self.assertEqual(parse_canonical_json_bytes_antigravity(b'{"a":2,"b":1}\n'), {"a": 2, "b": 1})
        for payload in (b'{"a":1,"a":2}\n', b'{"b":1,"a":2}\n', b'{"a":2, "b":1}\n'):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_canonical_json_bytes_antigravity(payload)

    def test_canonical_json_and_jsonl_have_compact_sorted_newline_bytes(self) -> None:
        """Whitespace or key-order drift would change the frozen artifact identity."""
        self.assertEqual(canonical_json_bytes_antigravity({"b": 1, "a": 2}), b'{"a":2,"b":1}\n')
        self.assertEqual(canonical_jsonl_bytes_antigravity(({"b": 1}, {"a": 2})), b'{"b":1}\n{"a":2}\n')

    def test_pinned_read_rejects_symlink_and_nonregular_source(self) -> None:
        """A symlink or directory substituted for an input artifact must never be read."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source_antigravity.json"
            source.write_bytes(b"payload")
            link = root / "link_antigravity.json"
            link.symlink_to(source)
            with self.assertRaises((OSError, ValueError)):
                read_pinned_regular_bytes_antigravity(link)
            with self.assertRaises(ValueError):
                read_pinned_regular_bytes_antigravity(root)
            self.assertEqual(read_pinned_regular_bytes_antigravity(source), (b"payload", hashlib.sha256(b"payload").hexdigest()))

    def test_create_only_write_is_no_clobber_and_fully_progress_checked(self) -> None:
        """A pre-existing winner or zero-progress write must not be reported as published."""
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "result_antigravity.json"
            write_create_only_antigravity(destination, b"payload")
            self.assertEqual(destination.read_bytes(), b"payload")
            with self.assertRaises(FileExistsError):
                write_create_only_antigravity(destination, b"other")
            stalled = Path(temporary) / "stalled_antigravity.json"
            with mock.patch("os.write", return_value=0), mock.patch("os.fsync") as fsync:
                with self.assertRaises(OSError) as caught:
                    write_create_only_antigravity(stalled, b"payload")
            self.assertEqual(caught.exception.errno, errno.EIO)
            fsync.assert_not_called()

    def test_directory_fsync_and_exclusive_rename_preserve_existing_winner(self) -> None:
        """A concurrently created destination must survive the Darwin exclusive-rename path."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fsync_directory_antigravity(root)
            source = root / "source_antigravity"
            destination = root / "destination_antigravity"
            source.mkdir()
            destination.mkdir()
            with self.assertRaises(OSError):
                exclusive_rename_antigravity(source, destination)
            self.assertTrue(destination.exists())
