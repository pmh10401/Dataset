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
    pin_exact_regular_inventory_antigravity,
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
    write_create_only_antigravity,
)


class Exp441ArtifactsAntigravityTests(unittest.TestCase):
    def test_grouped_pin_rejects_first_file_mutation_during_later_file_read(self) -> None:
        """A byte change after the first descriptor reaches EOF cannot yield a mixed six-file snapshot."""
        names = (
            "source_snapshot_antigravity.json", "selection_antigravity.json", "progress_antigravity.json",
            "pid_antigravity", "events_antigravity.jsonl", "text_antigravity.log",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt_antigravity"; root.mkdir(); os.chmod(root, 0o700)
            for index, name in enumerate(names):
                (root / name).write_bytes(f"file-{index}\n".encode()); os.chmod(root / name, 0o600)
            original_read = os.read; reads = {"count": 0}
            def mutate_after_first_eof(descriptor: int, size: int) -> bytes:
                payload = original_read(descriptor, size)
                reads["count"] += 1
                if reads["count"] == 3:
                    (root / names[0]).write_bytes(b"FILE-0\n")
                return payload
            with mock.patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity.os.read", side_effect=mutate_after_first_eof):
                with self.assertRaisesRegex(ValueError, "changed"):
                    pin_exact_regular_inventory_antigravity(root, names)

    def test_grouped_pin_rejects_replacement_unlink_chmod_and_truncate_with_descriptor_cleanup(self) -> None:
        """Namespace and in-place mutations after the first read fail closed and close every opened descriptor."""
        names = (
            "source_snapshot_antigravity.json", "selection_antigravity.json", "progress_antigravity.json",
            "pid_antigravity", "events_antigravity.jsonl", "text_antigravity.log",
        )
        for mutation in ("replace", "unlink", "chmod", "truncate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "attempt_antigravity"; root.mkdir(); os.chmod(root, 0o700)
                for index, name in enumerate(names):
                    (root / name).write_bytes(f"file-{index}\n".encode()); os.chmod(root / name, 0o600)
                target = root / names[0]; real_read = os.read; real_close = os.close; reads = {"count": 0}; closed: list[int] = []
                def mutate_after_first_eof(descriptor: int, size: int) -> bytes:
                    payload = real_read(descriptor, size); reads["count"] += 1
                    if reads["count"] == 3:
                        if mutation == "replace":
                            replacement = root / "replacement_antigravity.tmp"; replacement.write_bytes(b"FILE-0\n"); os.chmod(replacement, 0o600); os.replace(replacement, target)
                        elif mutation == "unlink":
                            target.unlink()
                        elif mutation == "chmod":
                            os.chmod(target, 0o644)
                        else:
                            target.write_bytes(b"")
                    return payload
                def record_close(descriptor: int) -> None:
                    closed.append(descriptor); real_close(descriptor)
                with mock.patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity.os.read", side_effect=mutate_after_first_eof), \
                     mock.patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity.os.close", side_effect=record_close):
                    with self.assertRaises(ValueError):
                        pin_exact_regular_inventory_antigravity(root, names)
                self.assertEqual(len(closed), len(names) + 1)

    def test_retained_grouped_pin_revalidates_post_read_mutation_and_closes_root_fd(self) -> None:
        """A change after initial grouped validation remains detectable before a caller consumes the bytes."""
        names = ("source_snapshot_antigravity.json", "selection_antigravity.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt_antigravity"; root.mkdir(); os.chmod(root, 0o700)
            for index, name in enumerate(names):
                (root / name).write_bytes(f"file-{index}\n".encode()); os.chmod(root / name, 0o600)
            with pin_exact_regular_inventory_antigravity(root, names) as pinned:
                root_descriptor = pinned._root_descriptor_antigravity
                (root / names[0]).write_bytes(b"FILE-0\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    pinned.revalidate_antigravity()
            with self.assertRaises(OSError):
                os.fstat(root_descriptor)

    def test_retained_grouped_pin_rejects_root_replacement_with_same_inventory(self) -> None:
        """Replacing the 0700 attempt directory with an identical-name sibling cannot redirect a retained authority."""
        names = ("source_snapshot_antigravity.json", "selection_antigravity.json")
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = parent / "attempt_antigravity"; replacement = parent / "replacement_antigravity"; root.mkdir(); replacement.mkdir()
            for directory in (root, replacement):
                os.chmod(directory, 0o700)
                for index, name in enumerate(names): (directory / name).write_bytes(f"file-{index}\n".encode()); os.chmod(directory / name, 0o600)
            with pin_exact_regular_inventory_antigravity(root, names) as pinned:
                root.rename(parent / "original_attempt_antigravity"); replacement.rename(root)
                with self.assertRaisesRegex(ValueError, "root changed"):
                    pinned.revalidate_antigravity()

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

    def test_create_only_write_uses_exact_flags_loops_partial_writes_and_fsyncs(self) -> None:
        """A short successful write must be completed before one durable successful publication."""
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "partial_antigravity.json"
            real_open, real_write, real_fsync = os.open, os.write, os.fsync
            writes: list[int] = []

            def partial_write(descriptor: int, payload: bytes) -> int:
                writes.append(len(payload))
                return real_write(descriptor, payload[:2] if len(writes) == 1 else payload)

            with mock.patch("os.open", wraps=real_open) as opened, mock.patch("os.write", side_effect=partial_write), mock.patch("os.fsync", wraps=real_fsync) as fsync:
                write_create_only_antigravity(destination, b"payload")
            self.assertEqual(destination.read_bytes(), b"payload")
            self.assertGreaterEqual(len(writes), 2)
            self.assertEqual(opened.call_args.args[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW)
            self.assertEqual(opened.call_args.args[2], 0o600)
            fsync.assert_called_once()

    def test_exclusive_rename_rejects_destination_created_after_staging(self) -> None:
        """A destination that appears after staging begins remains the concurrent winner."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "staging_antigravity"
            destination = root / "winner_antigravity"
            source.mkdir()
            destination.mkdir()
            with self.assertRaises(OSError):
                exclusive_rename_antigravity(source, destination)
            self.assertTrue(source.exists())
            self.assertTrue(destination.exists())

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
