"""Read-only authentication tests for the immutable Exp445 failed publication."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


_REAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
)


def _tree_bytes_antigravity(root: Path) -> dict[str, str]:
    return {
        item.relative_to(root).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(root.rglob("*"))
        if item.is_file() and not item.is_symlink()
    }


def _canonical_json_antigravity(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"


@contextmanager
def _fixture_authority_antigravity():
    predecessor = importlib.import_module(
        "experiments.exp446_exp445_publication_materialization_repair_antigravity.predecessor_antigravity"
    )
    with tempfile.TemporaryDirectory(prefix="exp446_exp445_predecessor_antigravity_") as temporary:
        temporary_root = Path(temporary)
        fixture_root = temporary_root / _REAL_ROOT_ANTIGRAVITY.name
        shutil.copytree(_REAL_ROOT_ANTIGRAVITY, fixture_root)
        database = temporary_root / "univariate_ts.db"
        database.write_bytes(b"fixture database bytes are never opened")
        with mock.patch.object(predecessor, "_EXP445_ROOT_ANTIGRAVITY", fixture_root), mock.patch.object(
            predecessor, "_SOURCE_DATABASE_PATH_ANTIGRAVITY", database
        ):
            yield fixture_root, database, predecessor


class Exp445PredecessorAuthorityAntigravityTests(unittest.TestCase):
    def test_profile_and_predecessor_modules_are_provided_antigravity(self) -> None:
        profile = importlib.import_module(
            "experiments.exp446_exp445_publication_materialization_repair_antigravity.profile_antigravity"
        )
        predecessor = importlib.import_module(
            "experiments.exp446_exp445_publication_materialization_repair_antigravity.predecessor_antigravity"
        )
        self.assertTrue(profile.EXP445_ROOT_ANTIGRAVITY)
        self.assertTrue(callable(predecessor.verify_exp445_predecessor_antigravity))

    def test_returns_values_only_deeply_immutable_authority_without_mutation_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, database, predecessor):
            before = _tree_bytes_antigravity(root)
            database_before = hashlib.sha256(database.read_bytes()).hexdigest()
            original_open = predecessor.os.open
            flags: list[int] = []

            def read_only_open(path: object, open_flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
                flags.append(open_flags)
                forbidden = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                if open_flags & forbidden:
                    raise AssertionError("verifier attempted write-capable open")
                if dir_fd is None:
                    return original_open(path, open_flags, mode)
                return original_open(path, open_flags, mode, dir_fd=dir_fd)

            with mock.patch.object(predecessor.os, "open", side_effect=read_only_open), mock.patch.object(
                predecessor.os, "rename", side_effect=AssertionError("rename is forbidden")
            ), mock.patch.object(predecessor.os, "replace", side_effect=AssertionError("replace is forbidden")), mock.patch.object(
                predecessor.os, "unlink", side_effect=AssertionError("unlink is forbidden")
            ):
                first = predecessor.verify_exp445_predecessor_antigravity()
                second = predecessor.verify_exp445_predecessor_antigravity()

            self.assertEqual(first, second)
            self.assertEqual(len(first.record_sha256_prefix), 30)
            self.assertEqual(first.record_prefix_payload_sha256, predecessor.EXP445_RECORD_PREFIX_PAYLOAD_SHA256_ANTIGRAVITY)
            self.assertFalse(hasattr(first, "__dict__"))
            self.assertFalse(any(isinstance(value, (Path, int)) for value in (
                first.reviewed_commit, first.record_sha256_prefix, first.record_prefix_payload_sha256, first.terminal_sha256,
            )))
            with self.assertRaises(AttributeError):
                first.__dict__["reviewed_commit"] = "changed"  # type: ignore[attr-defined]
            with self.assertRaises((AttributeError, TypeError)):
                first.terminal_sha256["x"] = "x"  # type: ignore[index]
            with self.assertRaises((AttributeError, TypeError)):
                first.record_sha256_prefix += ("x",)
            self.assertTrue(flags)
            self.assertEqual(_tree_bytes_antigravity(root), before)
            self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), database_before)

    def test_rejects_symlink_hardlink_rename_and_same_size_overwrite_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, _, predecessor):
            record = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            saved = root.parent / "saved_record_antigravity.json"
            shutil.copyfile(record, saved)
            record.unlink()
            record.symlink_to(saved)
            with self.assertRaises(ValueError):
                predecessor.verify_exp445_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            record = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            os.link(record, root.parent / "linked_record_antigravity.json")
            with self.assertRaises(ValueError):
                predecessor.verify_exp445_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            record = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            replacement = root.parent / "replacement_record_antigravity.json"
            shutil.copyfile(record, replacement)
            original_read = predecessor.os.read
            changed = False

            def rename_after_read(fd: int, size: int) -> bytes:
                nonlocal changed
                payload = original_read(fd, size)
                if not changed and os.fstat(fd).st_ino == record.stat().st_ino:
                    os.replace(replacement, record)
                    changed = True
                return payload

            with mock.patch.object(predecessor.os, "read", side_effect=rename_after_read):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp445_predecessor_antigravity()
            self.assertTrue(changed)

    def test_rejects_cross_leaf_same_size_overwrite_after_record_zero_is_read_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, _, predecessor):
            records = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity"
            first, second = records / "0000_record_antigravity.json", records / "0001_record_antigravity.json"
            original = first.read_bytes()
            writer = os.open(first, os.O_WRONLY)
            original_read = predecessor.os.read
            changed = False

            def overwrite_first_while_second_is_read(fd: int, size: int) -> bytes:
                nonlocal changed
                payload = original_read(fd, size)
                if not changed and os.fstat(fd).st_ino == second.stat().st_ino:
                    os.lseek(writer, 0, os.SEEK_SET)
                    os.write(writer, b"X" + original[1:])
                    os.fsync(writer)
                    changed = True
                return payload

            try:
                with mock.patch.object(predecessor.os, "read", side_effect=overwrite_first_while_second_is_read):
                    with self.assertRaises(ValueError):
                        predecessor.verify_exp445_predecessor_antigravity()
            finally:
                os.close(writer)
            self.assertTrue(changed)

    def test_rejects_source_database_replacement_during_verification_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, database, predecessor):
            replacement = database.with_name("replacement_database_antigravity.db")
            replacement.write_bytes(b"replacement database remains single-linked")
            record = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            original_read = predecessor.os.read
            replaced = False

            def replace_database_after_first_record_read(fd: int, size: int) -> bytes:
                nonlocal replaced
                payload = original_read(fd, size)
                if not replaced and os.fstat(fd).st_ino == record.stat().st_ino:
                    os.replace(replacement, database)
                    replaced = True
                return payload

            with mock.patch.object(predecessor.os, "read", side_effect=replace_database_after_first_record_read):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp445_predecessor_antigravity()
            self.assertTrue(replaced)

        with _fixture_authority_antigravity() as (root, _, predecessor):
            record = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            original = record.read_bytes()
            writer = os.open(record, os.O_WRONLY)
            original_read = predecessor.os.read
            changed = False

            def overwrite_after_read(fd: int, size: int) -> bytes:
                nonlocal changed
                payload = original_read(fd, size)
                if not changed and os.fstat(fd).st_ino == record.stat().st_ino:
                    os.lseek(writer, 0, os.SEEK_SET)
                    os.write(writer, b"X" + original[1:])
                    os.fsync(writer)
                    changed = True
                return payload

            try:
                with mock.patch.object(predecessor.os, "read", side_effect=overwrite_after_read):
                    with self.assertRaises(ValueError):
                        predecessor.verify_exp445_predecessor_antigravity()
            finally:
                os.close(writer)
            self.assertTrue(changed)

    def test_rejects_no_follow_root_attempt_records_and_exact_inventory_drift_antigravity(self) -> None:
        for relative in (
            ".",
            "attempts_antigravity/attempt_0000_antigravity",
            "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity",
        ):
            with self.subTest(relative=relative), _fixture_authority_antigravity() as (root, _, predecessor):
                target = root if relative == "." else root / relative
                saved = root.parent / f"saved_{target.name}_antigravity"
                target.rename(saved)
                target.symlink_to(saved, target_is_directory=True)
                with self.assertRaises(ValueError):
                    predecessor.verify_exp445_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            records = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity"
            (records / "0030_record_antigravity.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                predecessor.verify_exp445_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            records = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity"
            (records / "0000_record_antigravity.json").unlink()
            with self.assertRaises(ValueError):
                predecessor.verify_exp445_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            records = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity"
            first, second = records / "0000_record_antigravity.json", records / "0001_record_antigravity.json"
            temporary = records / "swap_antigravity.json"
            first.rename(temporary)
            second.rename(first)
            temporary.rename(second)
            with self.assertRaises(ValueError):
                predecessor.verify_exp445_predecessor_antigravity()

    def test_rejects_changed_failure_live_pid_publication_evaluation_and_database_link_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, _, predecessor):
            failure = root / "attempts_antigravity/attempt_0000_antigravity/publication_failure_antigravity.json"
            payload = json.loads(failure.read_text())
            payload["error"]["message"] = "changed"
            changed = _canonical_json_antigravity(payload)
            failure.write_bytes(changed)
            hashes = dict(predecessor.EXP445_TERMINAL_SHA256_ANTIGRAVITY)
            hashes["publication_failure_antigravity.json"] = hashlib.sha256(changed).hexdigest()
            with mock.patch.object(predecessor, "EXP445_TERMINAL_SHA256_ANTIGRAVITY", hashes):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp445_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            pid = root / "attempts_antigravity/attempt_0000_antigravity/pid_antigravity.json"
            payload = json.loads(pid.read_text())
            payload.update(active=True, pid=os.getpid())
            changed = _canonical_json_antigravity(payload)
            pid.write_bytes(changed)
            hashes = dict(predecessor.EXP445_TERMINAL_SHA256_ANTIGRAVITY)
            hashes["pid_antigravity.json"] = hashlib.sha256(changed).hexdigest()
            with mock.patch.object(predecessor, "EXP445_TERMINAL_SHA256_ANTIGRAVITY", hashes):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp445_predecessor_antigravity()

        for directory in ("publication_antigravity", "evaluation_antigravity"):
            with self.subTest(directory=directory), _fixture_authority_antigravity() as (root, _, predecessor):
                (root / directory).mkdir()
                with self.assertRaises(ValueError):
                    predecessor.verify_exp445_predecessor_antigravity()

        with _fixture_authority_antigravity() as (_, database, predecessor):
            os.link(database, database.with_name("linked_database_antigravity.db"))
            with self.assertRaises(ValueError):
                predecessor.verify_exp445_predecessor_antigravity()
