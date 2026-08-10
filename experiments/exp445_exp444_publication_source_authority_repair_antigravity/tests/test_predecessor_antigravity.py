"""Read-only authentication tests for the frozen Exp444 terminal attempt."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


_REAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp444_train_only_stability_precision_gate_antigravity_20260810"
)
_REAL_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")


def _tree_bytes_antigravity(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _canonical_json_antigravity(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


@contextmanager
def _fixture_authority_antigravity():
    """Use a copied terminal fixture; production paths are patched only in tests."""
    from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity

    with tempfile.TemporaryDirectory(prefix="exp445_predecessor_antigravity_") as temporary:
        temporary_root = Path(temporary)
        fixture_root = temporary_root / "exp444_train_only_stability_precision_gate_antigravity_20260810"
        shutil.copytree(_REAL_ROOT_ANTIGRAVITY, fixture_root)
        fixture_database = temporary_root / "univariate_ts.db"
        fixture_database.write_bytes(b"fixture database bytes are never opened")
        with mock.patch.object(predecessor_antigravity, "_EXP444_ROOT_ANTIGRAVITY", fixture_root), mock.patch.object(
            predecessor_antigravity, "_SOURCE_DATABASE_PATH_ANTIGRAVITY", fixture_database
        ):
            yield fixture_root, fixture_database, predecessor_antigravity


class Exp444PredecessorAuthorityAntigravityTests(unittest.TestCase):
    def test_verifies_complete_immutable_terminal_and_never_mutates_fixture_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, database, predecessor):
            before = _tree_bytes_antigravity(root)
            database_before = hashlib.sha256(database.read_bytes()).hexdigest()
            original_open = predecessor.os.open
            observed_flags: list[int] = []

            def read_only_open(path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
                observed_flags.append(flags)
                forbidden = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                if flags & forbidden:
                    raise AssertionError("predecessor verifier attempted a write-capable open")
                if dir_fd is None:
                    return original_open(path, flags, mode)
                return original_open(path, flags, mode, dir_fd=dir_fd)

            with mock.patch.object(predecessor.os, "open", side_effect=read_only_open):
                with mock.patch.object(predecessor.os, "rename", side_effect=AssertionError("rename is forbidden")), mock.patch.object(
                    predecessor.os, "replace", side_effect=AssertionError("replace is forbidden")
                ), mock.patch.object(predecessor.os, "unlink", side_effect=AssertionError("unlink is forbidden")):
                    first = predecessor.verify_exp444_predecessor_antigravity()
                    second = predecessor.verify_exp444_predecessor_antigravity()

            self.assertEqual(first, second)
            self.assertEqual(first.attempt_root, root / "attempts_antigravity/attempt_0000_antigravity")
            self.assertEqual(len(first.record_sha256_prefix), 30)
            self.assertTrue(observed_flags)
            self.assertEqual(_tree_bytes_antigravity(root), before)
            self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), database_before)

    def test_rejects_semantic_state_prefix_and_aggregate_drift_antigravity(self) -> None:
        mutations = (
            ("nonterminal", lambda state: state.update(status="running")),
            ("error", lambda state: state.update(error_count=1, error="preserved failure")),
            ("short_prefix", lambda state: state.update(record_sha256_prefix=state["record_sha256_prefix"][:-1])),
            ("reordered_prefix", lambda state: state.update(record_sha256_prefix=list(reversed(state["record_sha256_prefix"])))),
            ("aggregate", lambda state: state.update(payload_sha256="0" * 64)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), _fixture_authority_antigravity() as (root, _, predecessor):
                state_path = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/state_antigravity.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                mutate(state)
                state_bytes = _canonical_json_antigravity(state)
                state_path.write_bytes(state_bytes)
                before_failure = _tree_bytes_antigravity(root)
                hashes = dict(predecessor.EXP444_TERMINAL_SHA256_ANTIGRAVITY)
                hashes["checkpoint/state_antigravity.json"] = hashlib.sha256(state_bytes).hexdigest()
                with mock.patch.object(predecessor, "EXP444_TERMINAL_SHA256_ANTIGRAVITY", hashes):
                    with self.assertRaises(ValueError):
                        predecessor.verify_exp444_predecessor_antigravity()
                self.assertEqual(_tree_bytes_antigravity(root), before_failure)

    def test_rejects_terminal_state_context_and_runtime_binding_drift_antigravity(self) -> None:
        mutations = (
            ("checkpoint_antigravity/context_antigravity.json", b"x"),
            ("checkpoint_antigravity/state_antigravity.json", b"x"),
            ("progress_antigravity.json", b"x"),
            ("pid_antigravity.json", b"x"),
            ("events_antigravity.jsonl", b"x"),
            ("text_antigravity.log", b"x"),
        )
        for relative, replacement in mutations:
            with self.subTest(relative=relative), _fixture_authority_antigravity() as (root, _, predecessor):
                target = root / "attempts_antigravity/attempt_0000_antigravity" / relative
                target.write_bytes(replacement)
                with self.assertRaises(ValueError):
                    predecessor.verify_exp444_predecessor_antigravity()

    def test_rejects_semantic_binding_drift_even_when_fixture_hash_expectations_are_patched_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, _, predecessor):
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            context_path = attempt / "checkpoint_antigravity/context_antigravity.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["outer_workers"] = 10
            context_bytes = _canonical_json_antigravity(context)
            context_path.write_bytes(context_bytes)
            hashes = dict(predecessor.EXP444_TERMINAL_SHA256_ANTIGRAVITY)
            hashes["checkpoint/context_antigravity.json"] = hashlib.sha256(context_bytes).hexdigest()
            with mock.patch.object(predecessor, "EXP444_TERMINAL_SHA256_ANTIGRAVITY", hashes):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp444_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            pid_path = attempt / "pid_antigravity.json"
            pid = json.loads(pid_path.read_text(encoding="utf-8"))
            pid["pid"] = os.getpid()
            pid_bytes = _canonical_json_antigravity(pid)
            pid_path.write_bytes(pid_bytes)
            hashes = dict(predecessor.EXP444_TERMINAL_SHA256_ANTIGRAVITY)
            hashes["pid_antigravity.json"] = hashlib.sha256(pid_bytes).hexdigest()
            with mock.patch.object(predecessor, "EXP444_TERMINAL_SHA256_ANTIGRAVITY", hashes):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp444_predecessor_antigravity()

    def test_rejects_nonregular_hardlinked_or_replaced_authority_leaves_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, _, predecessor):
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            record = attempt / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            hardlink = root.parent / "copy_record_antigravity.json"
            os.link(record, hardlink)
            with self.assertRaises(ValueError):
                predecessor.verify_exp444_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            record = attempt / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            saved = root.parent / "saved_record_antigravity.json"
            shutil.copyfile(record, saved)
            record.unlink()
            record.symlink_to(saved)
            with self.assertRaises(ValueError):
                predecessor.verify_exp444_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            record = attempt / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            replacement = root.parent / "replacement_record_antigravity.json"
            shutil.copyfile(record, replacement)
            original_read = predecessor.os.read
            replaced = False

            def replace_after_read(fd: int, size: int) -> bytes:
                nonlocal replaced
                payload = original_read(fd, size)
                if not replaced and os.fstat(fd).st_ino == record.stat().st_ino:
                    os.replace(replacement, record)
                    replaced = True
                return payload

            with mock.patch.object(predecessor.os, "read", side_effect=replace_after_read):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp444_predecessor_antigravity()
            self.assertTrue(replaced)

        with _fixture_authority_antigravity() as (root, _, predecessor):
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            record = attempt / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            original = record.read_bytes()
            writer = os.open(record, os.O_WRONLY)
            original_read = predecessor.os.read
            overwritten = False
            after_overwrite: dict[str, str] | None = None

            def overwrite_after_read(fd: int, size: int) -> bytes:
                nonlocal after_overwrite, overwritten
                payload = original_read(fd, size)
                if not overwritten and os.fstat(fd).st_ino == record.stat().st_ino:
                    os.lseek(writer, 0, os.SEEK_SET)
                    os.write(writer, b"X" + original[1:])
                    os.fsync(writer)
                    overwritten = True
                    after_overwrite = _tree_bytes_antigravity(root)
                return payload

            try:
                with mock.patch.object(predecessor.os, "read", side_effect=overwrite_after_read):
                    with self.assertRaises(ValueError):
                        predecessor.verify_exp444_predecessor_antigravity()
            finally:
                os.close(writer)
            self.assertTrue(overwritten)
            self.assertIsNotNone(after_overwrite)
            self.assertEqual(_tree_bytes_antigravity(root), after_overwrite)

    def test_rejects_inventory_prefix_lifecycle_publication_and_database_drift_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, database, predecessor):
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            records = attempt / "checkpoint_antigravity/records_antigravity"
            (records / "0030_record_antigravity.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                predecessor.verify_exp444_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            (root / "publication_antigravity").mkdir()
            with self.assertRaises(ValueError):
                predecessor.verify_exp444_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, _, predecessor):
            (root / "evaluation_antigravity").mkdir()
            with self.assertRaises(ValueError):
                predecessor.verify_exp444_predecessor_antigravity()

        with _fixture_authority_antigravity() as (root, database, predecessor):
            os.link(database, database.with_name("linked_database_antigravity.db"))
            with self.assertRaises(ValueError):
                predecessor.verify_exp444_predecessor_antigravity()

    def test_rechecks_database_link_count_immediately_before_return_antigravity(self) -> None:
        with _fixture_authority_antigravity() as (root, database, predecessor):
            original_check = predecessor._assert_source_database_antigravity
            calls = 0

            def link_before_final_check() -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    os.link(database, database.with_name("late_database_link_antigravity.db"))
                original_check()

            before_failure = _tree_bytes_antigravity(root)
            with mock.patch.object(predecessor, "_assert_source_database_antigravity", side_effect=link_before_final_check):
                with self.assertRaises(ValueError):
                    predecessor.verify_exp444_predecessor_antigravity()
            self.assertEqual(calls, 2)
            self.assertEqual(_tree_bytes_antigravity(root), before_failure)
