"""One-hash Exp425 database snapshot tests."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import shutil
import sqlite3
import struct
from tempfile import TemporaryDirectory
import sys
import time
import unittest
from unittest.mock import patch

from experiments.exp425_selective_recovery import database as snapshot_database
from experiments.exp425_selective_recovery.database import open_database_snapshot
from experiments.exp417_causal_multihorizon_shadow.config import Exp417FrozenConfig, FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit, FrozenLaneSnapshot, FrozenRunContext, FrozenTrainContract, GroupQualification


class DatabaseSnapshotTests(unittest.TestCase):
    def test_snapshot_module_exposes_context_manager(self) -> None:
        self.assertTrue(callable(open_database_snapshot))

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / "input.sqlite"
        connection = sqlite3.connect(self.database)
        try:
            connection.executescript("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE instances (dataset_id INTEGER, instance_index INTEGER, split TEXT, label TEXT, values_blob BLOB);")
            values = struct.pack("<2f", 1.0, 2.0)
            connection.executemany("INSERT INTO datasets VALUES (?, ?)", ((1, "A"), (2, "B")))
            connection.executemany("INSERT INTO instances VALUES (?, ?, ?, ?, ?)", ((1, 0, "TRAIN", "0", values), (1, 0, "TEST", "hidden", values), (2, 0, "TRAIN", "0", values), (2, 0, "TEST", "hidden", values)))
            connection.commit()
        finally:
            connection.close()
        self.digest = sha256(self.database.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_retained_descriptor_hashes_once_across_boundaries_and_close(self) -> None:
        with patch.object(snapshot_database, "_sha256_descriptor", wraps=snapshot_database._sha256_descriptor) as hashed:
            manager = open_database_snapshot(self.database, expected_sha256=self.digest)
            snapshot = manager.__enter__()
            snapshot.verify_boundary("first")
            snapshot.verify_boundary("second")
            snapshot.verify_boundary("third")
            manager.__exit__(None, None, None)
        self.assertEqual(hashed.call_count, 1)
        self.assertEqual(snapshot.audit.full_hash_count, 1)
        self.assertEqual(snapshot.audit.operational_nlink, 2 if sys.platform == "darwin" else 1)
        self.assertEqual(snapshot.audit.boundaries, ("open:accepted", "first", "second", "third", "close:before"))

    def test_boundary_rejects_path_replacement_without_rehashing(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        replacement = self.root / "replacement.sqlite"
        shutil.copyfile(self.database, replacement)
        os.replace(replacement, self.database)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            snapshot.verify_boundary("replacement")
        with self.assertRaises(ValueError):
            manager.__exit__(None, None, None)

    def test_sidecars_and_symlink_inputs_fail_closed(self) -> None:
        (self.root / "input.sqlite-wal").write_bytes(b"writer")
        with self.assertRaisesRegex(ValueError, "sidecar"):
            with open_database_snapshot(self.database, expected_sha256=self.digest):
                pass
        (self.root / "input.sqlite-wal").unlink()
        alias = self.root / "alias.sqlite"
        alias.symlink_to(self.database)
        with self.assertRaisesRegex(ValueError, "symlink"):
            with open_database_snapshot(alias, expected_sha256=self.digest):
                pass

    def test_every_preexisting_sqlite_artifact_fails_closed(self) -> None:
        for suffix in ("-wal", "-shm", "-journal", "-mj00000001"):
            with self.subTest(suffix=suffix):
                artifact = self.root / ("input.sqlite" + suffix)
                artifact.write_bytes(b"writer")
                with self.assertRaisesRegex(ValueError, "sidecar|unknown relevant"):
                    with open_database_snapshot(self.database, expected_sha256=self.digest):
                        pass
                artifact.unlink()

    def test_persistent_wal_mode_without_sidecars_fails_before_acceptance(self) -> None:
        connection = sqlite3.connect(self.database)
        try:
            self.assertEqual(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
        for suffix in ("-wal", "-shm"):
            artifact = self.root / ("input.sqlite" + suffix)
            if artifact.exists():
                artifact.unlink()
        self.assertEqual(self.database.read_bytes()[18:20], b"\x02\x02")
        digest = sha256(self.database.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "persistent WAL mode"):
            with open_database_snapshot(self.database, expected_sha256=digest):
                pass

    def test_close_and_invalid_digest_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA256"):
            with open_database_snapshot(self.database, expected_sha256="0" * 64):
                pass
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        manager.__exit__(None, None, None)
        with self.assertRaisesRegex(ValueError, "closed"):
            snapshot.verify_boundary("after-close")

    def test_hardlink_alias_is_rejected_before_and_after_acceptance(self) -> None:
        alias = self.root / "preexisting-alias.sqlite"
        os.link(self.database, alias)
        with self.assertRaisesRegex(ValueError, "unaliased"):
            with open_database_snapshot(self.database, expected_sha256=self.digest):
                pass
        alias.unlink()
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        os.link(self.database, alias)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            snapshot.verify_boundary("alias")
        with self.assertRaises(ValueError):
            manager.__exit__(None, None, None)

    def test_boundary_rejects_parent_namespace_replacement(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        moved = self.root.with_name(self.root.name + "-moved")
        self.root.rename(moved)
        self.root.mkdir()
        with self.assertRaisesRegex(ValueError, "parent (metadata|namespace)"):
            snapshot.verify_boundary("parent-replaced")
        with self.assertRaises(ValueError):
            manager.__exit__(None, None, None)
        shutil.rmtree(moved)

    def test_retained_sqlite_read_lock_blocks_a_journal_writer(self) -> None:
        """A competing SQLite writer cannot commit while the snapshot is live."""
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        writer = subprocess.run(
            ("python3", "-c", "import sqlite3, sys; c=sqlite3.connect(sys.argv[1], timeout=0); c.execute('INSERT INTO datasets VALUES (3, \\'writer\\')'); c.commit()", str(self.database)),
            capture_output=True, text=True,
        )
        self.assertNotEqual(writer.returncode, 0, writer.stderr)
        with self.assertRaisesRegex(ValueError, "parent metadata|sidecar"):
            snapshot.verify_boundary("writer-attempt")
        with self.assertRaises(ValueError):
            manager.__exit__(None, None, None)

    def test_wal_writer_is_blocked_and_loader_after_boundary_rejects_wal(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        writer = subprocess.run(
            ("python3", "-c", "import sqlite3, sys; c=sqlite3.connect(sys.argv[1], timeout=0); c.execute('PRAGMA journal_mode=WAL'); c.execute(\"INSERT INTO datasets VALUES (3, 'writer')\"); c.commit()", str(self.database)),
            capture_output=True, text=True,
        )
        self.assertNotEqual(writer.returncode, 0, writer.stderr)
        self.assertFalse((self.root / "input.sqlite-wal").exists())
        self.assertFalse((self.root / "input.sqlite-shm").exists())
        manager.__exit__(None, None, None)
        self.tearDown(); self.setUp()
        config = self._synthetic_config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config):
            loader_manager = open_database_snapshot(self.database, expected_sha256=self.digest)
            snapshot = loader_manager.__enter__()
            context = snapshot.freeze_context(self._lanes(config), verified_code_commit="a" * 40)
            original = snapshot_database.exp417_inputs.load_raw_train_values_from_connection

            def insert_wal_then_decode(*args, **kwargs):
                (self.root / "input.sqlite-wal").write_bytes(b"late writer")
                return original(*args, **kwargs)

            with patch.object(snapshot_database.exp417_inputs, "load_raw_train_values_from_connection", side_effect=insert_wal_then_decode):
                with self.assertRaisesRegex(ValueError, "parent metadata|sidecar"):
                    snapshot.load_train("A", context)
            self.assertEqual(snapshot.audit.boundaries[-2:], ("load_train:before", "load_train:after"))
            with self.assertRaises(ValueError):
                loader_manager.__exit__(None, None, None)

    def test_connect_uses_retained_descriptor_during_path_aba(self) -> None:
        """SQLite is never pointed at the published or owned pathname."""
        real_connect = snapshot_database.sqlite3.connect
        seen: list[str] = []

        def replace_published_during_connect(*args, **kwargs):
            uri = args[0] if args else kwargs["database"]
            seen.append(uri)
            replacement = self.root / "connect-replacement.sqlite"
            shutil.copyfile(self.database, replacement)
            os.replace(replacement, self.database)
            if sys.platform == "darwin":
                owned = next(self.root.glob(".exp425-snapshot-*/database.sqlite"))
                owned_replacement = owned.with_name("connect-owned-replacement.sqlite")
                shutil.copyfile(self.database, owned_replacement)
                os.replace(owned_replacement, owned)
            return real_connect(*args, **kwargs)

        with patch.object(snapshot_database.sqlite3, "connect", side_effect=replace_published_during_connect):
            with self.assertRaises(ValueError):
                with open_database_snapshot(self.database, expected_sha256=self.digest):
                    pass
        self.assertEqual(len(seen), 1)
        self.assertIn("/dev/fd/", seen[0])

    def _contract(self, dataset: str) -> FrozenTrainContract:
        return FrozenTrainContract(dataset, (GroupQualification(2, 100, "qualified", 0.01, 0.05, 0.70, {"level_shift": .50, "gradual_drift": .50, "amplitude_scaling": .50, "period_change": .50}, True),), BackendAudit("cpu", True, 0.0, 1.0, 1.0, False), "a" * 64)

    def _synthetic_config(self):
        return replace(FROZEN_CONFIG, expected_datasets=2, expected_hard=0, expected_standard=0, expected_priority=0, expected_automatic_normal=2, database_sha256=self.digest)

    @staticmethod
    def _lanes(config):
        return FrozenLaneSnapshot({}, {}, {}, {"A": (0,), "B": (0,)}, config=config, population=("A", "B"))

    def test_one_hash_context_two_train_two_test_and_three_boundaries(self) -> None:
        """The exact Exp417 decoders run on one immutable descriptor snapshot."""
        config = self._synthetic_config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config), patch.object(snapshot_database, "_sha256_descriptor", wraps=snapshot_database._sha256_descriptor) as hashed:
            with open_database_snapshot(self.database, expected_sha256=self.digest) as snapshot:
                context = snapshot.freeze_context(self._lanes(config), verified_code_commit="a" * 40)
                equal_but_foreign = FrozenRunContext(context.database_sha256, context.datasets, context.lanes, context.test_identity_manifest, context.code_commit, context.runtime_versions, config=config)
                with self.assertRaisesRegex(ValueError, "not this snapshot"):
                    snapshot.load_train("A", equal_but_foreign)
                with self.assertRaisesRegex(ValueError, "already frozen"):
                    snapshot.freeze_context(self._lanes(config), verified_code_commit="a" * 40)
                trains = tuple(snapshot.load_train(dataset, context) for dataset in context.datasets)
                tests = tuple(snapshot.load_test(dataset, self._contract(dataset), context) for dataset in context.datasets)
                for name in ("explicit:one", "explicit:two", "explicit:three"):
                    snapshot.verify_boundary(name)
                self.assertEqual(context.datasets, ("A", "B"))
                self.assertEqual(dict(context.test_identity_manifest), {"A": ((0, 2),), "B": ((0, 2),)})
                self.assertEqual(tuple(item.series[0].values for item in trains), ((1.0, 2.0), (1.0, 2.0)))
                self.assertEqual(tuple(item.series[0].instance_index for item in tests), (0, 0))
            self.assertEqual(hashed.call_count, 1)
            self.assertEqual(snapshot.audit.full_hash_count, 1)
            self.assertEqual(snapshot.audit.lifecycle, "closed")

    def test_context_rejects_nonfrozen_config_and_nonexact_population(self) -> None:
        config = self._synthetic_config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config), open_database_snapshot(self.database, expected_sha256=self.digest) as snapshot:
            with self.assertRaisesRegex(ValueError, "frozen lanes"):
                snapshot.freeze_context(self._lanes(replace(config, candidate_alpha=.02)), verified_code_commit="a" * 40)
            partial = FrozenLaneSnapshot({}, {}, {}, {"A": (0,), "C": (0,)}, config=config, population=("A", "C"))
            with self.assertRaisesRegex(ValueError, "outside the database"):
                snapshot.freeze_context(partial, verified_code_commit="a" * 40)

    def test_context_rejects_missing_extra_and_wrong_lane_coordinates(self) -> None:
        config = self._synthetic_config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config), open_database_snapshot(self.database, expected_sha256=self.digest) as snapshot:
            cases = (
                FrozenLaneSnapshot({}, {}, {}, {"A": (0,), "B": (1,)}, config=config, population=("A", "B")),
                FrozenLaneSnapshot({}, {}, {}, {"A": (0,), "outside": (0,)}, config=config, population=("A", "B")),
            )
            for lanes in cases:
                with self.subTest(lanes=lanes):
                    with self.assertRaisesRegex(ValueError, "coordinates|outside the selected"):
                        snapshot.freeze_context(lanes, verified_code_commit="a" * 40)

    def test_each_new_snapshot_requires_its_own_context_identity(self) -> None:
        config = self._synthetic_config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config):
            with open_database_snapshot(self.database, expected_sha256=self.digest) as first:
                first_context = first.freeze_context(self._lanes(config), verified_code_commit="a" * 40)
            with open_database_snapshot(self.database, expected_sha256=self.digest) as second:
                second_context = second.freeze_context(self._lanes(config), verified_code_commit="a" * 40)
                self.assertIsNot(first_context, second_context)
                with self.assertRaisesRegex(ValueError, "not this snapshot"):
                    second.load_train("A", first_context)
                self.assertEqual(second.load_train("A", second_context).dataset, "A")

    def test_mutation_matrix_fails_closed_at_a_boundary(self) -> None:
        for mutation in ("truncate", "byte", "chmod", "disappear", "sidecar", "unknown"):
            with self.subTest(mutation=mutation):
                manager = open_database_snapshot(self.database, expected_sha256=self.digest)
                snapshot = manager.__enter__()
                if mutation == "truncate":
                    with self.database.open("r+b") as output: output.truncate(1)
                elif mutation == "byte":
                    with self.database.open("r+b") as output: output.seek(100); output.write(b"x")
                elif mutation == "chmod": os.chmod(self.database, 0o600)
                elif mutation == "disappear": self.database.unlink()
                elif mutation == "sidecar": (self.root / "input.sqlite-shm").write_bytes(b"x")
                else: (self.root / "input.sqlite-mj00000001").write_bytes(b"x")
                with self.assertRaises(ValueError): snapshot.verify_boundary(mutation)
                with self.assertRaises(ValueError): manager.__exit__(None, None, None)
                self.tearDown(); self.setUp()

    def test_loader_preserves_decoder_error_and_runs_after_boundary(self) -> None:
        config = self._synthetic_config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config), open_database_snapshot(self.database, expected_sha256=self.digest) as snapshot:
            context = snapshot.freeze_context(self._lanes(config), verified_code_commit="a" * 40)
            before = len(snapshot.audit.boundaries)
            with patch.object(snapshot_database.exp417_inputs, "load_raw_train_values_from_connection", side_effect=ValueError("bad float32 row")):
                with self.assertRaisesRegex(ValueError, "bad float32 row"): snapshot.load_train("A", context)
            self.assertEqual(snapshot.audit.boundaries[before:], ("load_train:before", "load_train:after"))

    def test_malformed_rows_and_wrong_context_contracts_still_run_after_boundary(self) -> None:
        connection = sqlite3.connect(self.database)
        try:
            connection.execute("UPDATE instances SET values_blob = ? WHERE split = 'TRAIN' AND dataset_id = 1", (b"bad",))
            connection.commit()
        finally:
            connection.close()
        self.digest = sha256(self.database.read_bytes()).hexdigest()
        config = self._synthetic_config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config), open_database_snapshot(self.database, expected_sha256=self.digest) as snapshot:
            context = snapshot.freeze_context(self._lanes(config), verified_code_commit="a" * 40)
            with self.assertRaises(ValueError):
                snapshot.load_train("A", context)
            self.assertEqual(snapshot.audit.boundaries[-2:], ("load_train:before", "load_train:after"))
            # Frozen dataclasses reject bad state at construction, so this
            # deliberately models a corrupted deserialised context at the API
            # boundary and proves the snapshot still executes its after-check.
            bad = object.__new__(FrozenRunContext)
            for field, value in (("database_sha256", "b" * 64), ("datasets", context.datasets), ("lanes", context.lanes), ("test_identity_manifest", context.test_identity_manifest), ("code_commit", context.code_commit), ("runtime_versions", context.runtime_versions), ("config", context.config)):
                object.__setattr__(bad, field, value)
            with self.assertRaisesRegex(ValueError, "context"):
                snapshot.load_train("A", bad)
            self.assertEqual(snapshot.audit.boundaries[-2:], ("load_train:before", "load_train:after"))
            with self.assertRaises(ValueError):
                snapshot.load_test("A", self._contract("B"), context)
            self.assertEqual(snapshot.audit.boundaries[-2:], ("load_test:before", "load_test:after"))

    @unittest.skipUnless(sys.platform == "darwin", "owned hard-link behavior is macOS-only")
    def test_owned_link_replacement_is_never_unlinked_during_cleanup(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        owned = self.root / snapshot._owned_directory / snapshot._owned_name
        replacement = owned.with_name("replacement.sqlite")
        replacement.write_bytes(b"replacement")
        os.replace(replacement, owned)
        with self.assertRaisesRegex(ValueError, "descriptor metadata changed|owned database link was replaced"):
            snapshot.close()
        self.assertEqual(owned.read_bytes(), b"replacement")
        self.assertEqual(snapshot.audit.cleanup_status, "retained_after_cleanup_failure")
        self.assertEqual(snapshot.audit.lifecycle, "closed_cleanup_failed")

    @unittest.skipUnless(sys.platform == "darwin", "owned hard-link behavior is macOS-only")
    def test_metadata_mutation_keeps_lock_through_owned_cleanup(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        owned_directory = self.root / snapshot._owned_directory
        with self.database.open("r+b") as output:
            output.seek(100)
            output.write(b"x")
        with self.assertRaisesRegex(ValueError, "descriptor metadata changed"):
            snapshot.close()
        self.assertFalse(owned_directory.exists())
        self.assertEqual(snapshot.audit.cleanup_status, "removed")
        self.assertFalse(snapshot.audit.sqlite_read_lock_held)

    @unittest.skipUnless(sys.platform == "darwin", "owned hard-link behavior is macOS-only")
    def test_owned_directory_replacement_and_early_open_error_are_safe(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        owned_directory = self.root / snapshot._owned_directory
        moved = owned_directory.with_name(owned_directory.name + "-moved")
        owned_directory.rename(moved)
        owned_directory.mkdir()
        fake = owned_directory / snapshot._owned_name
        fake.write_bytes(b"replacement")
        with self.assertRaises(ValueError):
            snapshot.close()
        self.assertEqual(fake.read_bytes(), b"replacement")
        shutil.rmtree(owned_directory)
        shutil.rmtree(moved)
        with patch.object(snapshot_database.os, "link", side_effect=OSError("link failure")):
            with self.assertRaisesRegex(OSError, "link failure"):
                with open_database_snapshot(self.database, expected_sha256=self.digest):
                    pass
        self.assertFalse(tuple(self.root.glob(".exp425-snapshot-*")))
        # An opening failure after private-link creation removes only the exact
        # owned link/directory and leaves no normal-path artifact behind.
        with patch.object(snapshot_database.sqlite3, "connect", side_effect=sqlite3.OperationalError("open failure")):
            with self.assertRaisesRegex(sqlite3.OperationalError, "open failure"):
                with open_database_snapshot(self.database, expected_sha256=self.digest):
                    pass
        self.assertFalse(tuple(self.root.glob(".exp425-snapshot-*")))

    @unittest.skipUnless(sys.platform == "darwin", "owned hard-link behavior is macOS-only")
    def test_close_rechecks_parent_namespace_after_owned_link_cleanup(self) -> None:
        manager = open_database_snapshot(self.database, expected_sha256=self.digest)
        snapshot = manager.__enter__()
        original_rmdir = snapshot_database.os.rmdir
        moved = self.root.with_name(self.root.name + "-post-cleanup")

        def replace_parent_after_owned_rmdir(name, *args, **kwargs):
            original_rmdir(name, *args, **kwargs)
            if name == snapshot._owned_directory and kwargs.get("dir_fd") == snapshot._parent_fd:
                self.root.rename(moved)
                self.root.mkdir()

        with patch.object(snapshot_database.os, "rmdir", side_effect=replace_parent_after_owned_rmdir):
            with self.assertRaisesRegex(ValueError, "parent namespace"):
                snapshot.close()
        shutil.rmtree(moved)


class DatabaseBenchmarkTests(unittest.TestCase):
    def test_256mib_snapshot_reads_beat_twenty_full_descriptor_hashes(self) -> None:
        """Reproducible benchmark; no product behavior depends on this timing."""
        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "benchmark.sqlite"
            connection = sqlite3.connect(database)
            try:
                connection.executescript("CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT); CREATE TABLE instances (dataset_id INTEGER, instance_index INTEGER, split TEXT, label TEXT, values_blob BLOB); CREATE TABLE padding (payload BLOB);")
                values = struct.pack("<2f", 1.0, 2.0)
                connection.execute("INSERT INTO datasets VALUES (1, 'A')")
                connection.executemany("INSERT INTO instances VALUES (?, ?, ?, ?, ?)", ((1, 0, "TRAIN", "0", values), (1, 0, "TEST", "hidden", values)))
                connection.execute("INSERT INTO padding VALUES (zeroblob(?))", (256 * 1024 * 1024,))
                connection.commit()
            finally: connection.close()
            digest = sha256(database.read_bytes()).hexdigest()
            config = replace(FROZEN_CONFIG, expected_datasets=1, expected_hard=0, expected_standard=0, expected_priority=0, expected_automatic_normal=1, database_sha256=digest)
            with patch.object(Exp417FrozenConfig, "validate", lambda _self: None):
                lanes = FrozenLaneSnapshot({}, {}, {}, {"A": (0,)}, config=config, population=("A",))
            start = time.perf_counter()
            with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(snapshot_database, "FROZEN_CONFIG", config), patch.object(snapshot_database, "_sha256_descriptor", wraps=snapshot_database._sha256_descriptor) as hashed, open_database_snapshot(database, expected_sha256=digest) as snapshot:
                context = snapshot.freeze_context(lanes, verified_code_commit="a" * 40)
                read_rows = tuple(snapshot.load_train("A", context).series[0].values for _ in range(20))
                self.assertEqual(snapshot.audit.full_hash_count, 1)
                self.assertEqual(hashed.call_count, 1)
            snapshot_seconds = time.perf_counter() - start
            start = time.perf_counter(); verified_rows = []
            for _ in range(20):
                with snapshot_database.exp417_inputs.verified_read_only_connection(database, digest) as verified:
                    verified_rows.append(snapshot_database.exp417_inputs.load_raw_train_values_from_connection(verified, "A", database_sha256=digest, datasets=("A",)).series[0].values)
            full_hash_seconds = time.perf_counter() - start
            self.assertEqual(read_rows, ((1.0, 2.0),) * 20)
            self.assertEqual(verified_rows, list(read_rows))
            self.assertLess(snapshot_seconds, full_hash_seconds, (snapshot_seconds, full_hash_seconds))
            print(f"256MiB snapshot={snapshot_seconds:.3f}s verified_opens={full_hash_seconds:.3f}s")
