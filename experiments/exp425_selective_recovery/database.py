"""One-hash, descriptor-bound SQLite input snapshot for Exp425.

This deliberately owns no database semantics: population, TEST identities, and
raw-value decoding are the fixed Exp417 contract executed on one immutable
connection.  The surrounding descriptor checks make that connection a
process-lifetime snapshot without rehashing the database for every dataset.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
from types import MappingProxyType
from urllib.parse import quote
from uuid import uuid4

from experiments.exp417_causal_multihorizon_shadow import inputs as exp417_inputs
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    RawTestDataset,
    RawTrainDataset,
)


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


@dataclass(frozen=True)
class DatabaseSnapshotAudit:
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    full_hash_count: int
    boundaries: tuple[str, ...]


def _sha256_descriptor(descriptor: int) -> str:
    """Hash the accepted descriptor only; never reopen the user pathname."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _metadata(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink


class Exp425DatabaseSnapshot:
    """Opaque live owner of accepted DB/parent descriptors and immutable SQLite."""

    __slots__ = (
        "_database", "_name", "_parent_fd", "_database_fd", "_connection",
        "_before", "_operational", "_parent_identity", "_parent_metadata",
        "_owned_directory", "_owned_name", "_closed", "_audit",
    )

    def __init__(
        self,
        database: Path,
        name: str,
        parent_fd: int,
        database_fd: int,
        connection: sqlite3.Connection,
        before: os.stat_result,
        operational: os.stat_result,
        parent: os.stat_result,
        owned_directory: str | None,
        owned_name: str | None,
        digest: str,
    ) -> None:
        self._database = database
        self._name = name
        self._parent_fd = parent_fd
        self._database_fd = database_fd
        self._connection = connection
        self._before = before
        self._operational = operational
        self._parent_identity = _identity(parent)
        self._parent_metadata = _metadata(parent)
        self._owned_directory = owned_directory
        self._owned_name = owned_name
        self._closed = False
        self._audit = DatabaseSnapshotAudit(digest, before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, 1, ())

    @property
    def audit(self) -> DatabaseSnapshotAudit:
        return self._audit

    def _require_live(self) -> None:
        if self._closed:
            raise ValueError("Exp425 database snapshot is closed")

    def _sidecars_absent(self) -> None:
        for suffix in _SIDECAR_SUFFIXES:
            try:
                os.stat(self._name + suffix, dir_fd=self._parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError("SQLite sidecar or rollback journal is present")

    def _published_info(self) -> os.stat_result:
        try:
            info = os.stat(self._name, dir_fd=self._parent_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError("database pathname disappeared") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("database pathname is not the accepted regular file")
        return info

    def _owned_info(self) -> os.stat_result:
        if self._owned_directory is None or self._owned_name is None:
            return self._published_info()
        directory_fd = os.open(self._owned_directory, _DIRECTORY_FLAGS, dir_fd=self._parent_fd)
        try:
            info = os.stat(self._owned_name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError("owned database link disappeared") from error
        finally:
            os.close(directory_fd)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("owned database link is invalid")
        return info

    def verify_boundary(self, name: str) -> None:
        self._require_live()
        if not isinstance(name, str) or not name:
            raise ValueError("database boundary name is invalid")
        parent = os.fstat(self._parent_fd)
        if _identity(parent) != self._parent_identity or _metadata(parent) != self._parent_metadata:
            raise ValueError("database parent metadata changed")
        current = os.fstat(self._database_fd)
        if _metadata(current) != _metadata(self._operational):
            raise ValueError("database descriptor metadata changed")
        if _metadata(self._published_info()) != _metadata(self._operational):
            raise ValueError("database pathname metadata changed")
        if _metadata(self._owned_info()) != _metadata(self._operational):
            raise ValueError("owned database link metadata changed")
        self._sidecars_absent()
        self._audit = replace(self._audit, boundaries=self._audit.boundaries + (name,))

    def freeze_context(self, lanes: FrozenLaneSnapshot, *, verified_code_commit: str) -> FrozenRunContext:
        self.verify_boundary("freeze_context:before")
        if not isinstance(lanes, FrozenLaneSnapshot) or lanes.config != FROZEN_CONFIG or not re.fullmatch(r"[0-9a-f]{40}", verified_code_commit):
            raise ValueError("frozen lanes or verified code commit is invalid")
        try:
            rows = self._connection.execute("SELECT name FROM datasets ORDER BY name").fetchall()
            database_datasets = tuple(row[0] for row in rows)
            if (not database_datasets or any(not isinstance(item, str) or not item for item in database_datasets) or database_datasets != tuple(sorted(database_datasets)) or len(database_datasets) != len(set(database_datasets))):
                raise ValueError("database population is invalid")
            frozen_population = set(lanes.population) if lanes.population else set(database_datasets)
            if not lanes.population and len(frozen_population) != FROZEN_CONFIG.expected_datasets:
                raise ValueError("database population does not match frozen dataset count")
            if frozen_population - set(database_datasets):
                raise ValueError("frozen lane references a dataset outside the database population")
            selected = tuple(name for name in database_datasets if name in frozen_population)
            if len(selected) != FROZEN_CONFIG.expected_datasets:
                raise ValueError("database population does not cover frozen lane universe")
            manifest: dict[str, tuple[tuple[int, int], ...]] = {}
            for dataset in selected:
                rows = self._connection.execute(exp417_inputs._TEST_IDENTITY_SQL, (dataset,)).fetchall()
                identities = tuple((index, length // 4) for index, length in rows)
                if (not identities or any(type(index) is not int or type(length) is not int or length <= 0 or length % 4 for index, length in rows) or tuple(index for index, _ in identities) != tuple(sorted(index for index, _ in identities)) or len({index for index, _ in identities}) != len(identities)):
                    raise ValueError("database TEST identity manifest is invalid")
                manifest[dataset] = identities
        except sqlite3.Error as error:
            raise ValueError("Exp417 database population or identity query failed") from error
        context = FrozenRunContext(self._audit.sha256, selected, lanes, manifest, verified_code_commit, exp417_inputs._runtime_versions(), config=FROZEN_CONFIG)
        self.verify_boundary("freeze_context:after")
        return context

    def load_train(self, dataset: str, context: FrozenRunContext) -> RawTrainDataset:
        self.verify_boundary("load_train:before")
        try:
            if not isinstance(context, FrozenRunContext) or context.database_sha256 != self._audit.sha256 or context.config != FROZEN_CONFIG or dataset not in context.datasets:
                raise ValueError("TRAIN context does not bind this database snapshot")
            result = exp417_inputs.load_raw_train_values_from_connection(self._connection, dataset, database_sha256=context.database_sha256, datasets=context.datasets)
        except sqlite3.Error as error:
            raise ValueError("Exp417 TRAIN decoder query failed") from error
        finally:
            self.verify_boundary("load_train:after")
        return result

    def load_test(self, dataset: str, contract: FrozenTrainContract, context: FrozenRunContext) -> RawTestDataset:
        self.verify_boundary("load_test:before")
        try:
            if not isinstance(context, FrozenRunContext) or context.database_sha256 != self._audit.sha256 or context.config != FROZEN_CONFIG or dataset not in context.datasets:
                raise ValueError("TEST context does not bind this database snapshot")
            result = exp417_inputs.load_raw_test_values_from_connection(self._connection, dataset, contract, context)
        except sqlite3.Error as error:
            raise ValueError("Exp417 TEST decoder query failed") from error
        finally:
            self.verify_boundary("load_test:after")
        return result

    def close(self) -> None:
        if self._closed:
            raise ValueError("Exp425 database snapshot is closed")
        error: BaseException | None = None
        try:
            self.verify_boundary("close:before")
        except BaseException as caught:
            error = caught
        try:
            self._connection.close()
            if self._owned_directory is not None and self._owned_name is not None:
                directory_fd = os.open(self._owned_directory, _DIRECTORY_FLAGS, dir_fd=self._parent_fd)
                try:
                    owned = os.stat(self._owned_name, dir_fd=directory_fd, follow_symlinks=False)
                    if _metadata(owned) != _metadata(self._operational):
                        raise ValueError("owned database link was replaced before cleanup")
                    os.unlink(self._owned_name, dir_fd=directory_fd)
                finally:
                    os.close(directory_fd)
                os.rmdir(self._owned_directory, dir_fd=self._parent_fd)
                # The owned link alone may change ctime/nlink. All other file
                # properties and the published identity must remain exact.
                current = os.fstat(self._database_fd)
                published = self._published_info()
                if _identity(current) != _identity(self._before) or current.st_size != self._before.st_size or current.st_mtime_ns != self._before.st_mtime_ns or current.st_nlink != self._before.st_nlink or _identity(published) != _identity(self._before) or published.st_size != self._before.st_size or published.st_mtime_ns != self._before.st_mtime_ns or published.st_nlink != self._before.st_nlink:
                    raise ValueError("database changed during owned-link cleanup")
                self._sidecars_absent()
        except BaseException as caught:
            if error is None:
                error = caught
        finally:
            self._closed = True
            try:
                self._connection.close()
            finally:
                os.close(self._database_fd)
                os.close(self._parent_fd)
        if error is not None:
            raise error

    def __enter__(self) -> "Exp425DatabaseSnapshot":
        self._require_live()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@contextmanager
def open_database_snapshot(database: Path, *, expected_sha256: str) -> Iterator[Exp425DatabaseSnapshot]:
    """Accept one immutable, sidecar-free SQLite inode and hash it exactly once."""
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected database SHA256 is invalid")
    path = Path(database)
    if path.name in {"", ".", ".."} or path.is_symlink():
        raise ValueError("database path is invalid or a symlink")
    parent_fd = database_fd = None
    snapshot: Exp425DatabaseSnapshot | None = None
    owned_directory = owned_name = None
    connection: sqlite3.Connection | None = None
    try:
        parent_fd = os.open(path.parent, _DIRECTORY_FLAGS)
        database_fd = os.open(path.name, _FILE_FLAGS, dir_fd=parent_fd)
        before = os.fstat(database_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("database must be an unaliased regular file")
        # A temporary tiny owner directory prevents alias collisions and gives
        # cleanup an exact, capability-like pathname to remove.
        for suffix in _SIDECAR_SUFFIXES:
            try:
                os.stat(path.name + suffix, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError("SQLite sidecar or rollback journal is present")
        actual = _sha256_descriptor(database_fd)
        if actual != expected_sha256:
            raise ValueError("database SHA256 does not match expected digest")
        if sys.platform == "darwin":
            owned_directory = ".exp425-snapshot-" + uuid4().hex
            owned_name = "database.sqlite"
            os.mkdir(owned_directory, 0o700, dir_fd=parent_fd)
            owned_fd = os.open(owned_directory, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            try:
                os.link(path.name, owned_name, src_dir_fd=parent_fd, dst_dir_fd=owned_fd, follow_symlinks=False)
                owned_info = os.stat(owned_name, dir_fd=owned_fd, follow_symlinks=False)
                if _identity(owned_info) != _identity(before) or not stat.S_ISREG(owned_info.st_mode):
                    raise ValueError("owned database link does not bind accepted inode")
            finally:
                os.close(owned_fd)
        operational = os.fstat(database_fd)
        expected_links = 2 if sys.platform == "darwin" else 1
        if _identity(operational) != _identity(before) or operational.st_size != before.st_size or operational.st_mtime_ns != before.st_mtime_ns or operational.st_nlink != expected_links:
            raise ValueError("database changed while creating snapshot")
        target = path if owned_directory is None else path.parent / owned_directory / owned_name
        connection = sqlite3.connect("file:" + quote(str(target)) + "?mode=ro&immutable=1", uri=True)
        connection.execute("PRAGMA query_only=ON")
        exp417_inputs.assert_label_free_sql_contract()
        snapshot = Exp425DatabaseSnapshot(path, path.name, parent_fd, database_fd, connection, before, operational, os.fstat(parent_fd), owned_directory, owned_name, actual)
        snapshot.verify_boundary("open:accepted")
        parent_fd = database_fd = None
        connection = None
        yield snapshot
    finally:
        if snapshot is not None:
            snapshot.close()
        else:
            if connection is not None:
                connection.close()
            if owned_directory is not None and parent_fd is not None:
                try:
                    owned_fd = os.open(owned_directory, _DIRECTORY_FLAGS, dir_fd=parent_fd)
                    try:
                        if owned_name is not None:
                            os.unlink(owned_name, dir_fd=owned_fd)
                    finally:
                        os.close(owned_fd)
                    os.rmdir(owned_directory, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            if database_fd is not None:
                os.close(database_fd)
            if parent_fd is not None:
                os.close(parent_fd)
