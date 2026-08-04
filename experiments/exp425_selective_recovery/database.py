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
import fcntl
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
# SQLite's locking page starts at 1 GiB.  Holding one shared-lock byte keeps a
# separate SQLite writer from acquiring the exclusive range required to commit.
# This is deliberately a POSIX record lock (rather than flock): SQLite itself
# uses POSIX byte-range locks on this page.
_SQLITE_SHARED_FIRST = 0x40000002


@dataclass(frozen=True)
class DatabaseSnapshotAudit:
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    operational_ctime_ns: int
    operational_nlink: int
    parent_device: int
    parent_inode: int
    full_hash_count: int
    boundaries: tuple[str, ...]
    sqlite_read_lock_held: bool = True
    lifecycle: str = "live"


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


def _accepted_path_info(parent_fd: int, name: str) -> os.stat_result:
    """Stat the published name through the retained parent without following it."""
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("database pathname disappeared") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("database pathname is not an accepted regular file")
    return info


def _safe_unlinked_owner(
    parent_fd: int,
    directory: str | None,
    name: str | None,
    directory_identity: tuple[int, int] | None,
    database_identity: tuple[int, int] | None,
    link_created: bool,
) -> None:
    """Remove only the exact private link and directory we created.

    A replaced name is left behind for operator inspection and reported rather
    than being unlinked during an exceptional pre-snapshot cleanup path.
    """
    if directory is None and name is None:
        return
    if directory is None or name is None or directory_identity is None or database_identity is None:
        raise ValueError("owned database cleanup state is incomplete")
    try:
        directory_fd = os.open(directory, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError as error:
        raise ValueError("owned database directory disappeared before cleanup") from error
    try:
        if _identity(os.fstat(directory_fd)) != directory_identity:
            raise ValueError("owned database directory was replaced before cleanup")
        try:
            owned = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            if link_created:
                raise ValueError("owned database link disappeared before cleanup") from error
            owned = None
        if owned is None:
            pass
        elif stat.S_ISLNK(owned.st_mode) or _identity(owned) != database_identity:
            raise ValueError("owned database link was replaced before cleanup")
        else:
            os.unlink(name, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        os.rmdir(directory, dir_fd=parent_fd)
    except (FileNotFoundError, OSError):
        pass


class Exp425DatabaseSnapshot:
    """Opaque live owner of accepted DB/parent descriptors and immutable SQLite."""

    __slots__ = (
        "_database", "_name", "_parent_fd", "_grandparent_fd", "_parent_name", "_database_fd", "_connection",
        "_before", "_operational", "_parent_identity", "_parent_metadata",
        "_owned_directory", "_owned_name", "_owned_directory_identity", "_closed", "_audit",
    )

    def __init__(
        self,
        database: Path,
        name: str,
        parent_fd: int,
        grandparent_fd: int,
        parent_name: str,
        database_fd: int,
        connection: sqlite3.Connection,
        before: os.stat_result,
        operational: os.stat_result,
        parent: os.stat_result,
        owned_directory: str | None,
        owned_name: str | None,
        owned_directory_identity: tuple[int, int] | None,
        digest: str,
    ) -> None:
        self._database = database
        self._name = name
        self._parent_fd = parent_fd
        self._grandparent_fd = grandparent_fd
        self._parent_name = parent_name
        self._database_fd = database_fd
        self._connection = connection
        self._before = before
        self._operational = operational
        self._parent_identity = _identity(parent)
        self._parent_metadata = _metadata(parent)
        self._owned_directory = owned_directory
        self._owned_name = owned_name
        self._owned_directory_identity = owned_directory_identity
        self._closed = False
        self._audit = DatabaseSnapshotAudit(digest, before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns, operational.st_ctime_ns, operational.st_nlink, parent.st_dev, parent.st_ino, 1, ())

    @property
    def audit(self) -> DatabaseSnapshotAudit:
        return self._audit

    def _require_live(self) -> None:
        if self._closed:
            raise ValueError("Exp425 database snapshot is closed")

    def _sidecars_absent(self) -> None:
        # SQLite's named journals all share the database basename.  Treat a
        # newly introduced, unknown suffix as relevant rather than assuming it
        # is harmless: this directory is part of the immutable snapshot.
        for entry in os.listdir(self._parent_fd):
            if entry != self._name and entry != self._owned_directory and entry.startswith(self._name + "-"):
                raise ValueError("unknown relevant SQLite artifact is present")
        for suffix in _SIDECAR_SUFFIXES:
            try:
                os.stat(self._name + suffix, dir_fd=self._parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError("SQLite sidecar or rollback journal is present")

    def _published_info(self) -> os.stat_result:
        return _accepted_path_info(self._parent_fd, self._name)

    def _owned_info(self) -> os.stat_result:
        if self._owned_directory is None or self._owned_name is None:
            return self._published_info()
        directory_fd = os.open(self._owned_directory, _DIRECTORY_FLAGS, dir_fd=self._parent_fd)
        try:
            if self._owned_directory_identity is None or _identity(os.fstat(directory_fd)) != self._owned_directory_identity:
                raise ValueError("owned database directory was replaced")
            info = os.stat(self._owned_name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError("owned database link disappeared") from error
        finally:
            os.close(directory_fd)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError("owned database link is invalid")
        return info

    def _parent_namespace_is_bound(self) -> None:
        parent = os.fstat(self._parent_fd)
        if _identity(parent) != self._parent_identity:
            raise ValueError("database parent identity changed")
        try:
            published_parent = os.stat(self._parent_name, dir_fd=self._grandparent_fd, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError("database parent namespace disappeared") from error
        if stat.S_ISLNK(published_parent.st_mode) or _identity(published_parent) != self._parent_identity:
            raise ValueError("database parent namespace was replaced")

    def verify_boundary(self, name: str) -> None:
        self._require_live()
        if not isinstance(name, str) or not name:
            raise ValueError("database boundary name is invalid")
        parent = os.fstat(self._parent_fd)
        if _identity(parent) != self._parent_identity or _metadata(parent) != self._parent_metadata:
            raise ValueError("database parent metadata changed")
        self._parent_namespace_is_bound()
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
        primary: BaseException | None = None
        try:
            if not isinstance(lanes, FrozenLaneSnapshot) or lanes.config != FROZEN_CONFIG or not re.fullmatch(r"[0-9a-f]{40}", verified_code_commit):
                raise ValueError("frozen lanes or verified code commit is invalid")
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
            primary = ValueError("Exp417 database population or identity query failed")
            primary.__cause__ = error
        except BaseException as error:
            primary = error
        if primary is None:
            try:
                context = FrozenRunContext(self._audit.sha256, selected, lanes, manifest, verified_code_commit, exp417_inputs._runtime_versions(), config=FROZEN_CONFIG)
            except BaseException as error:
                primary = error
        try:
            self.verify_boundary("freeze_context:after")
        except BaseException as boundary_error:
            if primary is not None:
                raise primary from boundary_error
            raise
        if primary is not None:
            raise primary
        return context

    def load_train(self, dataset: str, context: FrozenRunContext) -> RawTrainDataset:
        self.verify_boundary("load_train:before")
        primary: BaseException | None = None
        try:
            if not isinstance(context, FrozenRunContext) or context.database_sha256 != self._audit.sha256 or context.config != FROZEN_CONFIG or dataset not in context.datasets:
                raise ValueError("TRAIN context does not bind this database snapshot")
            result = exp417_inputs.load_raw_train_values_from_connection(self._connection, dataset, database_sha256=context.database_sha256, datasets=context.datasets)
        except sqlite3.Error as error:
            primary = ValueError("Exp417 TRAIN decoder query failed")
            primary.__cause__ = error
        except BaseException as error:
            primary = error
        try:
            self.verify_boundary("load_train:after")
        except BaseException as boundary_error:
            if primary is not None:
                raise primary from boundary_error
            raise
        if primary is not None:
            raise primary
        return result

    def load_test(self, dataset: str, contract: FrozenTrainContract, context: FrozenRunContext) -> RawTestDataset:
        self.verify_boundary("load_test:before")
        primary: BaseException | None = None
        try:
            if not isinstance(context, FrozenRunContext) or context.database_sha256 != self._audit.sha256 or context.config != FROZEN_CONFIG or dataset not in context.datasets:
                raise ValueError("TEST context does not bind this database snapshot")
            result = exp417_inputs.load_raw_test_values_from_connection(self._connection, dataset, contract, context)
        except sqlite3.Error as error:
            primary = ValueError("Exp417 TEST decoder query failed")
            primary.__cause__ = error
        except BaseException as error:
            primary = error
        try:
            self.verify_boundary("load_test:after")
        except BaseException as boundary_error:
            if primary is not None:
                raise primary from boundary_error
            raise
        if primary is not None:
            raise primary
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
                    if self._owned_directory_identity is None or _identity(os.fstat(directory_fd)) != self._owned_directory_identity:
                        raise ValueError("owned database directory was replaced before cleanup")
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
                self._parent_namespace_is_bound()
                self._sidecars_absent()
        except BaseException as caught:
            if error is None:
                error = caught
        finally:
            self._closed = True
            self._audit = replace(self._audit, sqlite_read_lock_held=False, lifecycle="closed")
            try:
                self._connection.close()
            finally:
                os.close(self._database_fd)
                os.close(self._parent_fd)
                os.close(self._grandparent_fd)
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
    path = Path(database).absolute()
    if path.name in {"", ".", ".."} or path.is_symlink():
        raise ValueError("database path is invalid or a symlink")
    grandparent_fd = parent_fd = database_fd = None
    snapshot: Exp425DatabaseSnapshot | None = None
    owned_directory = owned_name = None
    owned_directory_identity: tuple[int, int] | None = None
    accepted_identity: tuple[int, int] | None = None
    owned_link_created = False
    connection: sqlite3.Connection | None = None
    try:
        grandparent_fd = os.open(path.parent.parent, _DIRECTORY_FLAGS)
        parent_fd = os.open(path.parent.name, _DIRECTORY_FLAGS, dir_fd=grandparent_fd)
        database_fd = os.open(path.name, _FILE_FLAGS, dir_fd=parent_fd)
        before = os.fstat(database_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("database must be an unaliased regular file")
        accepted_identity = _identity(before)
        try:
            fcntl.lockf(database_fd, fcntl.LOCK_SH | fcntl.LOCK_NB, 1, _SQLITE_SHARED_FIRST, os.SEEK_SET)
        except OSError as error:
            raise ValueError("database SQLite read lock could not be acquired") from error
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
        # Close the acceptance race between descriptor hashing and SQLite open.
        # The pre-link state has exactly one name; on macOS the one owned link
        # below is the only permitted operational nlink/ctime transition.
        pre_link = os.fstat(database_fd)
        if _metadata(pre_link) != _metadata(before) or _metadata(_accepted_path_info(parent_fd, path.name)) != _metadata(before):
            raise ValueError("database changed between hash and snapshot acceptance")
        if _identity(os.fstat(parent_fd)) != _identity(os.stat(path.parent.name, dir_fd=grandparent_fd, follow_symlinks=False)):
            raise ValueError("database parent namespace changed during acceptance")
        if sys.platform == "darwin":
            owned_directory = ".exp425-snapshot-" + uuid4().hex
            owned_name = "database.sqlite"
            os.mkdir(owned_directory, 0o700, dir_fd=parent_fd)
            owned_fd = os.open(owned_directory, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            try:
                owned_directory_identity = _identity(os.fstat(owned_fd))
                os.link(path.name, owned_name, src_dir_fd=parent_fd, dst_dir_fd=owned_fd, follow_symlinks=False)
                owned_link_created = True
                owned_info = os.stat(owned_name, dir_fd=owned_fd, follow_symlinks=False)
                if _identity(owned_info) != _identity(before) or not stat.S_ISREG(owned_info.st_mode) or owned_info.st_nlink != 2:
                    raise ValueError("owned database link does not bind accepted inode")
            finally:
                os.close(owned_fd)
        operational = os.fstat(database_fd)
        expected_links = 2 if sys.platform == "darwin" else 1
        if _identity(operational) != _identity(before) or operational.st_size != before.st_size or operational.st_mtime_ns != before.st_mtime_ns or operational.st_nlink != expected_links:
            raise ValueError("database changed while creating snapshot")
        published_operational = _accepted_path_info(parent_fd, path.name)
        if _metadata(published_operational) != _metadata(operational):
            raise ValueError("database pathname changed while creating snapshot")
        target = path if owned_directory is None else path.parent / owned_directory / owned_name
        connection = sqlite3.connect("file:" + quote(str(target)) + "?mode=ro&immutable=1", uri=True)
        connection.execute("PRAGMA query_only=ON")
        exp417_inputs.assert_label_free_sql_contract()
        snapshot = Exp425DatabaseSnapshot(path, path.name, parent_fd, grandparent_fd, path.parent.name, database_fd, connection, before, operational, os.fstat(parent_fd), owned_directory, owned_name, owned_directory_identity, actual)
        snapshot.verify_boundary("open:accepted")
        grandparent_fd = parent_fd = database_fd = None
        connection = None
        yield snapshot
    finally:
        if snapshot is not None and not snapshot._closed:
            snapshot.close()
        else:
            if connection is not None:
                connection.close()
            if parent_fd is not None:
                _safe_unlinked_owner(parent_fd, owned_directory, owned_name, owned_directory_identity, accepted_identity, owned_link_created)
            if database_fd is not None:
                os.close(database_fd)
            if parent_fd is not None:
                os.close(parent_fd)
            if grandparent_fd is not None:
                os.close(grandparent_fd)
