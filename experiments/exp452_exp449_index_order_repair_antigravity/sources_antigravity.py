"""The sole one-shot TEST-label capability for Exp452 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable

from .predecessor_antigravity import (
    _claim_label_capability_antigravity,
    _finish_label_capability_antigravity,
)


_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_DATABASE_SHA256_ANTIGRAVITY = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
_DATABASE_INODE_ANTIGRAVITY = 1048138
_DATABASE_FLAGS_ANTIGRAVITY = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_LABEL_SQL_ANTIGRAVITY = (
    "SELECT i.instance_index, i.label FROM instances AS i "
    "JOIN datasets AS d ON d.id=i.dataset_id "
    "WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index"
)
_TARGET_DATASET_COUNT_ANTIGRAVITY = 48
_TARGET_LABEL_ROWS_ANTIGRAVITY = 72300
_IdentityAntigravity = tuple[int, int, int, int, int, int, int]


def _identity_antigravity(info: os.stat_result) -> _IdentityAntigravity:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _verify_database_antigravity() -> tuple[int, _IdentityAntigravity]:
    """Return one read-only descriptor after exact inode, link, and byte validation."""
    try:
        before = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        descriptor = os.open(_DATABASE_ANTIGRAVITY, _DATABASE_FLAGS_ANTIGRAVITY)
    except OSError as error:
        raise ValueError("source database is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        identity = _identity_antigravity(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_ino != _DATABASE_INODE_ANTIGRAVITY
            or _identity_antigravity(opened) != identity
        ):
            raise ValueError("source database inode/link contract is invalid")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        current = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        if digest.hexdigest() != _DATABASE_SHA256_ANTIGRAVITY or _identity_antigravity(current) != identity:
            raise ValueError("source database changed during digest verification")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _connection_antigravity(descriptor: int):  # type: ignore[no-untyped-def]
    import sqlite3

    connection = sqlite3.connect(
        f"file:/dev/fd/{descriptor}?mode=ro&immutable=1",
        uri=True,
        isolation_level=None,
    )
    permitted = frozenset({
        ("instances", "dataset_id"),
        ("instances", "split"),
        ("instances", "instance_index"),
        ("instances", "label"),
        ("datasets", "id"),
        ("datasets", "name"),
    })

    def authorizer(
        action: int,
        table: str | None,
        column: str | None,
        _database: str | None,
        _trigger: str | None,
    ) -> int:
        if action == sqlite3.SQLITE_SELECT or (
            action == sqlite3.SQLITE_READ and (table, column) in permitted
        ):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    connection.set_authorizer(authorizer)
    return connection


def _assert_database_unchanged_antigravity(
    descriptor: int,
    identity: _IdentityAntigravity,
) -> None:
    try:
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        opened = os.fstat(descriptor)
        current = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        if (
            digest.hexdigest() != _DATABASE_SHA256_ANTIGRAVITY
            or opened.st_nlink != 1
            or _identity_antigravity(opened) != identity
            or _identity_antigravity(current) != identity
        ):
            raise ValueError("source database changed during TEST-label session")
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise ValueError("source database changed during TEST-label session") from error


def convert_test_label_row_antigravity(instance_index: object, raw_label: object) -> tuple[int, int]:
    """Accept exactly contiguous integer positions and SQLite TEXT binary labels."""
    if type(instance_index) is not int or type(raw_label) is not str or raw_label not in ("0", "1"):
        raise ValueError("TEST label row is invalid")
    return instance_index, 0 if raw_label == "0" else 1


@dataclass(frozen=True, slots=True)
class LabelSessionAuditAntigravity:
    session_count: int
    setup_attempts: int
    setup_completions: int
    query_attempts: int
    query_completions: int
    label_rows: int
    close_attempts: int
    close_completions: int
    state: str
    primary_error: str | None
    cleanup_error: str | None

    def __post_init__(self) -> None:
        counts = (
            self.session_count,
            self.setup_attempts,
            self.setup_completions,
            self.query_attempts,
            self.query_completions,
            self.label_rows,
            self.close_attempts,
            self.close_completions,
        )
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or self.session_count != 1
            or self.setup_attempts != 1
            or self.setup_completions > self.setup_attempts
            or self.query_completions > self.query_attempts
            or self.close_completions > self.close_attempts
            or self.state not in {"opening", "open", "closed", "terminal", "close_failed", "setup_failed"}
            or (self.primary_error is not None and type(self.primary_error) is not str)
            or (self.cleanup_error is not None and type(self.cleanup_error) is not str)
        ):
            raise ValueError("label session audit is invalid")


def _no_audit_sink_antigravity(_audit: LabelSessionAuditAntigravity) -> None:
    return None


@dataclass(frozen=True, slots=True)
class _LabelDependenciesAntigravity:
    verify_database: Callable[[], tuple[int, _IdentityAntigravity]]
    open_connection: Callable[[int], object]
    assert_database_unchanged: Callable[[int, _IdentityAntigravity], None]
    close_descriptor: Callable[[int], None]
    audit_sink: Callable[[LabelSessionAuditAntigravity], None] = _no_audit_sink_antigravity

    def __post_init__(self) -> None:
        if any(
            not callable(value)
            for value in (
                self.verify_database,
                self.open_connection,
                self.assert_database_unchanged,
                self.close_descriptor,
                self.audit_sink,
            )
        ):
            raise ValueError("label dependencies are invalid")


def _production_dependencies_antigravity() -> _LabelDependenciesAntigravity:
    return _LabelDependenciesAntigravity(
        verify_database=_verify_database_antigravity,
        open_connection=_connection_antigravity,
        assert_database_unchanged=_assert_database_unchanged_antigravity,
        close_descriptor=os.close,
    )


def _error_text_antigravity(error: BaseException | None) -> str | None:
    return None if error is None else f"{type(error).__name__}: {error}"


def _open_test_labels_once_with_dependencies_antigravity(
    authority: object,
    dependencies: _LabelDependenciesAntigravity,
):  # type: ignore[no-untyped-def]
    """Private synthetic seam; authority is consumed before any setup can fail."""
    if type(dependencies) is not _LabelDependenciesAntigravity:
        raise ValueError("label dependencies are invalid")
    order = _claim_label_capability_antigravity(authority)
    if (
        type(order) is not tuple
        or len(order) != _TARGET_DATASET_COUNT_ANTIGRAVITY
        or len(set(order)) != len(order)
        or any(type(dataset) is not str or not dataset for dataset in order)
    ):
        _finish_label_capability_antigravity(authority, "setup_failed")
        raise ValueError("Exp449 stopped authority dataset order is invalid")

    descriptor = -1
    identity: _IdentityAntigravity | None = None
    connection = None
    state = "opening"
    setup_attempts = 1
    setup_completions = 0
    query_attempts = query_completions = label_rows = 0
    close_attempts = close_completions = 0
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    connection_close_attempted = False
    descriptor_close_attempted = False
    next_dataset = 0

    def audit_antigravity() -> LabelSessionAuditAntigravity:
        return LabelSessionAuditAntigravity(
            1,
            setup_attempts,
            setup_completions,
            query_attempts,
            query_completions,
            label_rows,
            close_attempts,
            close_completions,
            state,
            _error_text_antigravity(primary_error),
            _error_text_antigravity(cleanup_error),
        )

    def emit_antigravity() -> None:
        dependencies.audit_sink(audit_antigravity())

    def close_connection_antigravity() -> None:
        nonlocal connection_close_attempted, close_attempts, cleanup_error
        if connection is not None and not connection_close_attempted:
            connection_close_attempted = True
            close_attempts += 1
            try:
                connection.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

    def close_descriptor_antigravity() -> None:
        nonlocal descriptor_close_attempted, cleanup_error
        if descriptor >= 0 and not descriptor_close_attempted:
            descriptor_close_attempted = True
            try:
                dependencies.close_descriptor(descriptor)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error

    def terminalize_antigravity(error: BaseException | None) -> None:
        nonlocal state, primary_error
        if state not in {"opening", "open"}:
            return
        if error is not None and primary_error is None:
            primary_error = error
        close_connection_antigravity()
        close_descriptor_antigravity()
        state = "close_failed" if cleanup_error is not None else ("setup_failed" if setup_completions == 0 else "terminal")
        _finish_label_capability_antigravity(authority, state)
        emit_antigravity()

    try:
        descriptor, identity = dependencies.verify_database()
        if type(descriptor) is not int or descriptor < 0 or type(identity) is not tuple or len(identity) != 7:
            raise ValueError("source database descriptor authority is invalid")
        connection = dependencies.open_connection(descriptor)
        if connection is None or not callable(getattr(connection, "execute", None)) or not callable(getattr(connection, "close", None)):
            raise ValueError("source database connection is invalid")
        setup_completions = 1
        state = "open"
        _finish_label_capability_antigravity(authority, "open")
        emit_antigravity()
    except BaseException as error:
        terminalize_antigravity(error)
        raise

    def stored_error_antigravity() -> BaseException:
        if primary_error is not None:
            return primary_error
        if cleanup_error is not None:
            return cleanup_error
        return ValueError("TEST label session is closed")

    class LabelSessionAntigravity:
        __slots__ = ()

        def audit_antigravity(self) -> LabelSessionAuditAntigravity:
            return audit_antigravity()

        def close_state_antigravity(self) -> str:
            return state

        def terminal_error_antigravity(self) -> BaseException | None:
            return primary_error

        def close_error_antigravity(self) -> BaseException | None:
            return cleanup_error

        def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
            nonlocal next_dataset, query_attempts, query_completions, label_rows
            if state != "open":
                raise stored_error_antigravity()
            if type(dataset) is not str or next_dataset >= len(order) or dataset != order[next_dataset]:
                error = ValueError("TEST labels must be read once in frozen dataset order")
                terminalize_antigravity(error)
                raise error
            query_attempts += 1
            try:
                rows = tuple(
                    convert_test_label_row_antigravity(index, raw_label)
                    for index, raw_label in connection.execute(_LABEL_SQL_ANTIGRAVITY, (dataset,))
                )
                if not rows or tuple(index for index, _label in rows) != tuple(range(len(rows))):
                    raise ValueError("TEST indices are not contiguous")
                next_count = label_rows + len(rows)
                if next_count > _TARGET_LABEL_ROWS_ANTIGRAVITY or (
                    next_dataset + 1 < len(order) and next_count >= _TARGET_LABEL_ROWS_ANTIGRAVITY
                ):
                    raise ValueError("TEST label row count differs from frozen authority")
                if next_dataset + 1 == len(order) and next_count != _TARGET_LABEL_ROWS_ANTIGRAVITY:
                    raise ValueError("TEST label row count differs from frozen authority")
                next_dataset += 1
                label_rows = next_count
                query_completions += 1
                emit_antigravity()
                return rows
            except BaseException as error:
                terminalize_antigravity(error)
                raise

        def close_antigravity(self) -> None:
            nonlocal state, close_completions, primary_error, cleanup_error, close_attempts, connection_close_attempted
            if state == "closed":
                return
            if state != "open":
                raise stored_error_antigravity()
            if next_dataset != len(order) or label_rows != _TARGET_LABEL_ROWS_ANTIGRAVITY:
                error = ValueError("TEST label session did not complete the frozen dataset order")
                terminalize_antigravity(error)
                raise error
            connection_close_attempted = True
            close_attempts += 1
            try:
                connection.close()
            except BaseException as error:
                cleanup_error = error
                close_descriptor_antigravity()
                state = "close_failed"
                _finish_label_capability_antigravity(authority, state)
                emit_antigravity()
                raise
            try:
                assert identity is not None
                dependencies.assert_database_unchanged(descriptor, identity)
            except BaseException as error:
                primary_error = error
                close_descriptor_antigravity()
                state = "close_failed" if cleanup_error is not None else "terminal"
                _finish_label_capability_antigravity(authority, state)
                emit_antigravity()
                raise
            close_descriptor_antigravity()
            if cleanup_error is not None:
                state = "close_failed"
                _finish_label_capability_antigravity(authority, state)
                emit_antigravity()
                raise cleanup_error
            close_completions = 1
            state = "closed"
            _finish_label_capability_antigravity(authority, state)
            emit_antigravity()

    return LabelSessionAntigravity()


def open_test_labels_once_antigravity(authority: object):  # type: ignore[no-untyped-def]
    """Open the fixed database exactly once from an issued twice-captured authority."""
    return _open_test_labels_once_with_dependencies_antigravity(
        authority,
        _production_dependencies_antigravity(),
    )


__all__ = [
    "LabelSessionAuditAntigravity",
    "convert_test_label_row_antigravity",
    "open_test_labels_once_antigravity",
]
