"""Exp447 label-capable source authority built on source-free publication identity."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import PublicationIdentityAntigravity
from .publication_identity_antigravity import (
    _DATASET_ROOT_ANTIGRAVITY,
    _DIRECTORY_FLAGS_ANTIGRAVITY,
    _LEAF_FLAGS_ANTIGRAVITY,
    _PinnedLeafAntigravity,
    _assert_directory_antigravity,
    _assert_pinned_leaf_antigravity,
    _exact_names_antigravity,
    _identity_antigravity,
    _open_directory_antigravity,
    _pin_leaf_antigravity,
    _read_leaf_antigravity,
    _read_pinned_leaf_antigravity,
    _strict_jsonl_antigravity,
    capture_exp446_publication_antigravity,
)

_EXP404_HARD_ANTIGRAVITY = _DATASET_ROOT_ANTIGRAVITY / "exp403_408_feature_ml_validation" / "exp404_hard_confidence.jsonl"
_DATABASE_ANTIGRAVITY = _DATASET_ROOT_ANTIGRAVITY / "univariate_ts.db"
_DATABASE_SHA256_ANTIGRAVITY = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
_DATABASE_INODE_ANTIGRAVITY = 1048138


@dataclass(frozen=True, slots=True)
class HardPredictionsAntigravity:
    """The byte-pinned Exp404 Hard coordinates, with no database capability."""

    sha256: str
    coordinates: frozenset[tuple[str, int]]


def _verify_database_path_antigravity(path: Path, expected_sha256: str, expected_inode: int) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(path, follow_symlinks=False)
        descriptor = os.open(path, _LEAF_FLAGS_ANTIGRAVITY)
    except OSError as error:
        raise ValueError("source database is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        expected = _identity_antigravity(before)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_ino != expected_inode or _identity_antigravity(opened) != expected:
            raise ValueError("source database inode/link contract is invalid")
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        current = os.stat(path, follow_symlinks=False)
        if digest.hexdigest() != expected_sha256 or any(_identity_antigravity(item) != expected for item in (opened, current)) or current.st_nlink != 1:
            raise ValueError("source database changed during digest verification")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, expected
    except BaseException:
        os.close(descriptor)
        raise


def _verify_database_antigravity() -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    return _verify_database_path_antigravity(_DATABASE_ANTIGRAVITY, _DATABASE_SHA256_ANTIGRAVITY, _DATABASE_INODE_ANTIGRAVITY)


def capture_hard_predictions_antigravity(identity: PublicationIdentityAntigravity) -> HardPredictionsAntigravity:
    """Capture Exp404 Hard only after a verified Exp446 identity is supplied."""
    if not isinstance(identity, PublicationIdentityAntigravity):
        raise ValueError("verified Exp446 identity is required")
    parent_fd = -1
    pinned: _PinnedLeafAntigravity | None = None
    try:
        parent_fd, parent_identity = _open_directory_antigravity(None, _EXP404_HARD_ANTIGRAVITY.parent)
        pinned = _pin_leaf_antigravity(parent_fd, _EXP404_HARD_ANTIGRAVITY.name)
        payload = _read_pinned_leaf_antigravity(pinned)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != FROZEN_CONFIG_ANTIGRAVITY.exp404_hard_prediction_sha256:
            raise ValueError("Exp404 Hard digest differs from frozen authority")
        rows = _strict_jsonl_antigravity(payload, "Exp404 Hard")
        coordinates: set[tuple[str, int]] = set()
        required = {"abstention_reason", "candidate_index", "confidence_by_alpha", "contains_evaluation_metrics", "contains_test_labels", "dataset", "eligibility", "scores_by_alpha"}
        for row in rows:
            if set(row) != required or row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False or type(row.get("dataset")) is not str or not row["dataset"] or type(row.get("candidate_index")) is not int or row["candidate_index"] < 0:
                raise ValueError("Exp404 Hard row violates frozen label-free schema")
            coordinate = (row["dataset"], row["candidate_index"])
            if coordinate in coordinates:
                raise ValueError("Exp404 Hard contains duplicate coordinates")
            coordinates.add(coordinate)
        if len(coordinates) != 2218:
            raise ValueError("Exp404 Hard coordinate population differs")
        _assert_pinned_leaf_antigravity(parent_fd, pinned)
        _assert_directory_antigravity(None, _EXP404_HARD_ANTIGRAVITY.parent, parent_fd, parent_identity)
        return HardPredictionsAntigravity(digest, frozenset(coordinates))
    finally:
        if pinned is not None:
            os.close(pinned.descriptor)
        if parent_fd >= 0:
            os.close(parent_fd)


def open_test_labels_antigravity(identity: PublicationIdentityAntigravity, independently_confirmed: PublicationIdentityAntigravity | None):
    """Open the one-use TEST-label session only after byte/root-identical captures."""
    if not isinstance(identity, PublicationIdentityAntigravity) or not isinstance(independently_confirmed, PublicationIdentityAntigravity) or identity != independently_confirmed:
        raise ValueError("two byte/root-identical Exp446 captures are required before labels")
    fresh_first = capture_exp446_publication_antigravity()
    fresh_second = capture_exp446_publication_antigravity()
    if fresh_first is fresh_second or any(capture != identity for capture in (fresh_first, fresh_second)):
        raise ValueError("fresh Exp446 source captures do not attest the outer authority")
    descriptor, expected = _verify_database_antigravity()
    try:
        import sqlite3
        uri = f"file:/dev/fd/{descriptor}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    except BaseException:
        os.close(descriptor)
        raise
    expected_order = identity.dataset_order
    next_index = 0
    closed = False
    def authorizer(action: int, table: str | None, column: str | None, _database: str | None, _trigger: str | None) -> int:
        permitted = {
            ("instances", "instance_index"), ("instances", "label"), ("instances", "split"), ("instances", "dataset_id"),
            ("datasets", "id"), ("datasets", "name"),
        }
        if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and (table, column) in permitted):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    connection.set_authorizer(authorizer)
    class LabelSessionAntigravity:
        __slots__ = ()
        def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
            nonlocal next_index
            if closed or next_index >= len(expected_order) or dataset != expected_order[next_index]:
                raise ValueError("TEST labels must be read once in frozen dataset order")
            rows = tuple(connection.execute("SELECT i.instance_index, i.label FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index", (dataset,)))
            if any(type(index) is not int or type(label) is not int or label not in (0, 1) for index, label in rows):
                raise ValueError("TEST label row is invalid")
            if tuple(index for index, _ in rows) != tuple(range(len(rows))):
                raise ValueError("TEST indices are not contiguous")
            next_index += 1
            return rows
        def close_antigravity(self) -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                connection.close()
                opened = os.fstat(descriptor)
                if _identity_antigravity(opened) != expected or opened.st_nlink != 1:
                    raise ValueError("source database changed during TEST-label session")
                rechecked_descriptor, rechecked = _verify_database_antigravity()
                try:
                    if rechecked != expected:
                        raise ValueError("source database changed during TEST-label session")
                finally:
                    os.close(rechecked_descriptor)
            finally:
                os.close(descriptor)
    return LabelSessionAntigravity()


__all__ = ["HardPredictionsAntigravity", "capture_exp446_publication_antigravity", "capture_hard_predictions_antigravity", "open_test_labels_antigravity"]
