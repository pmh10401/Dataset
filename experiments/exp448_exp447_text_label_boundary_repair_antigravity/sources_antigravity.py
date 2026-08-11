"""Exp448 predecessor authority and the sole exact TEXT label converter."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from types import MappingProxyType
import json
import subprocess

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
_EXP447_FAILURE_AUDIT_ANTIGRAVITY = _DATASET_ROOT_ANTIGRAVITY / "evaluation_audit_89aff47cdd7c43cd919a34f81982bc8b_antigravity.json"
_EXP447_FAILURE_AUDIT_SHA256_ANTIGRAVITY = "9f8f7d3d9b92670988ca1d0bb689711d12ef1536777881f18991143a090a2b97"
_EXP447_FAILURE_AUDIT_INODE_ANTIGRAVITY = 19812443
_EXP447_FAILURE_AUDIT_NLINK_ANTIGRAVITY = 1
_EXP447_RUN_ROOT_ANTIGRAVITY = _DATASET_ROOT_ANTIGRAVITY / "exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811"
_EXP447_REVIEWED_COMMIT_ANTIGRAVITY = "7e85d5c7948f8592db80f958d746f31138c67bfb"
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_AUDIT_FACTS_ANTIGRAVITY = {
    "schema": "exp447_evaluation_audit", "phase": "label_snapshot",
    "query_attempts": 1, "query_completions": 0, "label_rows": 0,
    "label_session_closed": True, "label_session_state": "closed",
    "destination_state": "absent", "error_type": "ValueError",
    "error": "TEST label row is invalid",
}



def _assert_exp447_byte_guard_antigravity() -> None:
    try:
        subprocess.check_output((
            "git", "-C", str(_REPOSITORY_ANTIGRAVITY), "diff", "--exit-code",
            _EXP447_REVIEWED_COMMIT_ANTIGRAVITY, "--",
            "experiments/exp447_exp446_postfreeze_performance_evaluation_antigravity",
            "scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py",
        ), stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Exp447 predecessor bytes differ from the frozen failure truth") from error


def require_absent_exp447_root_antigravity() -> None:
    try:
        info = _EXP447_RUN_ROOT_ANTIGRAVITY.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("Exp447 predecessor root absence cannot be verified") from error
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("Exp447 predecessor root is a symlink")
    raise ValueError("Exp447 predecessor root unexpectedly exists")


def verify_exp447_failure_authority_antigravity() -> None:
    """Authenticate the immutable Exp447 failure audit under one no-follow pin."""
    try:
        before = os.stat(_EXP447_FAILURE_AUDIT_ANTIGRAVITY, follow_symlinks=False)
        descriptor = os.open(_EXP447_FAILURE_AUDIT_ANTIGRAVITY, _LEAF_FLAGS_ANTIGRAVITY)
    except OSError as error:
        raise ValueError("Exp447 failure audit is absent or unsafe") from error
    try:
        opened = os.fstat(descriptor)
        identity = _identity_antigravity(before)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != _EXP447_FAILURE_AUDIT_NLINK_ANTIGRAVITY
            or before.st_ino != _EXP447_FAILURE_AUDIT_INODE_ANTIGRAVITY
            or _identity_antigravity(opened) != identity
        ):
            raise ValueError("Exp447 failure audit inode/link contract is invalid")
        payload = _read_pinned_leaf_antigravity(_PinnedLeafAntigravity(
            _EXP447_FAILURE_AUDIT_ANTIGRAVITY.name, descriptor, identity,
        ))
        if hashlib.sha256(payload).hexdigest() != _EXP447_FAILURE_AUDIT_SHA256_ANTIGRAVITY:
            raise ValueError("Exp447 failure audit digest differs from frozen truth")
        try:
            audit = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Exp447 failure audit is not JSON") from error
        if type(audit) is not dict or any(audit.get(name) != expected for name, expected in _AUDIT_FACTS_ANTIGRAVITY.items()):
            raise ValueError("Exp447 failure audit facts differ from frozen truth")
        current = os.stat(_EXP447_FAILURE_AUDIT_ANTIGRAVITY, follow_symlinks=False)
        if _identity_antigravity(current) != identity or _identity_antigravity(os.fstat(descriptor)) != identity:
            raise ValueError("Exp447 failure audit changed during verification")
    finally:
        os.close(descriptor)


def convert_test_label_row_antigravity(instance_index: object, raw_label: object) -> tuple[int, int]:
    """Accept exactly Python int index and TEXT '0'/'1', then discard the raw text."""
    if type(instance_index) is not int:
        raise ValueError("TEST instance_index must be an exact integer")
    if type(raw_label) is not str or raw_label not in ("0", "1"):
        raise ValueError("TEST label row is invalid")
    return instance_index, 0 if raw_label == "0" else 1


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


def open_test_labels_antigravity(identity: PublicationIdentityAntigravity, independently_confirmed: PublicationIdentityAntigravity | None, hard: HardPredictionsAntigravity | None):
    """Open labels only after the authenticated audit, source captures, and Exp404 authority."""
    _assert_exp447_byte_guard_antigravity()
    verify_exp447_failure_authority_antigravity()
    require_absent_exp447_root_antigravity()
    if (not isinstance(identity, PublicationIdentityAntigravity)
            or not isinstance(independently_confirmed, PublicationIdentityAntigravity)
            or identity is independently_confirmed or identity != independently_confirmed):
        raise ValueError("two distinct byte/root-identical Exp446 captures are required before labels")
    if not isinstance(hard, HardPredictionsAntigravity) or hard.sha256 != FROZEN_CONFIG_ANTIGRAVITY.exp404_hard_prediction_sha256 or len(hard.coordinates) != 2218:
        raise ValueError("verified Exp404 Hard authority is required before labels")
    fresh_first = capture_exp446_publication_antigravity()
    fresh_second = capture_exp446_publication_antigravity()
    if fresh_first is fresh_second or any(capture is identity or capture is independently_confirmed or capture != identity for capture in (fresh_first, fresh_second)):
        raise ValueError("fresh Exp446 source captures do not attest the outer authority")
    fresh_hard = capture_hard_predictions_antigravity(fresh_first)
    if hard is fresh_hard or hard != fresh_hard:
        raise ValueError("fresh Exp404 Hard capture does not attest the supplied authority")
    descriptor, expected = _verify_database_antigravity()
    connection = None
    try:
        import sqlite3
        uri = f"file:/dev/fd/{descriptor}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        def authorizer(action: int, table: str | None, column: str | None, _database: str | None, _trigger: str | None) -> int:
            permitted = {
                ("instances", "instance_index"), ("instances", "label"), ("instances", "split"), ("instances", "dataset_id"),
                ("datasets", "id"), ("datasets", "name"),
            }
            if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and (table, column) in permitted):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        connection.set_authorizer(authorizer)
    except BaseException:
        if connection is not None:
            try:
                connection.close()
            except BaseException:
                pass
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    expected_order = identity.dataset_order
    next_index = 0
    closed = False
    close_failure: BaseException | None = None
    descriptor_released = False

    def release_descriptor() -> None:
        nonlocal descriptor_released
        if not descriptor_released:
            os.close(descriptor)
            descriptor_released = True

    class LabelSessionAntigravity:
        __slots__ = ()

        def close_state_antigravity(self) -> str:
            return "closed" if closed else ("close_failed" if close_failure is not None else "open")

        def close_error_antigravity(self) -> BaseException | None:
            return close_failure

        def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
            nonlocal next_index
            if closed or close_failure is not None or next_index >= len(expected_order) or dataset != expected_order[next_index]:
                raise ValueError("TEST labels must be read once in frozen dataset order")
            rows = tuple(connection.execute("SELECT i.instance_index, i.label FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index", (dataset,)))
            converted = tuple(convert_test_label_row_antigravity(index, raw_label) for index, raw_label in rows)
            if tuple(index for index, _ in converted) != tuple(range(len(converted))):
                raise ValueError("TEST indices are not contiguous")
            next_index += 1
            return converted
        def close_antigravity(self) -> None:
            nonlocal closed, close_failure
            if closed:
                return
            if close_failure is not None:
                raise close_failure
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
                release_descriptor()
            except BaseException as error:
                close_failure = error
                try:
                    release_descriptor()
                except OSError:
                    pass
                raise
            closed = True
    return LabelSessionAntigravity()


__all__ = ["HardPredictionsAntigravity", "capture_exp446_publication_antigravity", "capture_hard_predictions_antigravity", "convert_test_label_row_antigravity", "open_test_labels_antigravity", "require_absent_exp447_root_antigravity", "verify_exp447_failure_authority_antigravity"]
