"""Frozen Exp442 source replay and values-only SQLite boundary for Exp444.

The module has one narrow prediction input edge: frozen Exp442 publication
bytes, the frozen Exp404 Hard prediction bytes, and a descriptor-pinned SQLite
database.  No TEST label-capable query is registered.  The database session
returns copied float32 values only; its raw sqlite connection and cursors stay
inside method closures and are never exposed to callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
from urllib.parse import quote
import weakref

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CandidateInputAntigravity,
    FrozenSourceSnapshotAntigravity,
    SourceCandidateAntigravity,
    candidate_attestation_sha256_antigravity,
)


NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY = """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index"""

BOUND_TEST_VALUE_SQL_ANTIGRAVITY = """SELECT i.instance_index, i.values_blob
FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST' AND i.instance_index = ?"""

_ALLOWED_SQL = frozenset((NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, BOUND_TEST_VALUE_SQL_ANTIGRAVITY))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY: Callable[[Path], None] | None = None
_DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY: Callable[[Path], None] | None = None
_CLOSE_STEP_HOOK_ANTIGRAVITY: Callable[[str, Path, Path], None] | None = None


class DuplicateJsonKeyError(ValueError):
    """Raised instead of silently accepting the last duplicate JSON key."""


@dataclass(frozen=True)
class FrozenSourcePathsAntigravity:
    database: Path
    exp404_hard_prediction: Path
    exp442_prediction: Path
    exp442_candidate_ranking: Path
    exp442_selection: Path
    exp442_artifact_map: Path
    exp442_manifest: Path
    exp442_source_audit: Path
    exp442_status: Path


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _regular_metadata(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_regular_bytes_antigravity(path: Path, expected_sha256: str | None = None) -> tuple[bytes, str]:
    """Read one exact regular inode and reject replacement during the read."""
    path = Path(path)
    try:
        initial = path.lstat()
    except OSError as error:
        raise ValueError(f"source is unavailable: {path}") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode) or initial.st_nlink < 1:
        raise ValueError(f"source is not a regular linked file: {path}")
    hook = _SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY
    if hook is not None:
        hook(path)
    try:
        descriptor = os.open(path, _FILE_FLAGS)
    except OSError as error:
        raise ValueError(f"source open failed: {path}") from error
    try:
        before = os.fstat(descriptor)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or _regular_metadata(before) != _regular_metadata(initial):
            raise ValueError(f"source inode changed before read: {path}")
        digest = sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        named = path.lstat()
        if _regular_metadata(before) != _regular_metadata(after) or _regular_metadata(after) != _regular_metadata(named) or stat.S_ISLNK(named.st_mode):
            raise ValueError(f"source changed while reading: {path}")
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(f"source SHA256 mismatch: {path.name}")
        return b"".join(chunks), observed
    except OSError as error:
        raise ValueError(f"source read failed: {path}") from error
    finally:
        os.close(descriptor)


def _parse_json_object_antigravity(payload: bytes, name: str) -> Mapping[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DuplicateJsonKeyError(f"{name} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"{name} contains {token}")),
        )
    except DuplicateJsonKeyError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _parse_jsonl_antigravity(payload: bytes, name: str) -> tuple[Mapping[str, object], ...]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{name} is truncated JSONL")
    rows: list[Mapping[str, object]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{name} has a blank row")
        rows.append(_parse_json_object_antigravity(line, f"{name} row {number}"))
    return tuple(rows)


def _canonical_coordinate_identity_antigravity(coordinates: Sequence[tuple[str, int]]) -> str:
    values = tuple(coordinates)
    if len(values) != len(set(values)):
        raise ValueError("selected coordinates are duplicated")
    if any(type(dataset) is not str or not dataset or type(index) is not int or index < 0 for dataset, index in values):
        raise ValueError("selected coordinates are invalid")
    payload = b"".join(
        json.dumps({"dataset": dataset, "instance_index": index}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        for dataset, index in sorted(values)
    )
    return sha256(payload).hexdigest()


def selected_coordinate_identity_sha256_antigravity(coordinates: Sequence[tuple[str, int]]) -> str:
    return _canonical_coordinate_identity_antigravity(coordinates)


def _candidate_signature_antigravity(candidate: SourceCandidateAntigravity) -> tuple[object, ...]:
    return (candidate.dataset, candidate.instance_index, candidate.raw_length, candidate.original_depths, candidate.original_rank)


def _normalize_sql_antigravity(sql: str) -> str:
    if type(sql) is not str:
        raise ValueError("prediction SQL must be text")
    return " ".join(sql.strip().split()).rstrip(";").strip()


def validate_prediction_sql_antigravity(sql: str) -> bool:
    """Allow only the two exact values-only statements before SQLite runs."""
    normalized = _normalize_sql_antigravity(sql)
    allowed = { _normalize_sql_antigravity(item) for item in _ALLOWED_SQL }
    if normalized in allowed:
        return True
    lowered = normalized.casefold()
    if "labels_blob" in lowered or (re.search(r"\blabel\b", lowered) and re.search(r"split\s*=\s*'test'", lowered)):
        raise ValueError("label-capable SQL is forbidden")
    if "select *" in lowered or "unknown" in lowered:
        raise ValueError("unregistered prediction SQL")
    raise ValueError("unregistered prediction SQL")


def _validate_query_parameters_antigravity(
    sql: str,
    parameters: object,
    *,
    allowed_train_datasets: frozenset[str],
    allowed_test_coordinates: frozenset[tuple[str, int]],
) -> tuple[object, ...]:
    normalized = _normalize_sql_antigravity(sql)
    validate_prediction_sql_antigravity(sql)
    if normalized.casefold() == _normalize_sql_antigravity(NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY).casefold():
        if type(parameters) not in (tuple, list) or len(parameters) != 1 or type(parameters[0]) is not str or not parameters[0]:
            raise ValueError("TRAIN values SQL requires one dataset parameter")
        if parameters[0] not in allowed_train_datasets:
            raise ValueError("unbound TRAIN dataset")
        return (parameters[0],)
    if type(parameters) not in (tuple, list) or len(parameters) != 2 or type(parameters[0]) is not str or not parameters[0] or type(parameters[1]) is not int or parameters[1] < 0:
        raise ValueError("TEST values SQL requires dataset and instance parameters")
    if (parameters[0], parameters[1]) not in allowed_test_coordinates:
        raise ValueError("unbound TEST coordinate")
    return (parameters[0], parameters[1])


def _decode_values_blob_antigravity(blob: object, name: str) -> np.ndarray:
    if type(blob) is not bytes or not blob or len(blob) % 4:
        raise ValueError(f"{name} values blob is invalid")
    values = np.frombuffer(blob, dtype="<f4").copy()
    return values


def _validated_candidate_bounds_antigravity(
    candidates: Sequence[SourceCandidateAntigravity],
) -> tuple[frozenset[str], frozenset[tuple[str, int]], tuple[tuple[object, ...], ...], str]:
    population = tuple(candidates)
    if len(population) != FROZEN_CONFIG.source_candidate_count or any(
        not isinstance(item, SourceCandidateAntigravity) for item in population
    ):
        raise ValueError("database session requires exactly 30 frozen candidates")
    datasets = frozenset(item.dataset for item in population)
    coordinates = frozenset(item.coordinate for item in population)
    if len(datasets) != len(population) or len(coordinates) != len(population):
        raise ValueError("database session requires one frozen candidate per dataset")
    expected_rank_order = tuple(range(1, FROZEN_CONFIG.source_candidate_count + 1))
    if tuple(item.original_rank for item in population) != expected_rank_order:
        raise ValueError("database session candidate rank order is not 1-30")
    identity = selected_coordinate_identity_sha256_antigravity(tuple(coordinates))
    if identity != FROZEN_CONFIG.exp442_selected_coordinate_identity_sha256:
        raise ValueError("database session candidate identity is not frozen")
    signatures = tuple(_candidate_signature_antigravity(item) for item in population)
    attestation = candidate_attestation_sha256_antigravity(population)
    return datasets, coordinates, signatures, attestation


_VERIFIED_REPLAY_TOKEN_ANTIGRAVITY = object()


class _VerifiedReplayAttestationAntigravity:
    """Module-private capability created only after exact frozen replay."""

    __slots__ = ("source", "database_sha256", "datasets", "coordinates", "signatures", "candidate_attestation_sha256")

    def __init__(
        self,
        token: object,
        source: Path,
        database_sha256: str,
        candidates: Sequence[SourceCandidateAntigravity],
    ) -> None:
        if token is not _VERIFIED_REPLAY_TOKEN_ANTIGRAVITY:
            raise ValueError("verified replay attestation is required")
        datasets, coordinates, signatures, attestation = _validated_candidate_bounds_antigravity(candidates)
        self.source = Path(source)
        self.database_sha256 = _digest(database_sha256, "verified replay database SHA256")
        self.datasets = datasets
        self.coordinates = coordinates
        self.signatures = signatures
        self.candidate_attestation_sha256 = attestation


class ValuesOnlyDatabaseSessionAntigravity:
    """Parent-owned pin/session with closure-only SQLite handles."""

    __slots__ = (
        "_source_path", "_pin_path", "_root", "_descriptor", "_source_metadata",
        "_pin_inode", "_source_nlink", "_database_sha256", "_closed", "_label_access_count",
        "_allowed_train_datasets", "_allowed_test_coordinates", "_bound_candidate_signatures",
        "_candidate_attestation_sha256",
        "denied_sql_attempts", "sql_trace", "_active_sql", "_finalizer", "__weakref__",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ValueError("verified replay session must be created by source verification")

    @classmethod
    def _from_verified_replay_antigravity(
        cls,
        attestation: _VerifiedReplayAttestationAntigravity,
        token: object,
    ) -> "ValuesOnlyDatabaseSessionAntigravity":
        if token is not _VERIFIED_REPLAY_TOKEN_ANTIGRAVITY or not isinstance(attestation, _VerifiedReplayAttestationAntigravity):
            raise ValueError("verified replay attestation is required")
        instance = cls.__new__(cls)
        instance._initialize_antigravity(
            attestation.source,
            attestation.database_sha256,
            attestation.datasets,
            attestation.coordinates,
            attestation.signatures,
            attestation.candidate_attestation_sha256,
        )
        return instance

    def _initialize_antigravity(
        self,
        source: Path,
        expected_sha256: str,
        allowed_train_datasets: frozenset[str],
        allowed_test_coordinates: frozenset[tuple[str, int]],
        bound_candidate_signatures: tuple[tuple[object, ...], ...],
        candidate_attestation_sha256: str,
    ) -> None:
        self._source_path = Path(source)
        self._pin_path: Path | None = None
        self._root: Path | None = None
        self._descriptor: int | None = None
        self._source_metadata: tuple[int, int, int, int, int, int] | None = None
        self._pin_inode: tuple[int, int] | None = None
        self._source_nlink: int | None = None
        self._database_sha256 = ""
        self._closed = False
        self._label_access_count = 0
        self._allowed_train_datasets = allowed_train_datasets
        self._allowed_test_coordinates = allowed_test_coordinates
        self._bound_candidate_signatures = bound_candidate_signatures
        self._candidate_attestation_sha256 = candidate_attestation_sha256
        self.denied_sql_attempts = 0
        self.sql_trace: list[str] = []
        self._active_sql: str | None = None
        self._finalizer = None

        source = self._source_path
        try:
            initial = source.lstat()
        except OSError as error:
            raise ValueError("database pathname is unavailable") from error
        if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode) or initial.st_nlink < 1:
            raise ValueError("database pathname must be a linked regular file")
        hook = _DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY
        if hook is not None:
            hook(source)
        descriptor: int | None = None
        root: Path | None = None
        pin: Path | None = None
        linked = False
        try:
            descriptor = os.open(source, _FILE_FLAGS)
            before = os.fstat(descriptor)
            if _regular_metadata(before) != _regular_metadata(initial):
                raise ValueError("database pathname changed before hashing")
            digest = sha256()
            os.lseek(descriptor, 0, os.SEEK_SET)
            while block := os.read(descriptor, 1024 * 1024):
                digest.update(block)
            os.lseek(descriptor, 0, os.SEEK_SET)
            observed = digest.hexdigest()
            if observed != expected_sha256:
                raise ValueError("database SHA256 mismatch")
            after = os.fstat(descriptor)
            named = source.lstat()
            if _regular_metadata(after) != _regular_metadata(before) or _regular_metadata(named) != _regular_metadata(after):
                raise ValueError("database changed while hashing")
            root = Path(tempfile.mkdtemp(prefix="exp444_values_pin_antigravity_"))
            pin = root / "database_pin_antigravity.db"
            os.link(source, pin, follow_symlinks=False)
            linked = True
            pin_info = pin.lstat()
            operational = os.fstat(descriptor)
            if (pin_info.st_dev, pin_info.st_ino) != (operational.st_dev, operational.st_ino) or operational.st_nlink < before.st_nlink + 1:
                raise ValueError("database pin does not bind the source inode")
            named_operational = source.lstat()
            if _regular_metadata(named_operational) != _regular_metadata(operational):
                raise ValueError("database pathname changed while pinning")
            self._pin_path = pin
            self._root = root
            self._descriptor = descriptor
            self._source_metadata = _regular_metadata(operational)
            self._pin_inode = (pin_info.st_dev, pin_info.st_ino)
            self._source_nlink = before.st_nlink
            self._database_sha256 = observed
            self._finalizer = weakref.finalize(self, ValuesOnlyDatabaseSessionAntigravity._finalize, descriptor, pin, root)
            descriptor = None
        except BaseException:
            if linked and pin is not None:
                try:
                    pin.unlink()
                except OSError:
                    pass
            if root is not None:
                try:
                    root.rmdir()
                except OSError:
                    pass
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _finalize(descriptor: int, pin: Path, root: Path) -> None:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            pin.unlink()
        except OSError:
            pass
        try:
            root.rmdir()
        except OSError:
            pass

    @property
    def label_access_count(self) -> int:
        return self._label_access_count

    @property
    def execution_count(self) -> int:
        return len(self.sql_trace)

    @property
    def database_sha256(self) -> str:
        return self._database_sha256

    @property
    def pin_path(self) -> Path:
        if self._pin_path is None:
            raise ValueError("database session is closed")
        return self._pin_path

    @property
    def descriptor_inode(self) -> tuple[int, int]:
        if self._pin_inode is None:
            raise ValueError("database session is closed")
        return self._pin_inode

    def _require_live(self) -> None:
        if self._closed or self._descriptor is None or self._pin_path is None or self._source_metadata is None:
            raise ValueError("database session is closed")

    def _verify_boundary(self, *, rehash: bool = False) -> None:
        self._require_live()
        descriptor = self._descriptor
        assert descriptor is not None and self._pin_path is not None and self._source_metadata is not None and self._pin_inode is not None
        current = os.fstat(descriptor)
        named = self._source_path.lstat()
        pin = self._pin_path.lstat()
        if _regular_metadata(current) != self._source_metadata or _regular_metadata(named) != self._source_metadata or _regular_metadata(pin) != self._source_metadata or (pin.st_dev, pin.st_ino) != self._pin_inode:
            raise ValueError("database source or pin identity changed")
        if rehash:
            digest = sha256()
            os.lseek(descriptor, 0, os.SEEK_SET)
            while block := os.read(descriptor, 1024 * 1024):
                digest.update(block)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if digest.hexdigest() != self._database_sha256:
                raise ValueError("database descriptor SHA256 changed")

    def _authorizer(self, action: int, argument_one: str | None, argument_two: str | None, _database: str | None, _trigger: str | None) -> int:
        if action == sqlite3.SQLITE_READ:
            table = (argument_one or "").casefold()
            column = (argument_two or "").casefold()
            if table not in {"instances", "datasets"}:
                return sqlite3.SQLITE_DENY
            if table == "instances" and column not in {"instance_index", "values_blob", "dataset_id", "split", "label"}:
                return sqlite3.SQLITE_DENY
            if table == "datasets" and column not in {"id", "name"}:
                return sqlite3.SQLITE_DENY
            if column == "labels_blob":
                return sqlite3.SQLITE_DENY
            if column == "label" and _normalize_sql_antigravity(self._active_sql or "").casefold() != _normalize_sql_antigravity(NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY).casefold():
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def query_antigravity(self, sql: str, parameters: object = ()) -> tuple[tuple[object, ...], ...]:
        """Execute one registered statement and return detached tuples only."""
        self._verify_boundary()
        try:
            bound = _validate_query_parameters_antigravity(
                sql,
                parameters,
                allowed_train_datasets=self._allowed_train_datasets,
                allowed_test_coordinates=self._allowed_test_coordinates,
            )
        except ValueError:
            self.denied_sql_attempts += 1
            raise
        self._active_sql = _normalize_sql_antigravity(sql)
        connection: sqlite3.Connection | None = None
        pin_fd: int | None = None
        try:
            assert self._pin_path is not None and self._pin_inode is not None and self._source_metadata is not None
            pin_fd = os.open(self._pin_path, _FILE_FLAGS)
            pin_info = os.fstat(pin_fd)
            if (pin_info.st_dev, pin_info.st_ino) != self._pin_inode or _regular_metadata(pin_info) != self._source_metadata:
                raise ValueError("database pin changed before query")
            uri_root = Path("/dev/fd") if Path("/dev/fd").is_dir() else Path("/proc/self/fd")
            uri = f"file:{quote(str(uri_root / str(pin_fd)), safe='/')}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True)
            connection.set_authorizer(self._authorizer)
            connection.execute("PRAGMA query_only=ON")
            self.sql_trace.append(self._active_sql)
            cursor = connection.execute(sql, bound)
            rows = tuple(tuple(row) for row in cursor.fetchall())
            return rows
        except sqlite3.Error as error:
            raise ValueError("values-only SQLite query failed") from error
        finally:
            self._active_sql = None
            if connection is not None:
                connection.close()
            if pin_fd is not None:
                os.close(pin_fd)
            self._verify_boundary()

    def _invoke_close_hook_antigravity(self, step: str, pin: Path, root: Path) -> None:
        hook = _CLOSE_STEP_HOOK_ANTIGRAVITY
        if hook is not None:
            hook(step, pin, root)

    def _restore_failure_pin_antigravity(self, pin: Path) -> None:
        assert self._pin_inode is not None and self._source_nlink is not None
        if pin.exists():
            restored = pin.lstat()
        else:
            named = self._source_path.lstat()
            if (named.st_dev, named.st_ino) != self._pin_inode:
                raise ValueError("database source cannot restore failure pin")
            os.link(self._source_path, pin, follow_symlinks=False)
            restored = pin.lstat()
        if (restored.st_dev, restored.st_ino) != self._pin_inode or restored.st_nlink != self._source_nlink + 1:
            raise ValueError("restored failure pin identity is invalid")

    def close(self) -> None:
        if self._closed:
            return
        descriptor = self._descriptor
        pin = self._pin_path
        root = self._root
        assert descriptor is not None and pin is not None and root is not None
        pin_removed = False
        try:
            # Validate every condition that can be checked while the evidence
            # pin still exists.  In particular, no unrelated file may occupy
            # the private root and no external hard link may have appeared.
            self._verify_boundary(rehash=True)
            if tuple(root.iterdir()) != (pin,):
                raise ValueError("database failure root contains unexpected files")
            pinned = pin.lstat()
            if self._source_nlink is None or pinned.st_nlink != self._source_nlink + 1:
                raise ValueError("database source link count changed before close")

            self._invoke_close_hook_antigravity("before_descriptor_close", pin, root)
            os.close(descriptor)
            descriptor = None
            self._descriptor = None

            self._invoke_close_hook_antigravity("before_pin_unlink", pin, root)
            pin.unlink()
            pin_removed = True
            self._invoke_close_hook_antigravity("after_pin_unlink", pin, root)

            named = self._source_path.lstat()
            if self._pin_inode is None or (named.st_dev, named.st_ino) != self._pin_inode:
                raise ValueError("database source inode changed during close")
            if self._source_nlink is None or named.st_nlink != self._source_nlink:
                raise ValueError("database source link count changed after close")
            self._invoke_close_hook_antigravity("before_root_rmdir", pin, root)
            root.rmdir()
        except BaseException as error:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as close_error:
                    error.add_note(f"database descriptor cleanup failed: {close_error}")
                self._descriptor = None
            if pin_removed:
                try:
                    self._restore_failure_pin_antigravity(pin)
                except BaseException as restore_error:
                    error.add_note(f"failure pin restoration failed: {restore_error}")
            self._closed = True
            if self._finalizer is not None:
                # A failed close is durable evidence: garbage collection must
                # not erase the retained failure pin/root.
                self._finalizer.detach()
            raise ValueError("pinned database close failed") from error
        else:
            self._closed = True
            if self._finalizer is not None:
                self._finalizer.detach()
            self._pin_path = None
            self._root = None
            self._pin_inode = None
            self._source_metadata = None
            self._source_nlink = None

    def __enter__(self) -> "ValuesOnlyDatabaseSessionAntigravity":
        self._require_live()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _fixture_value_antigravity(fixture: object, *names: str) -> object | None:
    if isinstance(fixture, Mapping):
        for name in names:
            if name in fixture:
                return fixture[name]
    for name in names:
        if hasattr(fixture, name):
            return getattr(fixture, name)
    return None


def _source_paths_antigravity(fixture: object | None) -> FrozenSourcePathsAntigravity:
    if fixture is None:
        database = Path(FROZEN_CONFIG.database_path)
        hard = Path(FROZEN_CONFIG.exp404_hard_prediction_path)
        prediction = Path(FROZEN_CONFIG.exp442_prediction_path)
        ranking = Path(FROZEN_CONFIG.exp442_candidate_ranking_path)
        selection = Path(FROZEN_CONFIG.exp442_selection_path)
    elif isinstance(fixture, (str, os.PathLike)):
        root = Path(fixture)
        database = root / "univariate_ts.db"
        hard = root / "exp404_hard_confidence.jsonl"
        prediction = root / "prediction_antigravity.jsonl"
        ranking = root / "candidate_ranking_antigravity.jsonl"
        selection = root / "selection_antigravity.json"
    else:
        def required(names: tuple[str, ...], label: str) -> Path:
            value = _fixture_value_antigravity(fixture, *names)
            if value is None:
                raise ValueError(f"fixture must bind exact {label} path")
            return Path(value)

        database = required(("database", "database_path"), "database")
        hard = required(("exp404_hard_prediction", "hard_prediction", "exp404_hard_prediction_path"), "Exp404 Hard prediction")
        prediction = required(("exp442_prediction", "prediction", "exp442_prediction_path"), "Exp442 prediction")
        ranking = required(("exp442_candidate_ranking", "candidate_ranking", "exp442_candidate_ranking_path"), "Exp442 candidate ranking")
        selection = required(("exp442_selection", "selection", "exp442_selection_path"), "Exp442 selection")
    publication = prediction.parent
    artifact_map = Path(_fixture_value_antigravity(fixture, "exp442_artifact_map", "artifact_map") or publication / "artifact_sha256_antigravity.json") if fixture is not None else publication / "artifact_sha256_antigravity.json"
    manifest = Path(_fixture_value_antigravity(fixture, "exp442_manifest", "manifest") or publication / "manifest_antigravity.json") if fixture is not None else publication / "manifest_antigravity.json"
    audit = Path(_fixture_value_antigravity(fixture, "exp442_source_audit", "source_audit") or publication / "source_audit_antigravity.json") if fixture is not None else publication / "source_audit_antigravity.json"
    status = Path(_fixture_value_antigravity(fixture, "exp442_status", "status") or publication / "status_antigravity.json") if fixture is not None else publication / "status_antigravity.json"
    return FrozenSourcePathsAntigravity(database, hard, prediction, ranking, selection, artifact_map, manifest, audit, status)


def _validate_candidate_row_antigravity(row: Mapping[str, object], *, rank_kind: str, allow_unselected: bool = False) -> SourceCandidateAntigravity | None:
    expected = {"agreement_count", "dataset", "depths", "global_rank", "instance_index", "raw_length", "second_empirical_rank", "second_excess_ratio", "selected", "selection_rank"}
    if set(row) != expected:
        raise ValueError(f"{rank_kind} candidate schema is invalid")
    dataset, index = row["dataset"], row["instance_index"]
    depths = row["depths"]
    if type(dataset) is not str or not dataset or type(index) is not int or index < 0 or not isinstance(depths, list) or len(depths) < 2 or tuple(depths) != tuple(sorted(set(depths))) or any(type(item) is not int or item < 1 for item in depths):
        raise ValueError(f"{rank_kind} candidate identity is invalid")
    if type(row["raw_length"]) is not int or row["raw_length"] < 1 or row["agreement_count"] != len(depths):
        raise ValueError(f"{rank_kind} candidate dimensions are invalid")
    for name in ("second_empirical_rank", "second_excess_ratio"):
        if type(row[name]) not in (int, float) or not math.isfinite(float(row[name])) or float(row[name]) < 0.0:
            raise ValueError(f"{rank_kind} candidate score evidence is invalid")
    if type(row["global_rank"]) is not int or row["global_rank"] < 1:
        raise ValueError(f"{rank_kind} candidate rank is invalid")
    if row["selected"] is False and allow_unselected:
        if row["selection_rank"] is not None:
            raise ValueError(f"{rank_kind} unselected rank is invalid")
        return None
    if row["selected"] is not True or type(row["selection_rank"]) is not int or row["selection_rank"] < 1:
        raise ValueError(f"{rank_kind} selected flag is invalid")
    return SourceCandidateAntigravity(dataset, index, row["raw_length"], tuple(depths), row["selection_rank"])


def _validate_artifact_map_antigravity(paths: FrozenSourcePathsAntigravity, expected_hashes: Mapping[str, str]) -> Mapping[str, str]:
    payload, map_digest = _read_regular_bytes_antigravity(paths.exp442_artifact_map)
    artifact_map = _parse_json_object_antigravity(payload, "Exp442 artifact map")
    if set(artifact_map) != {"files", "schema", "self_excluded"} or artifact_map["schema"] != "exp442.artifact-map.v1" or artifact_map["self_excluded"] != "artifact_sha256_antigravity.json":
        raise ValueError("Exp442 artifact map is invalid")
    files = artifact_map["files"]
    if not isinstance(files, Mapping) or set(files) != {"candidate_ranking_antigravity.jsonl", "manifest_antigravity.json", "prediction_antigravity.jsonl", "source_audit_antigravity.json", "status_antigravity.json"}:
        raise ValueError("Exp442 artifact map file set is invalid")
    resolved = {
        "candidate_ranking_antigravity.jsonl": paths.exp442_candidate_ranking,
        "manifest_antigravity.json": paths.exp442_manifest,
        "prediction_antigravity.jsonl": paths.exp442_prediction,
        "source_audit_antigravity.json": paths.exp442_source_audit,
        "status_antigravity.json": paths.exp442_status,
    }
    observed: dict[str, str] = {}
    for name, path in resolved.items():
        expected = files[name]
        _digest(expected, f"artifact map {name}")
        _, digest = _read_regular_bytes_antigravity(path, expected)
        observed[name] = digest
    if observed["prediction_antigravity.jsonl"] != expected_hashes["prediction"] or observed["candidate_ranking_antigravity.jsonl"] != expected_hashes["ranking"]:
        raise ValueError("Exp442 artifact map does not bind frozen prediction hashes")
    return MappingProxyType({"artifact_map": map_digest, **observed})


def _validate_publication_manifest_antigravity(payload: bytes, expected_hashes: Mapping[str, str]) -> None:
    manifest = _parse_json_object_antigravity(payload, "Exp442 publication manifest")
    required = {"adaptive_followup", "bootstrap_interpretation", "candidate_budget", "candidate_count", "candidate_per_dataset", "candidate_ranking_sha256", "confirmatory_statistical_claim", "contains_test_labels", "dataset_count", "eligible_candidate_count", "experiment_id", "external_generalization_supported", "hard_coordinate_identity_sha256", "independent_validation", "label_access_count", "lifecycle_profile_sha256", "official_run_purpose", "operational_alert_eligible", "outside_hard_candidate_count", "prediction_sha256", "provisional_candidate_count", "review_only", "reviewed_commit", "run_identity", "schema", "scientific_winner_selected", "selected_coordinate_identity_sha256", "selection_sha256", "source_audit_sha256", "source_snapshot_sha256", "uses_frozen_test_candidate_empty_gate", "uses_test_labels_for_prediction", "winner_count"}
    if set(manifest) != required or manifest["schema"] != "exp442.publication.v1" or manifest["experiment_id"] != "Exp442" or manifest["candidate_count"] != 30 or manifest["candidate_budget"] != 30 or manifest["candidate_per_dataset"] != 1 or manifest["dataset_count"] != 1117 or manifest["winner_count"] != 49 or manifest["label_access_count"] != 0 or manifest["contains_test_labels"] is not False or manifest["uses_test_labels_for_prediction"] is not False:
        raise ValueError("Exp442 publication manifest is invalid")
    if manifest["prediction_sha256"] != expected_hashes["prediction"] or manifest["candidate_ranking_sha256"] != expected_hashes["ranking"] or manifest["selection_sha256"] != expected_hashes["selection"] or manifest["selected_coordinate_identity_sha256"] != FROZEN_CONFIG.exp442_selected_coordinate_identity_sha256:
        raise ValueError("Exp442 publication manifest identity is invalid")


def _parse_selection_antigravity(payload: bytes, expected_hashes: Mapping[str, str]) -> tuple[tuple[SourceCandidateAntigravity, ...], frozenset[tuple[str, int]]]:
    selection = _parse_json_object_antigravity(payload, "Exp442 selection")
    required = {"adaptive_followup", "bootstrap_interpretation", "confirmatory_statistical_claim", "eligible_candidates", "experiment_id", "external_generalization_supported", "hard_coordinates", "independent_validation", "lifecycle_profile_sha256", "official_run_purpose", "operational_alert_eligible", "outside_hard_candidates", "provisional_candidates", "review_only", "reviewed_commit", "run_identity", "scientific_winner_selected", "selected", "source_snapshot_sha256", "winners"}
    if set(selection) != required or selection["experiment_id"] != "Exp442" or selection["review_only"] is not True or selection["scientific_winner_selected"] is not False:
        raise ValueError("Exp442 selection schema/governance is invalid")
    selected_rows = selection["selected"]
    winners = selection["winners"]
    if not isinstance(selected_rows, list) or not isinstance(winners, list) or len(selected_rows) != 30 or len(winners) != 49:
        raise ValueError("Exp442 selection population is invalid")
    selected_candidates = tuple(_validate_candidate_row_antigravity(row, rank_kind="selection") for row in selected_rows if isinstance(row, Mapping))
    if len(selected_candidates) != len(selected_rows) or tuple(item.original_rank for item in selected_candidates) != tuple(range(1, 31)):
        raise ValueError("Exp442 selection ranks are not contiguous 1-30")
    if len({item.coordinate for item in selected_candidates}) != 30 or len({item.dataset for item in selected_candidates}) != 30:
        raise ValueError("Exp442 selection contains duplicate candidates")
    hard_raw = selection["hard_coordinates"]
    if not isinstance(hard_raw, list):
        raise ValueError("Exp442 Hard coordinates are invalid")
    hard: list[tuple[str, int]] = []
    for value in hard_raw:
        if not isinstance(value, list) or len(value) != 2 or type(value[0]) is not str or not value[0] or type(value[1]) is not int or value[1] < 0:
            raise ValueError("Exp442 Hard coordinates are invalid")
        hard.append((value[0], value[1]))
    if len(hard) != len(set(hard)):
        raise ValueError("Exp442 Hard coordinates are duplicated")
    return selected_candidates, frozenset(hard)


def _parse_prediction_and_ranking_antigravity(prediction_payload: bytes, ranking_payload: bytes) -> tuple[tuple[SourceCandidateAntigravity, ...], tuple[SourceCandidateAntigravity, ...]]:
    prediction_rows = _parse_jsonl_antigravity(prediction_payload, "Exp442 prediction")
    ranking_rows = _parse_jsonl_antigravity(ranking_payload, "Exp442 candidate ranking")
    if len(prediction_rows) != 30 or len(ranking_rows) != 49:
        raise ValueError("Exp442 candidate populations are invalid")
    prediction = tuple(_validate_candidate_row_antigravity(row, rank_kind="prediction") for row in prediction_rows)
    ranking_rows_parsed = tuple(_validate_candidate_row_antigravity(row, rank_kind="ranking", allow_unselected=True) for row in ranking_rows)
    ranking = tuple(item for item in ranking_rows_parsed if item is not None)
    if set(item.original_rank for item in prediction) != set(range(1, 31)):
        raise ValueError("Exp442 prediction ranks are not 1-30")
    global_ranks = tuple(row["global_rank"] for row in ranking_rows)
    if global_ranks != tuple(range(1, 50)) or tuple(row["selected"] for row in ranking_rows[:30]) != (True,) * 30 or any(row["selected"] is not False for row in ranking_rows[30:]):
        raise ValueError("Exp442 ranking ranks are not 1-49")
    if tuple(item.original_rank for item in ranking) != tuple(range(1, 31)):
        raise ValueError("Exp442 selected ranking ranks are not 1-30")
    if len({item.coordinate for item in prediction}) != 30 or len({item.dataset for item in prediction}) != 30:
        raise ValueError("Exp442 prediction contains duplicate candidates")
    selected_ranking = tuple(item for item in ranking if item.original_rank <= 30)
    if set(item.coordinate for item in selected_ranking) != set(item.coordinate for item in prediction):
        raise ValueError("Exp442 prediction/ranking selection mismatch")
    return prediction, ranking


def _validate_hard_prediction_antigravity(payload: bytes) -> frozenset[tuple[str, int]]:
    rows = _parse_jsonl_antigravity(payload, "Exp404 Hard prediction")
    coordinates: set[tuple[str, int]] = set()
    for row in rows:
        required = {"abstention_reason", "candidate_index", "confidence_by_alpha", "contains_evaluation_metrics", "contains_test_labels", "dataset", "eligibility", "scores_by_alpha"}
        if set(row) != required or row["contains_test_labels"] is not False or row["contains_evaluation_metrics"] is not False or type(row["dataset"]) is not str or not row["dataset"] or type(row["candidate_index"]) is not int or row["candidate_index"] < 0:
            raise ValueError("Exp404 Hard prediction is invalid")
        coordinate = (row["dataset"], row["candidate_index"])
        if coordinate in coordinates:
            raise ValueError("Exp404 Hard prediction has duplicate coordinates")
        coordinates.add(coordinate)
    return frozenset(coordinates)


def _validate_candidate_replay_agreement_antigravity(
    prediction: Sequence[SourceCandidateAntigravity],
    ranking: Sequence[SourceCandidateAntigravity],
    selection: Sequence[SourceCandidateAntigravity],
) -> None:
    prediction_signatures = tuple(sorted(_candidate_signature_antigravity(item) for item in prediction))
    selection_signatures = tuple(sorted(_candidate_signature_antigravity(item) for item in selection))
    ranking_signatures = tuple(sorted(_candidate_signature_antigravity(item) for item in ranking))
    if prediction_signatures != selection_signatures:
        raise ValueError("Exp442 prediction/selection candidate mismatch")
    if ranking_signatures != selection_signatures:
        raise ValueError("Exp442 ranking/selection candidate mismatch")


def verify_frozen_sources_antigravity(fixture: object | None = None) -> FrozenSourceSnapshotAntigravity:
    """Verify exact Exp442/Exp404 bytes and return a label-free DB session."""
    paths = _source_paths_antigravity(fixture)
    expected = {
        "database": FROZEN_CONFIG.database_sha256,
        "hard": FROZEN_CONFIG.exp404_hard_prediction_sha256,
        "prediction": FROZEN_CONFIG.exp442_prediction_sha256,
        "ranking": FROZEN_CONFIG.exp442_candidate_ranking_sha256,
        "selection": FROZEN_CONFIG.exp442_selection_sha256,
    }
    hard_payload, hard_digest = _read_regular_bytes_antigravity(paths.exp404_hard_prediction, expected["hard"])
    prediction_payload, prediction_digest = _read_regular_bytes_antigravity(paths.exp442_prediction, expected["prediction"])
    ranking_payload, ranking_digest = _read_regular_bytes_antigravity(paths.exp442_candidate_ranking, expected["ranking"])
    selection_payload, selection_digest = _read_regular_bytes_antigravity(paths.exp442_selection, expected["selection"])
    artifact_hashes = _validate_artifact_map_antigravity(paths, {"prediction": prediction_digest, "ranking": ranking_digest, "selection": selection_digest})
    manifest_payload, manifest_digest = _read_regular_bytes_antigravity(paths.exp442_manifest, artifact_hashes["manifest_antigravity.json"])
    _validate_publication_manifest_antigravity(manifest_payload, {"prediction": prediction_digest, "ranking": ranking_digest, "selection": selection_digest})
    # Parse the source audit and status bytes only for exact pinned integrity and
    # label-free governance; their numerical contents are not prediction inputs.
    source_audit_payload, source_audit_digest = _read_regular_bytes_antigravity(paths.exp442_source_audit, artifact_hashes["source_audit_antigravity.json"])
    status_payload, status_digest = _read_regular_bytes_antigravity(paths.exp442_status, artifact_hashes["status_antigravity.json"])
    source_audit = _parse_json_object_antigravity(source_audit_payload, "Exp442 source audit")
    status = _parse_json_object_antigravity(status_payload, "Exp442 status")
    if status.get("experiment_id") != "Exp442" or status.get("label_access_count") != 0 or status.get("contains_test_labels") is not False or source_audit.get("label_access_count", 0) != 0:
        raise ValueError("Exp442 source governance is invalid")
    prediction, ranking = _parse_prediction_and_ranking_antigravity(prediction_payload, ranking_payload)
    selected_from_selection, hard_coordinates = _parse_selection_antigravity(selection_payload, expected)
    _validate_candidate_replay_agreement_antigravity(prediction, ranking, selected_from_selection)
    selected_identity = _canonical_coordinate_identity_antigravity(tuple(item.coordinate for item in prediction))
    if selected_identity != FROZEN_CONFIG.exp442_selected_coordinate_identity_sha256:
        raise ValueError("Exp442 selected coordinate identity SHA256 mismatch")
    hard = _validate_hard_prediction_antigravity(hard_payload)
    hard_datasets = {dataset for dataset, _ in hard}
    if any(item.dataset in hard_datasets or item.coordinate in hard for item in prediction):
        raise ValueError("Exp442 selected candidate overlaps frozen Hard source")
    if hard_coordinates != hard:
        raise ValueError("Exp442 selection Hard coordinates disagree with Exp404")
    ordered_candidates = tuple(sorted(prediction, key=lambda item: item.original_rank))
    if tuple(item.original_rank for item in ordered_candidates) != tuple(range(1, 31)):
        raise ValueError("Exp442 verified candidate rank order is not contiguous 1-30")
    replay_attestation = _VerifiedReplayAttestationAntigravity(
        _VERIFIED_REPLAY_TOKEN_ANTIGRAVITY,
        paths.database,
        expected["database"],
        ordered_candidates,
    )
    session = ValuesOnlyDatabaseSessionAntigravity._from_verified_replay_antigravity(
        replay_attestation,
        _VERIFIED_REPLAY_TOKEN_ANTIGRAVITY,
    )
    hashes = {
        "database": expected["database"],
        "exp404_hard_prediction": hard_digest,
        "exp442_prediction": prediction_digest,
        "exp442_candidate_ranking": ranking_digest,
        "exp442_selection": selection_digest,
        "exp442_selected_coordinate_identity": selected_identity,
        "exp442_artifact_map": artifact_hashes["artifact_map"],
        "exp442_manifest": manifest_digest,
        "exp442_source_audit": source_audit_digest,
        "exp442_status": status_digest,
    }
    try:
        return FrozenSourceSnapshotAntigravity(
            candidates=ordered_candidates,
            selected_coordinate_identity_sha256=selected_identity,
            candidate_attestation_sha256=replay_attestation.candidate_attestation_sha256,
            source_hashes=hashes,
            database_path=paths.database,
            label_access_count=session.label_access_count,
            session=session,
        )
    except BaseException as error:
        try:
            session.close()
        except BaseException as close_error:
            error.add_note(f"source session cleanup failed: {close_error}")
        raise


def _prepare_candidate_input_antigravity(
    candidate: SourceCandidateAntigravity,
    train_rows: Sequence[Sequence[object]],
    test_rows: Sequence[Sequence[object]],
) -> CandidateInputAntigravity:
    decoded: list[tuple[int, np.ndarray]] = []
    for row in train_rows:
        if len(row) != 2 or type(row[0]) is not int or row[0] < 0:
            raise ValueError("TRAIN instance identity is invalid")
        decoded.append((row[0], _decode_values_blob_antigravity(row[1], "TRAIN")))
    indices = tuple(index for index, _ in decoded)
    if indices != tuple(sorted(set(indices))):
        raise ValueError("TRAIN instance identity is not ordered and unique")
    finite_group = tuple(
        (index, values)
        for index, values in decoded
        if values.size == candidate.raw_length and np.isfinite(values).all()
    )
    if len(finite_group) < FROZEN_CONFIG.minimum_train_rows:
        raise ValueError("candidate exact-length TRAIN group is too small")
    if len(test_rows) != 1 or len(test_rows[0]) != 2 or test_rows[0][0] != candidate.instance_index:
        raise ValueError("bound TEST values row is missing or duplicated")
    test_values = _decode_values_blob_antigravity(test_rows[0][1], "TEST")
    if test_values.size != candidate.raw_length:
        raise ValueError("bound TEST values length disagrees with candidate raw length")
    train_indices = tuple(index for index, _ in finite_group)
    train_matrix = np.stack(tuple(values for _, values in finite_group), axis=0)
    return CandidateInputAntigravity(candidate, train_indices, train_matrix, test_values)


def _load_candidate_values_from_session_antigravity(snapshot: FrozenSourceSnapshotAntigravity, session: ValuesOnlyDatabaseSessionAntigravity) -> tuple[CandidateInputAntigravity, ...]:
    if session.database_sha256 != snapshot.database_sha256:
        raise ValueError("database hash does not match source snapshot")
    train_cache: dict[str, tuple[tuple[object, ...], ...]] = {}
    result: list[CandidateInputAntigravity] = []
    for candidate in snapshot.candidates:
        if candidate.dataset not in train_cache:
            train_cache[candidate.dataset] = session.query_antigravity(
                NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, (candidate.dataset,)
            )
        train_rows = train_cache[candidate.dataset]
        test_rows = session.query_antigravity(BOUND_TEST_VALUE_SQL_ANTIGRAVITY, (candidate.dataset, candidate.instance_index))
        result.append(_prepare_candidate_input_antigravity(candidate, train_rows, test_rows))
    if session.label_access_count != 0:
        raise ValueError("TEST label access was observed")
    return tuple(result)


def _validate_snapshot_for_loading_antigravity(
    snapshot: FrozenSourceSnapshotAntigravity,
) -> tuple[frozenset[str], frozenset[tuple[str, int]], tuple[tuple[object, ...], ...], str]:
    if not isinstance(snapshot, FrozenSourceSnapshotAntigravity):
        raise ValueError("source snapshot is invalid")
    datasets, coordinates, signatures, attestation = _validated_candidate_bounds_antigravity(snapshot.candidates)
    identity = selected_coordinate_identity_sha256_antigravity(tuple(coordinates))
    if snapshot.selected_coordinate_identity_sha256 != identity:
        raise ValueError("source snapshot coordinate identity changed")
    if snapshot.candidate_attestation_sha256 != attestation:
        raise ValueError("source snapshot candidate attestation changed")
    expected_hashes = {
        "database": FROZEN_CONFIG.database_sha256,
        "exp404_hard_prediction": FROZEN_CONFIG.exp404_hard_prediction_sha256,
        "exp442_prediction": FROZEN_CONFIG.exp442_prediction_sha256,
        "exp442_candidate_ranking": FROZEN_CONFIG.exp442_candidate_ranking_sha256,
        "exp442_selection": FROZEN_CONFIG.exp442_selection_sha256,
        "exp442_selected_coordinate_identity": FROZEN_CONFIG.exp442_selected_coordinate_identity_sha256,
    }
    if any(snapshot.source_hashes.get(name) != digest for name, digest in expected_hashes.items()):
        raise ValueError("source snapshot frozen hashes are invalid")
    if Path(snapshot.database_path) != Path(FROZEN_CONFIG.database_path):
        raise ValueError("source snapshot database path is not frozen")
    if snapshot.label_access_count != 0:
        raise ValueError("source snapshot has label access")
    return datasets, coordinates, signatures, attestation


def load_candidate_values_antigravity(snapshot: FrozenSourceSnapshotAntigravity) -> tuple[CandidateInputAntigravity, ...]:
    """Materialize only normal TRAIN and bound TEST values for 30 candidates."""
    datasets, coordinates, signatures, attestation = _validate_snapshot_for_loading_antigravity(snapshot)
    supplied = snapshot.session
    if supplied is None:
        raise ValueError("parent-owned verified source session is required")
    if not isinstance(supplied, ValuesOnlyDatabaseSessionAntigravity):
        raise ValueError("source session is invalid")
    if supplied._allowed_train_datasets != datasets or supplied._allowed_test_coordinates != coordinates:
        raise ValueError("source session bounds do not match snapshot")
    if supplied._bound_candidate_signatures != signatures:
        raise ValueError("source session candidate signatures do not match snapshot")
    if supplied._candidate_attestation_sha256 != attestation:
        raise ValueError("source session candidate attestation does not match snapshot")
    return _load_candidate_values_from_session_antigravity(snapshot, supplied)


__all__ = [
    "DuplicateJsonKeyError", "FrozenSourcePathsAntigravity", "ValuesOnlyDatabaseSessionAntigravity",
    "NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY", "BOUND_TEST_VALUE_SQL_ANTIGRAVITY",
    "selected_coordinate_identity_sha256_antigravity", "validate_prediction_sql_antigravity",
    "verify_frozen_sources_antigravity", "load_candidate_values_antigravity",
]
