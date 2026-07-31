"""Frozen lane and raw-values input boundary for Exp417."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
from typing import Any
from urllib.parse import quote

import numpy as np

from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import (
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    RawSeries,
    RawTestDataset,
    RawTrainDataset,
)


_LANE_ROW_FIELDS = frozenset(
    {
        "dataset",
        "n_test",
        "automatic_hard_indices",
        "automatic_normal_exact_indices",
        "standard_review_indices",
        "priority_review_indices",
    }
)
_MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "prediction_sha256",
        "rows",
        "contains_test_labels",
        "contains_evaluation_metrics",
    }
)
_TRAIN_VALUES_SQL = """
SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index
"""
_TEST_VALUES_SQL = """
SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
"""
_TEST_IDENTITY_SQL = """
SELECT i.instance_index, length(i.values_blob)
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index
"""
_REPO_ROOT = Path(__file__).resolve().parents[2]
_INPUT_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


class _DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object repeats a key that would otherwise overwrite."""


def freeze_lane_inputs(
    exp291_prediction: Path,
    exp291_manifest: Path,
    exp292_prediction: Path,
    exp292_manifest: Path,
    *,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> FrozenLaneSnapshot:
    """Hash-gate both lane sources, then require their coordinates to agree."""
    _validate_config(config)
    paths = (
        Path(exp291_prediction),
        Path(exp291_manifest),
        Path(exp292_prediction),
        Path(exp292_manifest),
    )
    payloads = tuple(_read_pinned_bytes(path) for path in paths)
    expected_hashes = _lane_hashes(config)
    for path, (_, actual), wanted in zip(
        paths, payloads, expected_hashes, strict=True
    ):
        if actual != wanted:
            raise ValueError(f"Exp417 input SHA256 mismatch: {path.name}")
    rows_291 = _parse_prediction_rows(payloads[0][0], paths[0].name)
    manifest_291 = _parse_manifest(payloads[1][0], paths[1].name)
    rows_292 = _parse_prediction_rows(payloads[2][0], paths[2].name)
    manifest_292 = _parse_manifest(payloads[3][0], paths[3].name)
    lanes_291, counts_291 = _lanes_from_rows(rows_291)
    lanes_292, counts_292 = _lanes_from_rows(rows_292)
    _validate_manifest(manifest_291, config.exp291_prediction_sha256, len(rows_291))
    _validate_manifest(manifest_292, config.exp292_prediction_sha256, len(rows_292))
    if lanes_291 != lanes_292 or counts_291 != counts_292:
        raise ValueError("Exp291/Exp292 frozen lane mismatch")
    return FrozenLaneSnapshot(
        hard=lanes_291["hard"],
        standard=lanes_291["standard"],
        priority=lanes_291["priority"],
        automatic_normal=lanes_291["automatic_normal"],
        exp291_prediction_sha256=config.exp291_prediction_sha256,
        exp291_manifest_sha256=config.exp291_manifest_sha256,
        exp292_prediction_sha256=config.exp292_prediction_sha256,
        exp292_manifest_sha256=config.exp292_manifest_sha256,
        config=config,
        population=tuple(sorted(counts_291)),
    )


def list_dataset_names(
    database: Path,
    lanes: FrozenLaneSnapshot,
    *,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> tuple[str, ...]:
    """Return the exact sorted database population after its hash is pinned."""
    _validate_config(config)
    if not isinstance(lanes, FrozenLaneSnapshot) or lanes.config != config:
        raise ValueError("lanes must use the supplied frozen config")
    path = Path(database)
    with _verified_database_connection(path, config.database_sha256) as connection:
        try:
            rows = connection.execute("SELECT name FROM datasets ORDER BY name").fetchall()
        except sqlite3.Error as error:
            raise ValueError("Exp417 database population query failed") from error
    database_datasets = tuple(row[0] for row in rows)
    if (
        any(not isinstance(name, str) or not name.strip() for name in database_datasets)
        or database_datasets != tuple(sorted(database_datasets))
        or len(database_datasets) != len(set(database_datasets))
    ):
        raise ValueError("database population is invalid")
    if lanes.population:
        frozen_datasets = set(lanes.population)
    else:
        frozen_datasets = set(database_datasets)
        if len(frozen_datasets) != config.expected_datasets:
            raise ValueError("database population does not match the frozen dataset count")
    if not frozen_datasets.issubset(database_datasets):
        raise ValueError("frozen lane references a dataset outside the database population")
    datasets = tuple(
        name for name in database_datasets if name in frozen_datasets
    )
    if len(datasets) != config.expected_datasets:
        raise ValueError("database population does not cover the frozen lane universe")
    return datasets


def freeze_run_context(
    database: Path,
    lanes: FrozenLaneSnapshot,
    *,
    verified_code_commit: str,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> FrozenRunContext:
    """Bind database identity, population, lanes, code, and runtime versions."""
    if (
        not isinstance(verified_code_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", verified_code_commit) is None
    ):
        raise ValueError("verified_code_commit must be the reviewed clean Git HEAD")
    datasets = list_dataset_names(database, lanes, config=config)
    test_identity_manifest = list_dataset_test_identity_manifest(database, datasets, config=config)
    return FrozenRunContext(
        database_sha256=config.database_sha256,
        datasets=datasets,
        lanes=lanes,
        test_identity_manifest=test_identity_manifest,
        code_commit=verified_code_commit,
        runtime_versions=_runtime_versions(),
        config=config,
    )


def list_dataset_test_identity_manifest(
    database: Path,
    datasets: tuple[str, ...],
    *,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, tuple[tuple[int, int], ...]]:
    """Read exact TEST identities and raw float32 lengths from the pinned database."""
    _validate_config(config)
    if not isinstance(datasets, tuple) or not datasets:
        raise ValueError("datasets must be the frozen dataset tuple")
    path = Path(database)
    manifest: dict[str, tuple[tuple[int, int], ...]] = {}
    with _verified_database_connection(path, config.database_sha256) as connection:
        try:
            for dataset in datasets:
                rows = connection.execute(_TEST_IDENTITY_SQL, (dataset,)).fetchall()
                entries: list[tuple[int, int]] = []
                for index, byte_length in rows:
                    if type(index) is not int or type(byte_length) is not int or byte_length <= 0 or byte_length % 4:
                        raise ValueError("database TEST identity manifest has an invalid raw float32 length")
                    entries.append((index, byte_length // 4))
                indices = tuple(index for index, _ in entries)
                if not entries or indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
                    raise ValueError("database TEST identity manifest must have sorted unique identities")
                manifest[dataset] = tuple(entries)
        except sqlite3.Error as error:
            raise ValueError("Exp417 database TEST identity query failed") from error
    return manifest


def load_raw_train_values(
    database: Path,
    dataset: str,
    context: FrozenRunContext,
) -> RawTrainDataset:
    """Load only normal TRAIN raw float32 values without transformations."""
    _validate_context_dataset(dataset, context)
    with verified_read_only_connection(database, context.database_sha256) as connection:
        return load_raw_train_values_from_connection(
            connection,
            dataset,
            database_sha256=context.database_sha256,
            datasets=context.datasets,
        )


def load_raw_train_values_from_connection(
    connection: sqlite3.Connection,
    dataset: str,
    *,
    database_sha256: str,
    datasets: tuple[str, ...],
) -> RawTrainDataset:
    """Decode one normal-TRAIN dataset through an already verified connection."""
    _validate_frozen_population_dataset(dataset, database_sha256, datasets)
    series = _load_values_from_connection(connection, dataset, _TRAIN_VALUES_SQL, "TRAIN")
    return RawTrainDataset(dataset=dataset, series=series)


def load_raw_test_values(
    database: Path,
    dataset: str,
    frozen_train_contract: FrozenTrainContract,
    context: FrozenRunContext,
) -> RawTestDataset:
    """Load TEST values only after a validated TRAIN contract exists."""
    if not isinstance(frozen_train_contract, FrozenTrainContract):
        raise ValueError("frozen_train_contract must be a FrozenTrainContract")
    _validate_context_dataset(dataset, context)
    if frozen_train_contract.dataset != dataset:
        raise ValueError("frozen_train_contract dataset does not match requested TEST dataset")
    with verified_read_only_connection(database, context.database_sha256) as connection:
        return load_raw_test_values_from_connection(
            connection, dataset, frozen_train_contract, context
        )


def load_raw_test_values_from_connection(
    connection: sqlite3.Connection,
    dataset: str,
    frozen_train_contract: FrozenTrainContract,
    context: FrozenRunContext,
) -> RawTestDataset:
    """Decode TEST values through an already verified connection."""
    if not isinstance(frozen_train_contract, FrozenTrainContract):
        raise ValueError("frozen_train_contract must be a FrozenTrainContract")
    _validate_context_dataset(dataset, context)
    if frozen_train_contract.dataset != dataset:
        raise ValueError("frozen_train_contract dataset does not match requested TEST dataset")
    series = _load_values_from_connection(connection, dataset, _TEST_VALUES_SQL, "TEST")
    if tuple((item.instance_index, len(item.values)) for item in series) != context.test_identity_manifest[dataset]:
        raise ValueError("TEST values do not match the frozen identity manifest")
    return RawTestDataset(dataset=dataset, series=series)


def assert_label_free_sql_contract() -> None:
    """Fail if the fixed TEST query stops being identity-and-values only."""
    query = " ".join(_TEST_VALUES_SQL.lower().split())
    if " from " not in query:
        raise ValueError("Exp417 TEST SQL is malformed")
    selected = query.split(" from ", 1)[0].removeprefix("select ")
    columns = tuple(part.strip().removeprefix("i.") for part in selected.split(","))
    if columns != ("instance_index", "values_blob"):
        raise ValueError("Exp417 TEST SQL must select only identity and values")
    if re.search(r"\blabel\b", query):
        raise ValueError("Exp417 TEST SQL contains a forbidden label token")
    identity_query = " ".join(_TEST_IDENTITY_SQL.lower().split())
    if re.search(r"\blabel\b", identity_query) or "length(i.values_blob)" not in identity_query:
        raise ValueError("Exp417 TEST identity SQL must use only identity and values length")


def verify_input_snapshots(
    database: Path,
    exp291_prediction: Path,
    exp291_manifest: Path,
    exp292_prediction: Path,
    exp292_manifest: Path,
    *,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> None:
    """Verify exact source hashes before parsing or publishing a context."""
    _validate_config(config)
    verify_lane_input_snapshots(
        exp291_prediction,
        exp291_manifest,
        exp292_prediction,
        exp292_manifest,
        config=config,
    )
    _verify_database_hash(Path(database), config.database_sha256)


def verify_lane_input_snapshots(
    exp291_prediction: Path,
    exp291_manifest: Path,
    exp292_prediction: Path,
    exp292_manifest: Path,
    *,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> None:
    """Verify only the frozen lane artifacts, never the SQLite database."""
    _validate_config(config)
    paths = (
        Path(exp291_prediction),
        Path(exp291_manifest),
        Path(exp292_prediction),
        Path(exp292_manifest),
    )
    _verify_hashes(paths, _lane_hashes(config), "SHA256 mismatch")


def _validate_config(config: Exp417FrozenConfig) -> None:
    if not isinstance(config, Exp417FrozenConfig):
        raise ValueError("config must be an Exp417FrozenConfig")
    config.validate()


def _lane_hashes(config: Exp417FrozenConfig) -> tuple[str, str, str, str]:
    return (
        config.exp291_prediction_sha256,
        config.exp291_manifest_sha256,
        config.exp292_prediction_sha256,
        config.exp292_manifest_sha256,
    )


def _verify_hashes(paths: tuple[Path, ...], expected: tuple[str, ...], reason: str) -> None:
    for path, wanted in zip(paths, expected, strict=True):
        actual = _sha256_file(path)
        if actual != wanted:
            raise ValueError(f"Exp417 input SHA256 {reason}: {path.name}")


def _verify_database_hash(database: Path, expected: str) -> None:
    if _sha256_file(database) != expected:
        raise ValueError("Exp417 database SHA256 mismatch")


def _sha256_file(path: Path) -> str:
    return _read_pinned_bytes(path)[1]


def _read_pinned_bytes(path: Path) -> tuple[bytes, str]:
    """Read and hash one regular inode through one O_NOFOLLOW descriptor."""
    try:
        descriptor = os.open(Path(path), _INPUT_OPEN_FLAGS)
    except OSError as error:
        raise ValueError(f"Exp417 cannot hash input: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise ValueError(f"Exp417 input is not a linked regular file: {path}")
        digest = sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if _stable_inode_metadata(before) != _stable_inode_metadata(after):
            raise ValueError(f"Exp417 input inode changed while reading: {path}")
        return b"".join(chunks), digest.hexdigest()
    except OSError as error:
        raise ValueError(f"Exp417 cannot read pinned input: {path}") from error
    finally:
        os.close(descriptor)


def _read_prediction_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    payload, _ = _read_pinned_bytes(path)
    return _parse_prediction_rows(payload, Path(path).name)


def _parse_prediction_rows(
    payload: bytes, source_name: str
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    seen_datasets: set[str] = set()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"cannot decode prediction JSONL: {source_name}") from error
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        if not line.strip():
            raise ValueError(f"prediction JSONL has blank row {line_number}")
        row = _json_object(line, f"prediction row {line_number}")
        if not _LANE_ROW_FIELDS.issubset(row):
            raise ValueError(f"prediction JSONL has malformed row {line_number}")
        dataset = row["dataset"]
        if not isinstance(dataset, str) or not dataset.strip() or dataset in seen_datasets:
            raise ValueError("prediction JSONL has duplicate or invalid dataset")
        seen_datasets.add(dataset)
        rows.append(row)
    if not rows:
        raise ValueError("prediction JSONL must not be empty")
    return tuple(rows)


def _read_manifest(path: Path) -> Mapping[str, Any]:
    payload, _ = _read_pinned_bytes(path)
    return _parse_manifest(payload, Path(path).name)


def _parse_manifest(payload: bytes, source_name: str) -> Mapping[str, Any]:
    try:
        manifest = _json_object(payload.decode("utf-8"), "manifest")
    except UnicodeDecodeError as error:
        raise ValueError(f"cannot decode manifest: {source_name}") from error
    if not _MANIFEST_REQUIRED_FIELDS.issubset(manifest):
        raise ValueError("manifest has malformed fields")
    return manifest


def _json_object(payload: str, name: str) -> Mapping[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise _DuplicateJsonKeyError(f"{name} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{name} contains invalid constant: {value}")
            ),
        )
    except _DuplicateJsonKeyError:
        raise
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(result, Mapping):
        raise ValueError(f"{name} must be an object")
    return result


def _lanes_from_rows(
    rows: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, dict[str, tuple[int, ...]]], dict[str, int]]:
    lanes: dict[str, dict[str, tuple[int, ...]]] = {
        "hard": {},
        "standard": {},
        "priority": {},
        "automatic_normal": {},
    }
    counts: dict[str, int] = {}
    fields = (
        ("hard", "automatic_hard_indices"),
        ("automatic_normal", "automatic_normal_exact_indices"),
        ("standard", "standard_review_indices"),
        ("priority", "priority_review_indices"),
    )
    for row in rows:
        dataset = row["dataset"]
        count = row["n_test"]
        if type(count) is not int or count < 1:
            raise ValueError("prediction JSONL n_test must be a positive integer")
        assert isinstance(dataset, str)
        counts[dataset] = count
        seen: set[int] = set()
        for lane_name, field in fields:
            coordinates = row[field]
            if not isinstance(coordinates, list):
                raise ValueError(f"prediction JSONL {field} must be a list")
            validated: list[int] = []
            for coordinate in coordinates:
                if type(coordinate) is not int:
                    raise ValueError(f"prediction JSONL {field} coordinate must be an integer")
                if coordinate < 0 or coordinate >= count:
                    raise ValueError(f"prediction JSONL {field} coordinate is outside range")
                if coordinate in validated:
                    raise ValueError(f"prediction JSONL {field} has duplicate coordinate")
                validated.append(coordinate)
            if seen.intersection(validated):
                raise ValueError("prediction JSONL lanes overlap")
            seen.update(validated)
            ordered = tuple(sorted(validated))
            if ordered:
                lanes[lane_name][dataset] = ordered
    return lanes, counts


def _validate_manifest(manifest: Mapping[str, Any], prediction_sha256: str, row_count: int) -> None:
    declared_hash = manifest["prediction_sha256"]
    declared_count = manifest["rows"]
    if not isinstance(declared_hash, str) or declared_hash != prediction_sha256:
        raise ValueError("manifest prediction SHA256 disagrees with prediction")
    if type(declared_count) is not int or declared_count != row_count:
        raise ValueError("manifest row count disagrees with prediction")
    if manifest["contains_test_labels"] is not False or manifest["contains_evaluation_metrics"] is not False:
        raise ValueError("manifest declares forbidden content")


def _open_read_only(database: Path | int) -> sqlite3.Connection:
    try:
        if type(database) is int:
            descriptor_root = (
                Path("/dev/fd")
                if Path("/dev/fd").is_dir()
                else Path("/proc/self/fd")
            )
            descriptor_path = descriptor_root / str(database)
            uri = f"file:{quote(str(descriptor_path), safe='/')}?mode=ro&immutable=1"
        else:
            resolved = Path(database).resolve(strict=True)
            uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    except (OSError, sqlite3.Error) as error:
        raise ValueError(f"Exp417 cannot open read-only database: {database}") from error


@contextmanager
def verified_read_only_connection(
    database: Path, expected_sha256: str
):
    """Yield one immutable SQLite connection bound to a pinned source inode."""
    path = Path(database)
    try:
        descriptor = os.open(path, _INPUT_OPEN_FLAGS)
    except OSError as error:
        raise ValueError(f"Exp417 cannot open pinned database: {path}") from error
    connection: sqlite3.Connection | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise ValueError("Exp417 database must be a linked regular file")
        actual = _sha256_descriptor(descriptor)
        if actual != expected_sha256:
            raise ValueError("Exp417 database SHA256 mismatch")
        connection = _open_read_only(descriptor)
        yield connection
        after = os.fstat(descriptor)
        if _stable_inode_metadata(before) != _stable_inode_metadata(after):
            raise ValueError("Exp417 database inode changed during query")
    except OSError as error:
        raise ValueError("Exp417 pinned database verification failed") from error
    finally:
        if connection is not None:
            connection.close()
        os.close(descriptor)


@contextmanager
def _verified_database_connection(
    database: Path, expected_sha256: str
):
    """Backward-compatible private alias for one-use verified connections."""
    with verified_read_only_connection(database, expected_sha256) as connection:
        yield connection


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _stable_inode_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_context_dataset(dataset: str, context: FrozenRunContext) -> None:
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    if not isinstance(dataset, str) or not dataset.strip() or dataset not in context.datasets:
        raise ValueError("dataset is not in the frozen run context")


def _validate_frozen_population_dataset(
    dataset: str,
    database_sha256: str,
    datasets: tuple[str, ...],
) -> None:
    if not isinstance(database_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", database_sha256) is None:
        raise ValueError("database SHA256 is invalid")
    if (
        not isinstance(datasets, tuple)
        or not datasets
        or any(not isinstance(value, str) or not value.strip() for value in datasets)
        or datasets != tuple(sorted(datasets))
        or len(datasets) != len(set(datasets))
    ):
        raise ValueError("frozen dataset population is invalid")
    if not isinstance(dataset, str) or not dataset.strip() or dataset not in datasets:
        raise ValueError("dataset is not in the frozen population")


def _load_values(
    database: Path,
    dataset: str,
    context: FrozenRunContext,
    query: str,
    split: str,
) -> tuple[RawSeries, ...]:
    path = Path(database)
    with _verified_database_connection(path, context.database_sha256) as connection:
        return _load_values_from_connection(connection, dataset, query, split)


def _load_values_from_connection(
    connection: sqlite3.Connection,
    dataset: str,
    query: str,
    split: str,
) -> tuple[RawSeries, ...]:
    if not isinstance(connection, sqlite3.Connection):
        raise ValueError("connection must be a SQLite connection")
    try:
        rows = connection.execute(query, (dataset,)).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"Exp417 {split} values query failed") from error
    if not rows:
        raise ValueError(f"Exp417 {split} values are empty for dataset={dataset}")
    decoded = tuple(_decode_raw_series(dataset, split, row) for row in rows)
    return decoded


def _decode_raw_series(dataset: str, split: str, row: tuple[Any, Any]) -> RawSeries:
    if len(row) != 2 or type(row[0]) is not int:
        raise ValueError(f"Exp417 {split} instance identity is invalid: {dataset}")
    blob = row[1]
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError(f"Exp417 {split} values blob is invalid: {dataset}")
    payload = bytes(blob)
    if len(payload) % 4:
        raise ValueError(f"Exp417 {split} values byte length is not a multiple of four")
    values = np.frombuffer(payload, dtype="<f4")
    return RawSeries(
        instance_index=row[0],
        values=tuple(float(value) for value in values),
    )


def _lane_maps(lanes: FrozenLaneSnapshot) -> tuple[Mapping[str, tuple[int, ...]], ...]:
    return lanes.hard, lanes.standard, lanes.priority, lanes.automatic_normal


def _git_commit(repository: Path = _REPO_ROOT) -> str:
    try:
        return subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Exp417 cannot determine code commit") from error


def _runtime_versions() -> Mapping[str, str]:
    package_names = ("numpy", "numba", "torch")
    versions = {"python": sys.version.split()[0]}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "unavailable"
    return versions


__all__ = [
    "assert_label_free_sql_contract",
    "freeze_lane_inputs",
    "freeze_run_context",
    "list_dataset_test_identity_manifest",
    "list_dataset_names",
    "load_raw_test_values",
    "load_raw_train_values",
    "verified_read_only_connection",
    "verify_lane_input_snapshots",
    "verify_input_snapshots",
]
