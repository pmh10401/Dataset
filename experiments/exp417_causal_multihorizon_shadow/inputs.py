"""Frozen lane and raw-values input boundary for Exp417."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Any

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
    _verify_hashes(paths, _lane_hashes(config), "SHA256 mismatch")
    rows_291 = _read_prediction_rows(paths[0])
    manifest_291 = _read_manifest(paths[1])
    rows_292 = _read_prediction_rows(paths[2])
    manifest_292 = _read_manifest(paths[3])
    lanes_291, counts_291 = _lanes_from_rows(rows_291)
    lanes_292, counts_292 = _lanes_from_rows(rows_292)
    _validate_manifest(manifest_291, config.exp291_prediction_sha256, len(rows_291))
    _validate_manifest(manifest_292, config.exp292_prediction_sha256, len(rows_292))
    if lanes_291 != lanes_292 or counts_291 != counts_292:
        raise ValueError("Exp291/Exp292 frozen lane mismatch")
    _verify_hashes(paths, _lane_hashes(config), "changed during parsing")
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
    _verify_database_hash(path, config.database_sha256)
    connection = _open_read_only(path)
    try:
        rows = connection.execute("SELECT name FROM datasets ORDER BY name").fetchall()
    except sqlite3.Error as error:
        raise ValueError("Exp417 database population query failed") from error
    finally:
        connection.close()
    datasets = tuple(row[0] for row in rows)
    if (
        len(datasets) != config.expected_datasets
        or any(not isinstance(name, str) or not name.strip() for name in datasets)
        or datasets != tuple(sorted(datasets))
        or len(datasets) != len(set(datasets))
    ):
        raise ValueError("database population does not match the frozen dataset count")
    lane_datasets = set().union(*(set(lane) for lane in _lane_maps(lanes)))
    if not lane_datasets.issubset(datasets):
        raise ValueError("frozen lane references a dataset outside the database population")
    _verify_database_hash(path, config.database_sha256)
    return datasets


def freeze_run_context(
    database: Path,
    lanes: FrozenLaneSnapshot,
    *,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> FrozenRunContext:
    """Bind database identity, population, lanes, code, and runtime versions."""
    datasets = list_dataset_names(database, lanes, config=config)
    test_identity_manifest = list_dataset_test_identity_manifest(database, datasets, config=config)
    return FrozenRunContext(
        database_sha256=config.database_sha256,
        datasets=datasets,
        lanes=lanes,
        test_identity_manifest=test_identity_manifest,
        code_commit=_git_commit(),
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
    _verify_database_hash(path, config.database_sha256)
    manifest: dict[str, tuple[tuple[int, int], ...]] = {}
    connection = _open_read_only(path)
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
    finally:
        connection.close()
    _verify_database_hash(path, config.database_sha256)
    return manifest


def load_raw_train_values(
    database: Path,
    dataset: str,
    context: FrozenRunContext,
) -> RawTrainDataset:
    """Load only normal TRAIN raw float32 values without transformations."""
    _validate_context_dataset(dataset, context)
    series = _load_values(database, dataset, context, _TRAIN_VALUES_SQL, "TRAIN")
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
    series = _load_values(database, dataset, context, _TEST_VALUES_SQL, "TEST")
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
    paths = (
        Path(exp291_prediction),
        Path(exp291_manifest),
        Path(exp292_prediction),
        Path(exp292_manifest),
    )
    _verify_hashes(paths, _lane_hashes(config), "SHA256 mismatch")
    _verify_database_hash(Path(database), config.database_sha256)


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
    try:
        digest = sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as error:
        raise ValueError(f"Exp417 cannot hash input: {path}") from error


def _read_prediction_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    seen_datasets: set[str] = set()
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
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
    except OSError as error:
        raise ValueError(f"cannot read prediction JSONL: {path}") from error
    if not rows:
        raise ValueError("prediction JSONL must not be empty")
    return tuple(rows)


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        manifest = _json_object(Path(path).read_text(encoding="utf-8"), "manifest")
    except OSError as error:
        raise ValueError(f"cannot read manifest: {path}") from error
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


def _open_read_only(database: Path) -> sqlite3.Connection:
    try:
        resolved = Path(database).resolve(strict=True)
        return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    except (OSError, sqlite3.Error) as error:
        raise ValueError(f"Exp417 cannot open read-only database: {database}") from error


def _validate_context_dataset(dataset: str, context: FrozenRunContext) -> None:
    if not isinstance(context, FrozenRunContext):
        raise ValueError("context must be a FrozenRunContext")
    if not isinstance(dataset, str) or not dataset.strip() or dataset not in context.datasets:
        raise ValueError("dataset is not in the frozen run context")


def _load_values(
    database: Path,
    dataset: str,
    context: FrozenRunContext,
    query: str,
    split: str,
) -> tuple[RawSeries, ...]:
    path = Path(database)
    _verify_database_hash(path, context.database_sha256)
    connection = _open_read_only(path)
    try:
        rows = connection.execute(query, (dataset,)).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"Exp417 {split} values query failed") from error
    finally:
        connection.close()
    _verify_database_hash(path, context.database_sha256)
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


def _git_commit() -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
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
    "verify_input_snapshots",
]
