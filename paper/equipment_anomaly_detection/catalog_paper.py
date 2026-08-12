"""Task 8's independently recomputed, label-safe SQLite population authority."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sqlite3
from typing import Iterable

from .artifacts_paper import canonical_json_bytes, read_canonical, safe_root, sha256_bytes
from .config_paper import FROZEN_CONFIG

CATALOG_SCHEMA = "paper-task8-catalog.v1"
ACCEPTANCE_SCHEMA = "paper-task8-acceptance.v1"
POLICY_IDS = ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")


def coordinate_digest(coordinates: Iterable[tuple[str, int]]) -> str:
    return sha256_bytes(canonical_json_bytes([[name, index] for name, index in coordinates]))


def _digest(path: Path) -> dict[str, object]:
    raw = Path(path)
    # Validate the caller spelling and every existing component before this
    # function ever resolves or opens the database.
    safe_root(raw)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError("database must be a regular nonsymlink file")
    stat = raw.stat()
    return {"unresolved_path": str(raw), "resolved_path": str(raw.resolve(strict=True)), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": sha256(raw.read_bytes()).hexdigest()}


def _rows(connection: sqlite3.Connection, dataset: str, split: str) -> list[tuple[int, bytes]]:
    # TEST selection deliberately has no label column or predicate.
    query = ("SELECT i.instance_index,i.values_blob FROM instances i JOIN datasets d ON d.id=i.dataset_id "
             "WHERE d.name=? AND i.split='TRAIN' AND i.label='0' ORDER BY i.instance_index" if split == "TRAIN" else
             "SELECT i.instance_index,i.values_blob FROM instances i JOIN datasets d ON d.id=i.dataset_id "
             "WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index")
    rows = connection.execute(query, (dataset,)).fetchall()
    if not rows or any(type(index) is not int or index < 0 or type(blob) is not bytes or not blob for index, blob in rows):
        raise ValueError("catalog selected row is invalid")
    if tuple(index for index, _ in rows) != tuple(range(len(rows))):
        raise ValueError("catalog selected indices must be contiguous")
    return [(index, bytes(blob)) for index, blob in rows]


def _entry(connection: sqlite3.Connection, name: str, dataset_id: int, series_length: int) -> dict[str, object]:
    if type(dataset_id) is not int or type(series_length) is not int or series_length <= 0:
        raise ValueError("catalog dataset metadata is invalid")
    def identity(split: str) -> dict[str, object]:
        rows = _rows(connection, name, split)
        bare = {"dataset": name, "dataset_id": dataset_id, "series_length": series_length, "split": split,
                "indices": [index for index, _ in rows], "values_blob_sha256": [sha256(blob).hexdigest() for _, blob in rows]}
        return {**bare, "row_digest": sha256_bytes(canonical_json_bytes(bare)), "count": len(rows)}
    return {"dataset": name, "dataset_id": dataset_id, "series_length": series_length, "train": identity("TRAIN"), "test": identity("TEST")}


def build_catalog(db_path: Path, *, mode: str, smoke_datasets: tuple[str, ...] = ()) -> dict[str, object]:
    if mode not in {"smoke", "full"}: raise ValueError("catalog mode is invalid")
    database = _digest(Path(db_path))
    connection = sqlite3.connect(f"file:{database['resolved_path']}?mode=ro", uri=True)
    try:
        names = connection.execute("SELECT id,name,series_length FROM datasets ORDER BY name").fetchall()
        if (not names or any(type(row[0]) is not int or type(row[1]) is not str or not row[1] for row in names)
                or tuple(row[1] for row in names) != tuple(sorted(row[1] for row in names))
                or len({row[1] for row in names}) != len(names)):
            raise ValueError("catalog dataset inventory is invalid")
        tokens = tuple(row[0] for row in connection.execute("SELECT DISTINCT split FROM instances").fetchall())
        if set(tokens) - {"TRAIN", "TEST"}: raise ValueError("catalog has unknown split token")
        frozen_names=tuple(name for _,name,_ in names if name not in FROZEN_CONFIG.excluded_datasets)
        wanted = tuple(smoke_datasets) if mode == "smoke" else frozen_names
        if (mode == "smoke" and (not wanted or wanted != tuple(sorted(set(wanted))))
                or any(name not in frozen_names for name in wanted)):
            raise ValueError("smoke dataset subset is invalid")
        rows_by_name = {name: (dataset_id, length) for dataset_id, name, length in names}
        entries = [_entry(connection, name, *rows_by_name[name]) for name in wanted]
    finally:
        connection.close()
    bare = {"schema": CATALOG_SCHEMA, "mode": mode, "database": database, "datasets": entries}
    result = {**bare, "catalog_sha256": sha256_bytes(canonical_json_bytes(bare))}
    if mode == "full" and (len(entries) != 1117 or sum(entry["test"]["count"] for entry in entries) != 195900):
        raise ValueError("full catalog is unavailable or not official exact inventory")
    return result


def verify_catalog(value: object, db_path: Path, mode: str, smoke_datasets: tuple[str, ...] = ()) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"schema", "mode", "database", "datasets", "catalog_sha256"}:
        raise ValueError("catalog schema is invalid")
    recomputed = build_catalog(db_path, mode=mode, smoke_datasets=smoke_datasets)
    if value != recomputed:
        raise ValueError("catalog authority differs from database")
    return recomputed


def load_acceptance(path: Path, catalog: dict[str, object]) -> dict[str, object]:
    value = read_canonical(Path(path))
    if not isinstance(value, dict) or set(value) != {"schema", "catalog_sha256", "oracle", "policies", "acceptance_sha256"}:
        raise ValueError("acceptance schema is invalid")
    bare = {key: value[key] for key in value if key != "acceptance_sha256"}
    if value["schema"] != ACCEPTANCE_SCHEMA or value["catalog_sha256"] != catalog["catalog_sha256"] or value["acceptance_sha256"] != sha256_bytes(canonical_json_bytes(bare)):
        raise ValueError("acceptance authority differs")
    oracle = value["oracle"]
    if not isinstance(oracle, dict) or set(oracle) != {"identifier", "version", "relative_path", "sha256"} or any(type(oracle[key]) is not str and key != "sha256" for key in ("identifier", "version", "relative_path")) or type(oracle["sha256"]) is not str or len(oracle["sha256"]) != 64:
        raise ValueError("acceptance oracle is invalid")
    relative = Path(oracle["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("acceptance oracle path is invalid")
    entries = {entry["dataset"]: entry for entry in catalog["datasets"]}
    policies = value["policies"]
    if not isinstance(policies, list) or len(policies) != 4:
        raise ValueError("acceptance policy inventory is invalid")
    for order, (policy, expected) in enumerate(zip(policies, POLICY_IDS, strict=True)):
        if not isinstance(policy, dict) or set(policy) != {"policy_id", "order", "coordinates", "coordinate_count", "coordinate_sha256"} or policy["policy_id"] != expected or policy["order"] != order:
            raise ValueError("acceptance policy schema is invalid")
        coordinates = policy["coordinates"]
        if not isinstance(coordinates, list) or any(not isinstance(item, list) or len(item) != 2 or type(item[0]) is not str or type(item[1]) is not int for item in coordinates):
            raise ValueError("acceptance coordinates are invalid")
        pairs = tuple((name, index) for name, index in coordinates)
        if pairs != tuple(sorted(set(pairs))) or any(name not in entries or index < 0 or index >= entries[name]["test"]["count"] for name, index in pairs):
            raise ValueError("acceptance coordinates are invalid")
        if policy["coordinate_count"] != len(pairs) or policy["coordinate_sha256"] != coordinate_digest(pairs):
            raise ValueError("acceptance coordinate digest differs")
        if not pairs and policy["coordinate_count"] != 0:
            raise ValueError("acceptance empty policy differs")
    # The path is runtime capability metadata, deliberately not part of the
    # serialized acceptance contract or its digest.
    return {**value, "_path": str(Path(path))}
