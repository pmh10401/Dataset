from __future__ import annotations

from collections.abc import Iterator, Sequence
from hashlib import sha256
from pathlib import Path
import sqlite3
from urllib.parse import quote

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp411_near_repeat_qualification.inputs import (
    database_snapshot,
    trace_sql_contract,
)

from .config import FROZEN_CONFIG, Exp414FrozenConfig
from .contracts import MissingInventory, PreparedDataset
from .repair import build_dataset_repair_audit, repair_train_blob


def iter_prepared_train_datasets(
    db_path: Path,
    datasets: Sequence[str],
    *,
    config: Exp414FrozenConfig = FROZEN_CONFIG,
) -> Iterator[PreparedDataset]:
    """Stream repaired normal TRAIN runs without accessing TEST data or labels."""
    config.validate()
    before = database_snapshot(db_path)
    if tuple(before) != ("db",):
        raise ValueError("Exp414 SQLite snapshot artifact set is not pinned")

    resolved = Path(db_path).resolve()
    connection = sqlite3.connect(
        f"file:{quote(str(resolved), safe='/')}?mode=ro",
        uri=True,
    )
    trace: list[str] = []
    connection.set_trace_callback(trace.append)
    try:
        if connection.execute("PRAGMA quick_check;").fetchall() != [("ok",)]:
            raise ValueError("Exp414 SQLite PRAGMA quick_check failed")
        for dataset in sorted(str(dataset) for dataset in datasets):
            yield _load_one_prepared_dataset(connection, dataset)
        trace_sql_contract(trace)
    except sqlite3.Error as error:
        raise ValueError("Exp414 SQLite TRAIN query failed") from error
    finally:
        connection.close()

    after = database_snapshot(db_path)
    if dict(before) != dict(after):
        raise ValueError("Exp414 SQLite snapshot changed during TRAIN scan")


def scan_missing_inventory(
    db_path: Path,
    datasets: Sequence[str],
    *,
    config: Exp414FrozenConfig = FROZEN_CONFIG,
) -> MissingInventory:
    """Return the deterministic normal-TRAIN non-finite inventory."""
    prepared = tuple(
        iter_prepared_train_datasets(db_path, datasets, config=config)
    )
    affected = [
        row.dataset
        for row in prepared
        if any(
            audit.nan_count or audit.posinf_count or audit.neginf_count
            for audit in row.repair_audit.runs
        )
    ]
    audits = [audit for row in prepared for audit in row.repair_audit.runs]
    return MissingInventory(
        datasets=len(prepared),
        affected_datasets=len(affected),
        affected_runs=sum(
            1
            for audit in audits
            if audit.nan_count or audit.posinf_count or audit.neginf_count
        ),
        nonfinite_values=sum(
            audit.nan_count + audit.posinf_count + audit.neginf_count
            for audit in audits
        ),
        all_missing_runs=sum(
            1 for audit in audits if audit.repair_method == "all_missing_abstain"
        ),
        affected_datasets_sha256=sha256(
            canonical_json_bytes(sorted(affected))
        ).hexdigest(),
    )


def _load_one_prepared_dataset(
    connection: sqlite3.Connection,
    dataset: str,
) -> PreparedDataset:
    identifiers = connection.execute(
        "SELECT id FROM datasets WHERE name = ?",
        (dataset,),
    ).fetchall()
    if len(identifiers) != 1:
        raise ValueError(f"Exp414 SQLite dataset is missing or duplicated: {dataset}")
    dataset_id = int(identifiers[0][0])
    total, normal = connection.execute(
        """
        SELECT COUNT(*), SUM(CASE WHEN label = '0' THEN 1 ELSE 0 END)
        FROM instances
        WHERE dataset_id = ? AND split = 'TRAIN'
        """,
        (dataset_id,),
    ).fetchone()
    if int(total) == 0 or int(normal or 0) != int(total):
        raise ValueError(f"Exp414 SQLite TRAIN labels are invalid: {dataset}")

    rows = connection.execute(
        """
        SELECT instance_index, values_blob
        FROM instances
        WHERE dataset_id = ? AND split = 'TRAIN' AND label = '0'
        ORDER BY instance_index
        """,
        (dataset_id,),
    ).fetchall()
    if len(rows) != int(total):
        raise ValueError(f"Exp414 SQLite TRAIN row count mismatch: {dataset}")

    repaired_runs = []
    audits = []
    indices: set[int] = set()
    for instance_index, blob in rows:
        index = int(instance_index)
        if index in indices:
            raise ValueError(f"Exp414 SQLite TRAIN indices are duplicated: {dataset}")
        indices.add(index)
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise ValueError(f"Exp414 SQLite TRAIN blob is invalid: {dataset}")
        payload = bytes(blob)
        if len(payload) % 4:
            raise ValueError(f"Exp414 SQLite TRAIN blob width is invalid: {dataset}")
        run, audit = repair_train_blob(dataset, index, payload)
        audits.append(audit)
        if run is not None:
            if len(run.values) != audit.length or not np.isfinite(run.values).all():
                raise ValueError(f"Exp414 post-repair values are invalid: {dataset}")
            repaired_runs.append(run)

    repair_audit = build_dataset_repair_audit(dataset, repaired_runs, audits)
    return PreparedDataset(
        dataset=dataset,
        runs=tuple(repaired_runs) if repair_audit.status == "ready" else (),
        repair_audit=repair_audit,
    )


__all__ = ["iter_prepared_train_datasets", "scan_missing_inventory"]
