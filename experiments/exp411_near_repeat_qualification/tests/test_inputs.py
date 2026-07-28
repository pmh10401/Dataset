from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.exp411_near_repeat_qualification.config import FROZEN_CONFIG
from experiments.exp411_near_repeat_qualification.inputs import (
    load_train_runs,
    trace_sql_contract,
)


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_db(path: Path, *, invalid_train_label: bool = False) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                instance_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                values_blob BLOB NOT NULL,
                labels_blob BLOB
            );
            """
        )
        connection.execute(
            "INSERT INTO datasets(id, name) VALUES (1, 'Alpha_normal_0')"
        )
        rows = (
            (1, "TRAIN", 0, "0", _blob((0.0, 1.0, 2.0))),
            (
                1,
                "TRAIN",
                1,
                "1" if invalid_train_label else "0",
                _blob((0.0, 1.1, 2.2)),
            ),
            (1, "TEST", 0, "1", _blob((9.0, 9.0, 9.0))),
        )
        connection.executemany(
            """
            INSERT INTO instances(
                dataset_id, split, instance_index, label, values_blob
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


class InputTests(unittest.TestCase):
    def test_loader_returns_only_normal_train_runs(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            config = replace(
                FROZEN_CONFIG,
                expected_datasets=1,
                expected_families=1,
                database_sha256=_digest(path),
            )
            rows = load_train_runs(path, ("Alpha_normal_0",), config=config)
        self.assertEqual(tuple(rows), ("Alpha_normal_0",))
        self.assertEqual(
            [row.instance_index for row in rows["Alpha_normal_0"]],
            [0, 1],
        )
        self.assertEqual(rows["Alpha_normal_0"][0].values, (0.0, 1.0, 2.0))

    def test_loader_rejects_wal_sidecar(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            config = replace(
                FROZEN_CONFIG,
                expected_datasets=1,
                expected_families=1,
                database_sha256=_digest(path),
            )
            Path(f"{path}-wal").write_bytes(b"unpinned")
            with self.assertRaisesRegex(ValueError, "snapshot artifact"):
                load_train_runs(path, ("Alpha_normal_0",), config=config)

    def test_loader_rejects_non_normal_train_label(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path, invalid_train_label=True)
            config = replace(
                FROZEN_CONFIG,
                expected_datasets=1,
                expected_families=1,
                database_sha256=_digest(path),
            )
            with self.assertRaisesRegex(ValueError, "TRAIN labels"):
                load_train_runs(path, ("Alpha_normal_0",), config=config)

    def test_sql_contract_rejects_test_or_labels_blob_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            trace_sql_contract(
                ("SELECT labels_blob FROM instances WHERE split = 'TEST'",)
            )


if __name__ == "__main__":
    unittest.main()
