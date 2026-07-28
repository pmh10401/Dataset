from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp411_near_repeat_qualification.contracts import TrainRun
from experiments.exp414_missing_aware_near_repeat.config import FROZEN_CONFIG
from experiments.exp414_missing_aware_near_repeat.inputs import (
    iter_prepared_train_datasets,
    scan_missing_inventory,
)
from experiments.exp414_missing_aware_near_repeat.repair import repair_train_blob


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _write_db(
    path: Path,
    *,
    duplicate_indices: bool = False,
    invalid_train_label: bool = False,
    malformed_train_blob: bool = False,
) -> None:
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
        second_index = 0 if duplicate_indices else 1
        second_label = "1" if invalid_train_label else "0"
        second_blob = b"bad" if malformed_train_blob else _blob((np.nan, 3.0, 4.0))
        connection.executemany(
            """
            INSERT INTO instances(
                dataset_id, split, instance_index, label, values_blob, labels_blob
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (1, "TRAIN", 0, "0", _blob((0.0, 1.0, 2.0)), b"TRAIN_LABELS"),
                (1, "TRAIN", second_index, second_label, second_blob, b"TRAIN_LABELS"),
                (1, "TEST", 0, "1", b"TEST_SENTINEL", b"TEST_SENTINEL"),
            ),
        )
        connection.commit()


class InputTests(unittest.TestCase):
    def test_streaming_loader_uses_only_normal_train_values(self) -> None:
        captured_payloads: list[bytes] = []

        def capture_repair(dataset: str, instance_index: int, payload: bytes):
            captured_payloads.append(payload)
            return repair_train_blob(dataset, instance_index, payload)

        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            with patch(
                "experiments.exp414_missing_aware_near_repeat.inputs.repair_train_blob",
                side_effect=capture_repair,
            ):
                prepared = tuple(
                    iter_prepared_train_datasets(
                        path, ("Alpha_normal_0",), config=FROZEN_CONFIG
                    )
                )
            inventory = scan_missing_inventory(
                path, ("Alpha_normal_0",), config=FROZEN_CONFIG
            )

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].repair_audit.status, "ready")
        self.assertNotIn(b"TEST_SENTINEL", captured_payloads)
        self.assertEqual(inventory.affected_datasets, 1)
        self.assertEqual(inventory.affected_runs, 1)
        self.assertEqual(inventory.nonfinite_values, 1)

    def test_loader_rejects_duplicate_train_indices(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path, duplicate_indices=True)
            with self.assertRaisesRegex(ValueError, "indices are duplicated"):
                tuple(iter_prepared_train_datasets(path, ("Alpha_normal_0",)))

    def test_loader_rejects_duplicate_requested_datasets(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            with self.assertRaisesRegex(ValueError, "requested datasets are duplicated"):
                tuple(
                    iter_prepared_train_datasets(
                        path, ("Alpha_normal_0", "Alpha_normal_0")
                    )
                )

    def test_loader_rejects_nonzero_train_labels(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path, invalid_train_label=True)
            with self.assertRaisesRegex(ValueError, "TRAIN labels"):
                tuple(iter_prepared_train_datasets(path, ("Alpha_normal_0",)))

    def test_loader_rejects_malformed_train_blob_width(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path, malformed_train_blob=True)
            with self.assertRaisesRegex(ValueError, "blob width"):
                tuple(iter_prepared_train_datasets(path, ("Alpha_normal_0",)))

    def test_loader_rejects_wal_and_shm_sidecars(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            for suffix in ("-wal", "-shm"):
                Path(f"{path}{suffix}").write_bytes(b"unpinned")
                with self.assertRaisesRegex(ValueError, "snapshot artifact"):
                    tuple(iter_prepared_train_datasets(path, ("Alpha_normal_0",)))
                Path(f"{path}{suffix}").unlink()

    def test_loader_rejects_post_repair_nonfinite_values(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            bad_run = TrainRun(
                dataset="Alpha_normal_0",
                instance_index=0,
                values=(float("nan"), 1.0),
                values_sha256="0" * 64,
            )

            def nonfinite_repair(dataset: str, index: int, payload: bytes):
                run, audit = repair_train_blob(dataset, index, payload)
                return (bad_run, audit) if index == 0 else (run, audit)

            with patch(
                "experiments.exp414_missing_aware_near_repeat.inputs.repair_train_blob",
                side_effect=nonfinite_repair,
            ):
                with self.assertRaisesRegex(ValueError, "post-repair values"):
                    tuple(iter_prepared_train_datasets(path, ("Alpha_normal_0",)))

    def test_early_close_validates_sql_trace(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            stream = iter_prepared_train_datasets(path, ("Alpha_normal_0",))
            next(stream)
            with patch(
                "experiments.exp414_missing_aware_near_repeat.inputs.trace_sql_contract",
                side_effect=ValueError("trace cleanup marker"),
            ):
                with self.assertRaisesRegex(ValueError, "trace cleanup marker"):
                    stream.close()

    def test_early_close_rechecks_database_snapshot(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            _write_db(path)
            stream = iter_prepared_train_datasets(path, ("Alpha_normal_0",))
            next(stream)
            path.write_bytes(path.read_bytes() + b"changed-after-yield")
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                stream.close()


if __name__ == "__main__":
    unittest.main()
