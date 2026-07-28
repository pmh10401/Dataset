from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp410_family_robustness.config import FROZEN_CONFIG
from experiments.exp410_family_robustness.profiles import scan_dataset_profiles


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_profile_db(path: Path) -> None:
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
        connection.executemany(
            "INSERT INTO datasets(id, name) VALUES (?, ?)",
            ((1, "Alpha_normal_0"), (2, "Beta_normal_0")),
        )
        rows = (
            (1, "TRAIN", 0, "normal", _blob((1.0, 2.0))),
            (1, "TRAIN", 1, "normal", _blob((3.0, 4.0))),
            (1, "TEST", 0, "normal", _blob((1.0, 2.0))),
            (1, "TEST", 1, "normal", _blob((5.0, 6.0))),
            (1, "TEST", 2, "anomaly", _blob((3.0, 4.0))),
            (2, "TRAIN", 0, "normal", _blob((7.0, 8.0))),
            (2, "TEST", 0, "normal", _blob((7.0, 8.0))),
            (2, "TEST", 1, "anomaly", _blob((9.0, 10.0))),
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


class DatasetProfileTests(unittest.TestCase):
    def test_streaming_profile_counts_byte_exact_normal_overlap(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            write_profile_db(path)
            config = replace(
                FROZEN_CONFIG,
                expected_datasets=2,
                expected_families=2,
                database_sha256=_digest(path),
            )
            rows = scan_dataset_profiles(
                path,
                ("Alpha_normal_0", "Beta_normal_0"),
                config=config,
            )
            alpha = rows[0]
            self.assertEqual(alpha.train_normal_count, 2)
            self.assertEqual(alpha.test_anomaly_count, 1)
            self.assertEqual(alpha.test_normal_count, 2)
            self.assertEqual(alpha.exact_overlap_count, 1)
            self.assertEqual(alpha.exact_overlap_fraction, 0.5)

    def test_quick_check_or_snapshot_change_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "fixture.db"
            write_profile_db(path)
            with patch(
                "experiments.exp410_family_robustness.profiles.database_snapshot",
                side_effect=({"db": "a" * 64}, {"db": "b" * 64}),
            ):
                with self.assertRaisesRegex(ValueError, "snapshot changed"):
                    scan_dataset_profiles(
                        path,
                        ("Alpha_normal_0", "Beta_normal_0"),
                        config=replace(
                            FROZEN_CONFIG,
                            expected_datasets=2,
                            expected_families=2,
                            database_sha256="a" * 64,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
