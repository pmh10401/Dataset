from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.paper_public_baselines.data import (
    list_datasets_values_only,
    load_dataset_values,
    validate_frozen_population,
    verify_frozen_population,
)


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _build_database(path: Path, datasets: tuple[str, ...] = ("toy",)) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                split TEXT NOT NULL,
                instance_index INTEGER NOT NULL,
                label TEXT NOT NULL,
                values_blob BLOB NOT NULL
            );
            """
        )
        for dataset_id, name in enumerate(datasets, start=1):
            connection.execute(
                "INSERT INTO datasets(id, name) VALUES(?, ?)",
                (dataset_id, name),
            )
            rows = (
                (dataset_id, "TRAIN", 0, "0", _blob((0.0, 1.0, 2.0))),
                (dataset_id, "TRAIN", 1, "0", _blob((1.0, 2.0, 3.0))),
                (dataset_id, "TEST", 0, "0", _blob((2.0, 3.0, 4.0))),
                (dataset_id, "TEST", 1, "1", _blob((9.0, 9.0, 9.0))),
            )
            connection.executemany(
                """
                INSERT INTO instances(
                    dataset_id, split, instance_index, label, values_blob
                ) VALUES(?, ?, ?, ?, ?)
                """,
                rows,
            )
        connection.commit()
    finally:
        connection.close()


class DataBoundaryTests(unittest.TestCase):
    def test_values_loader_never_selects_label_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "toy.sqlite"
            _build_database(database)
            observed_sql: list[str] = []

            values = load_dataset_values(
                database,
                "toy",
                sql_observer=observed_sql.append,
            )

        self.assertEqual(values.dataset, "toy")
        self.assertEqual(values.train_indices, (0, 1))
        self.assertEqual(values.test_indices, (0, 1))
        self.assertEqual(len(values.train_rows), 2)
        self.assertEqual(len(values.test_rows), 2)
        self.assertTrue(observed_sql)
        self.assertFalse(any("label" in query.lower() for query in observed_sql))

    def test_population_is_name_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "population.sqlite"
            _build_database(database, datasets=("zeta", "alpha"))

            datasets = list_datasets_values_only(database)

        self.assertEqual(datasets, ("alpha", "zeta"))

    def test_frozen_population_order_excludes_unregistered_database_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "population.sqlite"
            _build_database(database, datasets=("extra", "alpha", "zeta"))
            frozen = validate_frozen_population(
                ("zeta", "alpha"), expected_count=2
            )

            verified = verify_frozen_population(database, frozen)

        self.assertEqual(verified, ("zeta", "alpha"))

    def test_frozen_population_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_frozen_population(("alpha", "alpha"), expected_count=2)

    def test_frozen_population_rejects_missing_database_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "population.sqlite"
            _build_database(database, datasets=("alpha",))

            with self.assertRaisesRegex(ValueError, "missing"):
                verify_frozen_population(database, ("alpha", "zeta"))

    def test_values_loader_rejects_noncontiguous_test_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "bad.sqlite"
            _build_database(database)
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "UPDATE instances SET instance_index=3 "
                    "WHERE split='TEST' AND instance_index=1"
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_dataset_values(database, "toy")


if __name__ == "__main__":
    unittest.main()
