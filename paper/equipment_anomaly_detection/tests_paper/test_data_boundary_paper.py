"""Real SQLite contracts for the paper values and lazy-label boundaries."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.config_paper import FROZEN_CONFIG


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _build_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                series_length INTEGER NOT NULL
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
            "INSERT INTO datasets(id, name, series_length) VALUES(?, ?, ?)",
            ((1, "toy", 2), (2, "removed", 2)),
        )
        connection.executemany(
            """
            INSERT INTO instances(dataset_id, split, instance_index, label, values_blob, labels_blob)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                (1, "TRAIN", 0, "0", _blob((0.0, 1.0)), b"forbidden"),
                (1, "TRAIN", 1, "0", _blob((1.0, 2.0)), b"forbidden"),
                (1, "TRAIN", 2, "1", _blob((9.0, 9.0)), b"forbidden"),
                (1, "TEST", 0, "0", _blob((2.0, 3.0)), b"forbidden"),
                (1, "TEST", 1, "1", _blob((8.0, 8.0)), b"forbidden"),
                (2, "TRAIN", 0, "0", _blob((4.0, 5.0)), b"forbidden"),
                (2, "TEST", 0, "0", _blob((6.0, 7.0)), b"forbidden"),
            ),
        )
        connection.commit()
    finally:
        connection.close()


class DataBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "toy.sqlite"
        _build_database(self.db_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture_config(self):
        return replace(
            FROZEN_CONFIG,
            database_path=str(self.db_path.resolve()),
            database_sha256=sha256(self.db_path.read_bytes()).hexdigest(),
            excluded_datasets=("removed", "also_removed"),
            dataset_count=1,
            test_row_count=2,
        )

    def _replace_label(self, replacement: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE instances SET label=? WHERE dataset_id="
                "(SELECT id FROM datasets WHERE name='toy') "
                "AND split='TEST' AND instance_index=1",
                (replacement,),
            )

    def test_test_values_are_label_free_and_preserve_little_endian_raw_blobs(self) -> None:
        """Selecting a TEST label or omitting raw-byte preservation must fail this boundary."""
        from paper.equipment_anomaly_detection import data_paper

        observed: list[str] = []
        with patch.object(data_paper, "FROZEN_CONFIG", self._fixture_config()):
            values = data_paper.load_dataset_values(
                self.db_path, "toy", sql_observer=observed.append
            )

        self.assertEqual(values.dataset, "toy")
        self.assertEqual(values.metadata_length, 2)
        self.assertEqual(values.train_indices, (0, 1))
        self.assertEqual(values.test_indices, (0, 1))
        self.assertEqual(values.test_blobs, (_blob((2.0, 3.0)), _blob((8.0, 8.0))))
        self.assertEqual(values.test_rows[0].dtype.str, "<f4")
        self.assertEqual(values.test_rows[0].tolist(), [2.0, 3.0])
        self.assertTrue(observed)
        self.assertTrue(all("label" not in sql.casefold() for sql in observed if "TEST" in sql))

    def test_population_is_sorted_and_excludes_the_frozen_names(self) -> None:
        """A changed sort or excluded-dataset filter must fail the population identity."""
        from paper.equipment_anomaly_detection import data_paper

        with patch.object(data_paper, "FROZEN_CONFIG", self._fixture_config()):
            population = data_paper.load_population(self.db_path)

        self.assertEqual(population, ("toy",))

    def test_values_reject_noncontiguous_test_indices(self) -> None:
        """Accepting a gap would detach score positions from the TEST row identity."""
        from paper.equipment_anomaly_detection import data_paper

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE instances SET instance_index=3 WHERE dataset_id=1 AND split='TEST' AND instance_index=1"
            )
        with patch.object(data_paper, "FROZEN_CONFIG", self._fixture_config()):
            with self.assertRaisesRegex(ValueError, "contiguous"):
                data_paper.load_dataset_values(self.db_path, "toy")

    def test_values_require_the_frozen_path_and_database_digest(self) -> None:
        """A substituted pathname or changed bytes must not share the database authority."""
        from paper.equipment_anomaly_detection import data_paper

        wrong_path = self.db_path.with_name("wrong.sqlite")
        _build_database(wrong_path)
        with patch.object(data_paper, "FROZEN_CONFIG", self._fixture_config()):
            with self.assertRaisesRegex(ValueError, "path"):
                data_paper.load_dataset_values(wrong_path, "toy")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("UPDATE instances SET label='1' WHERE id=4")
        stale = replace(self._fixture_config(), database_sha256="0" * 64)
        with patch.object(data_paper, "FROZEN_CONFIG", stale):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                data_paper.load_dataset_values(self.db_path, "toy")

    def test_data_module_does_not_eagerly_import_label_capability(self) -> None:
        """An eager label module import would make label capability available during scoring."""
        sys.modules.pop("paper.equipment_anomaly_detection.label_access_paper", None)
        from paper.equipment_anomaly_detection import data_paper

        importlib.reload(data_paper)

        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)

    def test_label_loader_accepts_only_exact_text_binary(self) -> None:
        """Whitespace, non-text, or non-binary TEST labels must never cross the lazy boundary."""
        from paper.equipment_anomaly_detection import label_access_paper

        with patch.object(label_access_paper, "FROZEN_CONFIG", self._fixture_config()):
            self.assertEqual(label_access_paper.load_test_labels(self.db_path, ("toy",))["toy"], (0, 1))
        self._replace_label(" 1")
        with patch.object(label_access_paper, "FROZEN_CONFIG", self._fixture_config()):
            with self.assertRaisesRegex(ValueError, "exact TEXT"):
                label_access_paper.load_test_labels(self.db_path, ("toy",))


if __name__ == "__main__":
    unittest.main()
