from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import math
import sqlite3
import struct
import subprocess
import sys
import unittest

from services.time_series_anomaly_diagnosis.dataset_importer import (
    ImportSummary,
    ImportedDataset,
    build_derived_database,
    choose_normal_label,
    install_database,
    render_import_report,
    validate_derived_database,
)


SCHEMA = """
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    univariate BOOLEAN NOT NULL,
    equal_length BOOLEAN NOT NULL,
    series_length INTEGER,
    has_missing BOOLEAN NOT NULL,
    train_normal_count INTEGER NOT NULL,
    train_anomaly_count INTEGER NOT NULL,
    train_total_count INTEGER NOT NULL,
    test_normal_count INTEGER NOT NULL,
    test_anomaly_count INTEGER NOT NULL,
    test_total_count INTEGER NOT NULL,
    total_normal_count INTEGER NOT NULL,
    total_anomaly_count INTEGER NOT NULL,
    total_count INTEGER NOT NULL
);
CREATE TABLE instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    instance_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    values_blob BLOB NOT NULL,
    labels_blob BLOB,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
CREATE UNIQUE INDEX idx_instances_dataset_split_index
    ON instances(dataset_id, split, instance_index);
"""


def values_blob(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def make_source_database(
    path: Path,
    train_labels: tuple[str, ...] = ("normal", "normal", "fault"),
    test_labels: tuple[str, ...] = ("normal", "fault"),
    *,
    name: str = "Demo",
    blobs: tuple[bytes, ...] | None = None,
) -> None:
    rows = (("TRAIN", label) for label in train_labels)
    rows = tuple(rows) + tuple(("TEST", label) for label in test_labels)
    row_blobs = blobs or tuple(values_blob(index, index + 0.5) for index in range(len(rows)))
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO datasets VALUES (1, ?, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
            (name,),
        )
        indices = {"TRAIN": 0, "TEST": 0}
        for row_id, ((split, label), blob) in enumerate(zip(rows, row_blobs, strict=True), 1):
            index = indices[split]
            indices[split] += 1
            connection.execute(
                "INSERT INTO instances VALUES (?, 1, ?, ?, ?, ?, ?)",
                (row_id, split, index, label, blob, b"kept"),
            )
        connection.commit()
    finally:
        connection.close()


def add_source_dataset(
    connection: sqlite3.Connection,
    dataset_id: int,
    name: str,
    train_labels: tuple[str, ...],
    test_labels: tuple[str, ...],
) -> None:
    connection.execute(
        "INSERT INTO datasets VALUES (?, ?, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)",
        (dataset_id, name),
    )
    row_id = connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM instances").fetchone()[0]
    for split, labels in (("TRAIN", train_labels), ("TEST", test_labels)):
        for index, label in enumerate(labels):
            connection.execute(
                "INSERT INTO instances VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (row_id, dataset_id, split, index, label, values_blob(dataset_id, index)),
            )
            row_id += 1


def read_labels(path: Path, split: str) -> tuple[str, ...]:
    with closing(sqlite3.connect(path)) as connection:
        return tuple(
            row[0]
            for row in connection.execute(
                "SELECT label FROM instances WHERE split=? ORDER BY instance_index", (split,)
            )
        )


class DatasetImporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.sqlite3"
        self.output = self.root / "output.sqlite3"
        self.data_directory = self.root / "data"
        self.current = self.data_directory / "univariate_ts.db"
        self.report = self.root / "report.html"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_majority_train_label_becomes_zero(self) -> None:
        make_source_database(self.source)

        summary = build_derived_database(self.source, self.output)

        self.assertEqual(summary.datasets[0].normal_source_label, "normal")
        self.assertEqual(read_labels(self.output, "TRAIN"), ("0", "0", "1"))
        self.assertEqual(read_labels(self.output, "TEST"), ("0", "1"))

    def test_counts_are_recomputed_and_metadata_is_preserved(self) -> None:
        make_source_database(self.source)

        summary = build_derived_database(self.source, self.output)

        self.assertEqual(summary.datasets, (ImportedDataset("Demo", "normal", 3, 2),))
        with closing(sqlite3.connect(self.output)) as connection:
            row = connection.execute("SELECT * FROM datasets").fetchone()
            self.assertEqual(row, (1, "Demo", 1, 1, 2, 0, 2, 1, 3, 1, 1, 2, 3, 2, 5))
            self.assertEqual(connection.execute("SELECT labels_blob FROM instances LIMIT 1").fetchone()[0], b"kept")

    def test_empty_and_tied_train_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "TRAIN labels are required"):
            choose_normal_label({})
        with self.assertRaisesRegex(ValueError, "unique majority"):
            choose_normal_label({"A": 1, "B": 1})

    def test_malformed_or_nonfinite_blobs_are_rejected(self) -> None:
        cases = ((b"bad", "little-endian float32"), (values_blob(math.inf), "finite"))
        for blob, message in cases:
            with self.subTest(message=message):
                source = self.root / f"{message}.sqlite3"
                output = self.root / f"{message}-output.sqlite3"
                make_source_database(
                    source,
                    blobs=(blob, values_blob(1), values_blob(2), values_blob(3), values_blob(4)),
                )
                with self.assertRaisesRegex(ValueError, message):
                    build_derived_database(source, output)

    def test_validation_rejects_wrong_counts(self) -> None:
        make_source_database(self.source)
        summary = build_derived_database(self.source, self.output)
        with closing(sqlite3.connect(self.output)) as connection:
            connection.execute("UPDATE datasets SET test_total_count=99")
            connection.commit()

        with self.assertRaisesRegex(ValueError, "counts"):
            validate_derived_database(self.output, summary)

    def test_validation_rejects_wrong_labels(self) -> None:
        make_source_database(self.source)
        summary = build_derived_database(self.source, self.output)
        with closing(sqlite3.connect(self.output)) as connection:
            connection.execute("UPDATE instances SET label='2' WHERE id=1")
            connection.commit()

        with self.assertRaisesRegex(ValueError, "derived labels"):
            validate_derived_database(self.output, summary)

    def test_report_escapes_source_values(self) -> None:
        summary = ImportSummary(
            source_database="bad<&.db",
            datasets=(ImportedDataset("<Demo>", 'normal "A"', 3, 2),),
        )

        report = render_import_report(summary)

        self.assertIn("bad&lt;&amp;.db", report)
        self.assertIn("&lt;Demo&gt;", report)
        self.assertIn("normal &quot;A&quot;", report)
        self.assertNotIn("<Demo>", report)

    def test_tied_train_labels_do_not_replace_current_database(self) -> None:
        self.data_directory.mkdir()
        self.current.write_bytes(b"existing database")
        make_source_database(self.source, train_labels=("A", "B"))

        with self.assertRaisesRegex(ValueError, "unique majority"):
            install_database(self.source, self.data_directory, self.report)

        self.assertEqual(self.current.read_bytes(), b"existing database")
        self.assertFalse(self.report.exists())

    def test_validation_failure_does_not_replace_current_database(self) -> None:
        self.data_directory.mkdir()
        self.current.write_bytes(b"existing database")
        make_source_database(self.source)

        with patch(
            "services.time_series_anomaly_diagnosis.dataset_importer.validate_derived_database",
            side_effect=ValueError("validation failed"),
        ):
            with self.assertRaisesRegex(ValueError, "validation failed"):
                install_database(self.source, self.data_directory, self.report)

        self.assertEqual(self.current.read_bytes(), b"existing database")
        self.assertEqual(list(self.data_directory.iterdir()), [self.current])

    def test_report_write_failure_does_not_replace_current_database(self) -> None:
        self.data_directory.mkdir()
        self.current.write_bytes(b"existing database")
        self.report.mkdir()
        make_source_database(self.source)

        with self.assertRaises(IsADirectoryError):
            install_database(self.source, self.data_directory, self.report)

        self.assertEqual(self.current.read_bytes(), b"existing database")

    def test_report_path_cannot_alias_source_database(self) -> None:
        self.data_directory.mkdir()
        self.current.write_bytes(b"existing database")
        make_source_database(self.source)
        original_source = self.source.read_bytes()

        with self.assertRaisesRegex(ValueError, "distinct paths"):
            install_database(self.source, self.data_directory, self.source)

        self.assertEqual(self.source.read_bytes(), original_source)
        self.assertEqual(self.current.read_bytes(), b"existing database")

    def test_report_path_cannot_alias_installed_database(self) -> None:
        self.data_directory.mkdir()
        self.current.write_bytes(b"existing database")
        make_source_database(self.source)

        with self.assertRaisesRegex(ValueError, "distinct paths"):
            install_database(self.source, self.data_directory, self.current)

        self.assertEqual(self.current.read_bytes(), b"existing database")

    def test_empty_source_does_not_replace_current_database(self) -> None:
        self.data_directory.mkdir()
        self.current.write_bytes(b"existing database")
        with closing(sqlite3.connect(self.source)) as connection:
            connection.executescript(SCHEMA)

        with self.assertRaisesRegex(ValueError, "at least one dataset"):
            install_database(self.source, self.data_directory, self.report)

        self.assertEqual(self.current.read_bytes(), b"existing database")

    def test_dataset_without_train_rows_does_not_replace_current_database(self) -> None:
        self.data_directory.mkdir()
        self.current.write_bytes(b"existing database")
        make_source_database(self.source, train_labels=())

        with self.assertRaisesRegex(ValueError, "TRAIN labels are required"):
            install_database(self.source, self.data_directory, self.report)

        self.assertEqual(self.current.read_bytes(), b"existing database")

    def test_install_chooses_majority_independently_for_each_dataset(self) -> None:
        with closing(sqlite3.connect(self.source)) as connection:
            connection.executescript(SCHEMA)
            add_source_dataset(connection, 2, "Second", ("bad", "bad", "good"), ("good",))
            add_source_dataset(connection, 1, "First", ("normal", "fault", "normal"), ("normal",))
            connection.commit()

        summary = install_database(self.source, self.data_directory, self.report)

        self.assertEqual(
            tuple((dataset.name, dataset.normal_source_label) for dataset in summary.datasets),
            (("First", "normal"), ("Second", "bad")),
        )
        with closing(sqlite3.connect(self.current)) as connection:
            labels = connection.execute(
                "SELECT d.name, i.split, i.instance_index, i.label FROM instances i "
                "JOIN datasets d ON d.id=i.dataset_id ORDER BY d.name, i.split DESC, i.instance_index"
            ).fetchall()
        self.assertEqual(
            labels,
            [
                ("First", "TRAIN", 0, "0"), ("First", "TRAIN", 1, "1"),
                ("First", "TRAIN", 2, "0"), ("First", "TEST", 0, "0"),
                ("Second", "TRAIN", 0, "0"), ("Second", "TRAIN", 1, "0"),
                ("Second", "TRAIN", 2, "1"), ("Second", "TEST", 0, "1"),
            ],
        )

    def test_output_open_failure_closes_source_connection(self) -> None:
        make_source_database(self.source)
        opened: list[sqlite3.Connection] = []
        real_connect = sqlite3.connect

        def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            if opened:
                raise sqlite3.OperationalError("output open failed")
            connection = real_connect(*args, **kwargs)
            opened.append(connection)
            return connection

        with patch(
            "services.time_series_anomaly_diagnosis.dataset_importer.sqlite3.connect",
            side_effect=tracking_connect,
        ):
            with self.assertRaises(sqlite3.OperationalError):
                build_derived_database(self.source, self.output)

        self.assertEqual(len(opened), 1)
        try:
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed"):
                opened[0].execute("SELECT 1")
        finally:
            opened[0].close()

    def test_install_replaces_database_after_validation_and_writes_report(self) -> None:
        make_source_database(self.source)

        summary = install_database(self.source, self.data_directory, self.report)

        self.assertEqual(summary.datasets[0].train_count, 3)
        self.assertEqual(read_labels(self.current, "TEST"), ("0", "1"))
        self.assertIn("등록 성공", self.report.read_text(encoding="utf-8"))

    def test_successful_replace_has_no_later_staged_cleanup(self) -> None:
        make_source_database(self.source)

        with patch.object(Path, "unlink", side_effect=OSError("late cleanup failed")):
            summary = install_database(self.source, self.data_directory, self.report)

        self.assertEqual(summary.datasets[0].name, "Demo")
        self.assertEqual(read_labels(self.current, "TRAIN"), ("0", "0", "1"))

    def test_source_cannot_be_the_install_target(self) -> None:
        self.data_directory.mkdir()
        make_source_database(self.current)
        before = self.current.read_bytes()

        with self.assertRaisesRegex(ValueError, "same path"):
            install_database(self.current, self.data_directory, self.report)

        self.assertEqual(self.current.read_bytes(), before)

    def test_module_command_installs_local_database(self) -> None:
        make_source_database(self.source)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "services.time_series_anomaly_diagnosis.dataset_importer",
                str(self.source),
                str(self.data_directory),
                str(self.report),
            ],
            check=True,
        )

        self.assertEqual(read_labels(self.current, "TRAIN"), ("0", "0", "1"))
        self.assertTrue(self.report.is_file())


if __name__ == "__main__":
    unittest.main()
