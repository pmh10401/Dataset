from collections import Counter
from argparse import ArgumentParser
from contextlib import closing
from dataclasses import dataclass
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
import math
import os
import sqlite3
import struct


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


@dataclass(frozen=True, slots=True)
class ImportedDataset:
    name: str
    normal_source_label: str
    train_count: int
    test_count: int


@dataclass(frozen=True, slots=True)
class ImportSummary:
    source_database: str
    datasets: tuple[ImportedDataset, ...]


def choose_normal_label(counts: dict[str, int]) -> str:
    if not counts:
        raise ValueError("TRAIN labels are required")
    largest = max(counts.values())
    winners = [label for label, count in counts.items() if count == largest]
    if len(winners) != 1:
        raise ValueError("TRAIN labels must have one unique majority")
    return winners[0]


def derived_label(source_label: str, normal_source_label: str) -> str:
    return "0" if source_label == normal_source_label else "1"


def _validate_blob(blob: object) -> None:
    if type(blob) is not bytes or len(blob) % 4:
        raise ValueError("values_blob must contain little-endian float32 values")
    if not all(math.isfinite(value) for value, in struct.iter_unpack("<f", blob)):
        raise ValueError("values_blob values must be finite")


def build_derived_database(source_path: Path, output_path: Path) -> ImportSummary:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("source and output must not be the same path")
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError("output database must be new or empty")

    imported = []
    with closing(sqlite3.connect(f"{source_path.as_uri()}?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(output_path)) as output:
            try:
                output.executescript(SCHEMA)
                datasets = source.execute(
                    "SELECT id, name, univariate, equal_length, series_length, has_missing "
                    "FROM datasets ORDER BY name"
                )
                for new_id, (source_id, name, univariate, equal_length, series_length, has_missing) in enumerate(datasets, 1):
                    rows = source.execute(
                        "SELECT split, instance_index, label, values_blob, labels_blob FROM instances "
                        "WHERE dataset_id=? ORDER BY CASE split WHEN 'TRAIN' THEN 0 WHEN 'TEST' THEN 1 ELSE 2 END, "
                        "instance_index, id",
                        (source_id,),
                    ).fetchall()
                    if any(split not in ("TRAIN", "TEST") for split, *_ in rows):
                        raise ValueError(f"dataset {name!r} has an invalid split")
                    normal_label = choose_normal_label(dict(Counter(label for split, _, label, _, _ in rows if split == "TRAIN")))
                    counts = Counter((split, derived_label(label, normal_label)) for split, _, label, _, _ in rows)
                    train_count = counts["TRAIN", "0"] + counts["TRAIN", "1"]
                    test_count = counts["TEST", "0"] + counts["TEST", "1"]
                    output.execute(
                        "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            new_id, name, univariate, equal_length, series_length, has_missing,
                            counts["TRAIN", "0"], counts["TRAIN", "1"], train_count,
                            counts["TEST", "0"], counts["TEST", "1"], test_count,
                            counts["TRAIN", "0"] + counts["TEST", "0"],
                            counts["TRAIN", "1"] + counts["TEST", "1"],
                            train_count + test_count,
                        ),
                    )
                    for split, instance_index, label, blob, labels_blob in rows:
                        _validate_blob(blob)
                        output.execute(
                            "INSERT INTO instances(dataset_id, split, instance_index, label, values_blob, labels_blob) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (new_id, split, instance_index, derived_label(label, normal_label), blob, labels_blob),
                        )
                    imported.append(ImportedDataset(name, normal_label, train_count, test_count))
                if not imported:
                    raise ValueError("at least one dataset is required")
                output.commit()
            except Exception:
                output.rollback()
                raise
    return ImportSummary(str(source_path), tuple(imported))


def validate_derived_database(path: Path, summary: ImportSummary) -> None:
    connection = sqlite3.connect(Path(path))
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise ValueError("derived database integrity check failed")
        invalid_labels = connection.execute(
            "SELECT COUNT(*) FROM instances WHERE label NOT IN ('0', '1')"
        ).fetchone()[0]
        if invalid_labels:
            raise ValueError("derived labels must be only 0 or 1")
        actual = {}
        for row in connection.execute(
            "SELECT d.name, d.train_total_count, d.test_total_count, "
            "SUM(CASE WHEN i.split='TRAIN' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN i.split='TEST' THEN 1 ELSE 0 END) "
            "FROM datasets d LEFT JOIN instances i ON i.dataset_id=d.id GROUP BY d.id ORDER BY d.name"
        ):
            actual[row[0]] = tuple(int(value or 0) for value in row[1:])
        expected = {
            dataset.name: (dataset.train_count, dataset.test_count) * 2
            for dataset in summary.datasets
        }
        if actual != expected:
            raise ValueError("derived database dataset counts do not match the import summary")
    finally:
        connection.close()


def render_import_report(summary: ImportSummary) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(dataset.name)}</td>"
        f"<td>{escape(dataset.normal_source_label)}</td>"
        f"<td>{dataset.train_count}</td><td>{dataset.test_count}</td><td>등록 성공</td>"
        "</tr>"
        for dataset in summary.datasets
    )
    return (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<title>로컬 데이터베이스 등록 결과</title></head><body>"
        "<h1>로컬 데이터베이스 등록 결과</h1>"
        f"<p>원본 파일: {escape(summary.source_database)}</p>"
        "<table><thead><tr><th>데이터셋</th><th>원본 정상 라벨</th>"
        "<th>TRAIN 수</th><th>TEST 수</th><th>상태</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )


def install_database(source_path: Path, data_directory: Path, report_path: Path) -> ImportSummary:
    source_path = Path(source_path).resolve()
    data_directory = Path(data_directory).resolve()
    target = (data_directory / "univariate_ts.db").resolve()
    report_path = Path(report_path).resolve()
    if source_path == target:
        raise ValueError("source and target must not be the same path")
    if report_path in (source_path, target):
        raise ValueError("source, target, and report must use distinct paths")
    data_directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=data_directory, suffix=".sqlite3", delete=False) as handle:
        staged = Path(handle.name)
    try:
        summary = build_derived_database(source_path, staged)
        validate_derived_database(staged, summary)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_import_report(summary), encoding="utf-8")
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return summary


def main() -> None:
    parser = ArgumentParser(description="로컬 SQLite 데이터베이스를 서비스에 등록합니다.")
    parser.add_argument("source_database", type=Path)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("report_path", type=Path)
    arguments = parser.parse_args()
    summary = install_database(
        arguments.source_database, arguments.data_directory, arguments.report_path
    )
    print(f"등록 성공: {len(summary.datasets)}개 데이터셋")


if __name__ == "__main__":
    main()
