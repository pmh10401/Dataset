from dataclasses import dataclass
from pathlib import Path
import sqlite3
from numba import njit
import numpy as np

from .model_resources import ModelResources

VALUE_SQL = (
    "SELECT i.split, i.instance_index, i.values_blob FROM instances i "
    "JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split IN ('TRAIN','TEST') "
    "ORDER BY CASE i.split WHEN 'TRAIN' THEN 0 ELSE 1 END, i.instance_index"
)
PREPARATION_VALUE_SQL = (
    "SELECT i.split, i.instance_index, i.values_blob FROM instances i "
    "JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND "
    "((i.split='TRAIN' AND i.label='0') OR i.split='TEST') "
    "ORDER BY CASE i.split WHEN 'TRAIN' THEN 0 ELSE 1 END, i.instance_index"
)


@dataclass(frozen=True)
class PreparedDatasetValues:
    dataset_name: str
    train_values: np.ndarray
    test_values: np.ndarray
    test_indices: tuple[int, ...]
    preparation_mode: str
    raw_train_rows: tuple[np.ndarray, ...] | None = None

    @property
    def target_length(self) -> int:
        return int(self.train_values.shape[1])


@njit(cache=True)
def resample_series(values: np.ndarray, target_length: int) -> np.ndarray:
    if values.ndim != 1 or values.size < 1 or target_length < 1:
        raise ValueError("a non-empty row and positive target_length are required")
    if values.size == target_length:
        return values.copy()
    output = np.empty(target_length, dtype=np.float64)
    if target_length == 1:
        output[0] = values[0]
        return output
    scale = (values.size - 1) / (target_length - 1)
    for index in range(target_length):
        position = index * scale
        left = int(position)
        right = min(left + 1, values.size - 1)
        weight = position - left
        output[index] = values[left] * (1.0 - weight) + values[right] * weight
    return output


def _decode_finite_row(blob: object) -> np.ndarray | None:
    if type(blob) is not bytes or not blob or len(blob) % 4:
        raise ValueError("values_blob must contain little-endian float32 values")
    values = np.frombuffer(blob, dtype="<f4").astype(np.float64)
    return values if np.isfinite(values).all() else None


def _standardize_resampled_rows(rows: list[np.ndarray], target_length: int) -> np.ndarray:
    prepared = np.empty((len(rows), target_length), dtype=np.float64)
    for index, values in enumerate(rows):
        prepared[index] = resample_series(values, target_length)
    scales = prepared.std(axis=1, keepdims=True)
    if np.any(scales == 0.0) or not np.isfinite(scales).all():
        raise ValueError("finite non-constant rows are required")
    standardized = (prepared - prepared.mean(axis=1, keepdims=True)) / scales
    if not np.isfinite(standardized).all():
        raise ValueError("standardized rows must be finite")
    return standardized


def _protect_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.frombuffer(np.asarray(row, dtype=np.float64).tobytes(), dtype=np.float64)
        for row in rows
    )


def prepare_dataset_values(
    connection: sqlite3.Connection,
    dataset_name: str,
) -> PreparedDatasetValues:
    if type(connection) is not sqlite3.Connection:
        raise ValueError("connection must be an exact sqlite3.Connection")
    if type(dataset_name) is not str or not dataset_name:
        raise ValueError("dataset_name must be an exact non-empty string")

    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    test_indices: list[int] = []
    for split, instance_index, blob in connection.execute(
        PREPARATION_VALUE_SQL, (dataset_name,)
    ):
        if split not in ("TRAIN", "TEST") or type(instance_index) is not int:
            raise ValueError("source row identity is invalid")
        values = _decode_finite_row(blob)
        if values is None:
            continue
        if split == "TRAIN":
            train.append(values)
        else:
            test.append(values)
            test_indices.append(instance_index)

    if len(train) < 5:
        raise ValueError("at least five finite normal TRAIN rows are required")
    if not test:
        raise ValueError("at least one finite TEST row is required")
    lengths = sorted(values.size for values in train)
    target_length = lengths[(len(lengths) - 1) // 2]
    return PreparedDatasetValues(
        dataset_name=dataset_name,
        train_values=_standardize_resampled_rows(train, target_length),
        test_values=_standardize_resampled_rows(test, target_length),
        test_indices=tuple(test_indices),
        preparation_mode="value_only_numba",
        raw_train_rows=_protect_rows(train),
    )


def open_source_database(path: Path) -> sqlite3.Connection:
    target = Path(path).resolve()
    connection = sqlite3.connect(
        f"{target.as_uri()}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def get_dataset_catalog(
    connection: sqlite3.Connection,
    resources: ModelResources,
) -> list[dict[str, object]]:
    entries = []
    rows = connection.execute(
        "SELECT name, train_total_count, test_total_count, series_length FROM datasets ORDER BY name"
    )
    for row in rows:
        dataset_name = row["name"]
        method_name = resources.route_by_dataset.get(dataset_name)
        if method_name is None:
            availability = "missing_verified_route"
        elif dataset_name in resources.recovery_datasets:
            availability = "recovery_numba"
        elif row["train_total_count"] < 5:
            availability = "insufficient_training_data"
        else:
            availability = "standard"
        entries.append(
            {
                "dataset_name": dataset_name,
                "method_name": method_name,
                "train_count": int(row["train_total_count"]),
                "test_count": int(row["test_total_count"]),
                "series_length": int(row["series_length"]),
                "availability": availability,
                "analysis_available": availability in {"standard", "recovery_numba"},
            }
        )
    return entries


def standardize_equal_length_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows or len({row.size for row in rows}) != 1:
        raise ValueError("standard preparation requires non-empty equal-length rows")
    values = np.asarray(rows, dtype=np.float64)
    scales = values.std(axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(scales == 0.0):
        raise ValueError("standard preparation requires finite non-constant rows")
    return (values - values.mean(axis=1, keepdims=True)) / scales


def load_dataset_values(
    connection: sqlite3.Connection,
    dataset_name: str,
    preparation_mode: str,
) -> PreparedDatasetValues:
    if preparation_mode == "recovery_numba":
        prepared = prepare_dataset_values(connection, dataset_name)
        return PreparedDatasetValues(
            dataset_name=dataset_name,
            train_values=prepared.train_values,
            test_values=prepared.test_values,
            test_indices=prepared.test_indices,
            preparation_mode=preparation_mode,
            raw_train_rows=prepared.raw_train_rows,
        )
    if preparation_mode != "standard":
        raise ValueError("dataset is not available for analysis")

    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    test_indices: list[int] = []
    expected_index = {"TRAIN": 0, "TEST": 0}

    rows = connection.execute(VALUE_SQL, (dataset_name,)).fetchall()
    if not rows:
        raise ValueError(f"dataset not found: {dataset_name}")

    for row in rows:
        split = row["split"]
        instance_index = row["instance_index"]
        blob = row["values_blob"]
        if split not in expected_index or instance_index != expected_index[split]:
            raise ValueError("source value rows are not sequential")
        expected_index[split] += 1
        if type(blob) is not bytes or not blob or len(blob) % 4 != 0:
            raise ValueError("source value blob is invalid")
        values = np.frombuffer(blob, dtype="<f4")
        (train if split == "TRAIN" else test).append(values)
        if split == "TEST":
            test_indices.append(instance_index)

    if not train or not test:
        raise ValueError("standard preparation requires non-empty train and test sets")
    if len({row.size for row in train + test}) != 1:
        raise ValueError("standard TRAIN and TEST lengths must match")

    return PreparedDatasetValues(
        dataset_name=dataset_name,
        train_values=standardize_equal_length_rows(train),
        test_values=standardize_equal_length_rows(test),
        test_indices=tuple(test_indices),
        preparation_mode=preparation_mode,
        raw_train_rows=_protect_rows(train),
    )


def load_test_labels(
    connection: sqlite3.Connection,
    dataset_name: str,
    test_indices: tuple[int, ...],
) -> dict[int, int]:
    rows = connection.execute(
        "SELECT i.instance_index, i.label FROM instances i "
        "JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' "
        "ORDER BY i.instance_index",
        (dataset_name,),
    ).fetchall()
    labels: dict[int, int] = {}
    for row in rows:
        index = row["instance_index"]
        label = row["label"]
        if type(index) is not int or label not in ("0", "1"):
            raise ValueError("derived TEST labels must be binary and indexed")
        labels[index] = int(label)
    if any(index not in labels for index in test_indices):
        raise ValueError("a retained TEST index has no derived label")
    return {index: labels[index] for index in test_indices}


def load_series_for_chart(
    connection: sqlite3.Connection,
    dataset_name: str,
    series_index: int,
    preparation_mode: str,
    target_length: int,
) -> list[float]:
    if preparation_mode not in {"standard", "recovery_numba"}:
        raise ValueError("dataset is not available for charting")
    row = connection.execute(
        "SELECT i.values_blob FROM instances i "
        "JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' AND i.instance_index=?",
        (dataset_name, series_index),
    ).fetchone()
    if row is None:
        raise ValueError("TEST series is unknown")
    blob = row["values_blob"]
    if type(blob) is not bytes or not blob or len(blob) % 4 != 0:
        raise ValueError("TEST series blob is invalid")
    values = np.frombuffer(blob, dtype="<f4").astype(np.float64)
    if preparation_mode == "recovery_numba":
        values = resample_series(values, target_length)
    elif len(values) != target_length:
        raise ValueError("TEST series length differs from the verified target")
    scale = values.std()
    if not np.isfinite(values).all() or scale == 0.0:
        raise ValueError("TEST series cannot be charted")
    return ((values - values.mean()) / scale).astype(float).tolist()
