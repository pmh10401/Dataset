from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


FORBIDDEN_PREDICTION_FIELDS = {
    "y_test",
    "label",
    "labels",
    "true_label",
    "true_anomaly_count",
    "hard_tp",
    "hard_fp",
    "hard_fn",
    "hard_f1",
    "precision",
    "recall",
    "f1",
}


@dataclass(frozen=True)
class HardPartition:
    guarded_hard_indices: tuple[int, ...]
    train_raw_exact_normal_indices: tuple[int, ...]


def partition_hard_indices(
    baseline_hard_indices: set[int], train_raw_exact_indices: set[int]
) -> HardPartition:
    baseline = {int(index) for index in baseline_hard_indices}
    removed = baseline & {int(index) for index in train_raw_exact_indices}
    guarded = baseline - removed
    if guarded & removed or guarded | removed != baseline:
        raise ValueError("Hard partition invariant failed")
    return HardPartition(tuple(sorted(guarded)), tuple(sorted(removed)))


def validate_prediction_row(row: dict[str, object]) -> None:
    forbidden = FORBIDDEN_PREDICTION_FIELDS & set(row)
    if forbidden:
        raise ValueError(f"forbidden prediction fields: {sorted(forbidden)}")
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    baseline = set(map(int, row["baseline_hard_indices"]))  # type: ignore[arg-type]
    guarded = set(map(int, row["guarded_hard_indices"]))  # type: ignore[arg-type]
    removed = set(map(int, row["train_raw_exact_normal_indices"]))  # type: ignore[arg-type]
    if guarded & removed or guarded | removed != baseline:
        raise ValueError("prediction Hard partition mismatch")
    n_test = int(row["n_test"])
    if any(index < 0 or index >= n_test for index in baseline):
        raise ValueError("prediction index out of range")


def load_blob_bundle_for_prediction(
    dataset: str, db_path: Path
) -> tuple[dict[int, bytes], dict[int, bytes]]:
    resolved = Path(db_path).resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT i.split, i.instance_index, i.values_blob
            FROM instances AS i
            JOIN datasets AS d ON d.id = i.dataset_id
            WHERE d.name = ? AND i.split IN ('TRAIN', 'TEST')
            ORDER BY i.split, i.instance_index
            """,
            (str(dataset),),
        ).fetchall()
    finally:
        connection.close()
    train: dict[int, bytes] = {}
    test: dict[int, bytes] = {}
    for split, index, blob in rows:
        target = train if split == "TRAIN" else test
        integer_index = int(index)
        if integer_index in target:
            raise ValueError(f"duplicate blob index: dataset={dataset} split={split} index={integer_index}")
        target[integer_index] = bytes(blob)
    if not train or not test:
        raise ValueError(f"missing TRAIN/TEST blobs: dataset={dataset}")
    return train, test


def build_prediction_rows(
    baseline_rows: list[dict[str, object]],
    *,
    db_path: Path,
    expected_datasets: int = 1117,
    expected_hard: int = 2253,
) -> list[dict[str, object]]:
    if len(baseline_rows) != expected_datasets:
        raise ValueError(f"baseline coverage mismatch: {len(baseline_rows)}/{expected_datasets}")
    output: list[dict[str, object]] = []
    total_hard = 0
    seen: set[str] = set()
    for baseline in baseline_rows:
        dataset = str(baseline["dataset"])
        if dataset in seen:
            raise ValueError(f"duplicate baseline dataset: {dataset}")
        seen.add(dataset)
        n_test = int(baseline["n_test"])
        hard = set(map(int, baseline["final_hard_indices"]))  # type: ignore[arg-type]
        total_hard += len(hard)
        train_blobs, test_blobs = load_blob_bundle_for_prediction(dataset, db_path)
        if len(test_blobs) != n_test or set(test_blobs) != set(range(n_test)):
            raise ValueError(f"TEST blob coverage mismatch: dataset={dataset}")
        train_by_blob: dict[bytes, list[int]] = {}
        for train_index, blob in train_blobs.items():
            train_by_blob.setdefault(blob, []).append(train_index)
        exact_by_candidate = {
            str(index): sorted(train_by_blob[test_blobs[index]])
            for index in sorted(hard)
            if test_blobs[index] in train_by_blob
        }
        partition = partition_hard_indices(hard, set(map(int, exact_by_candidate)))
        row: dict[str, object] = {
            "experiment_id": "experiment_290_train_raw_exact_normal_guard",
            "dataset": dataset,
            "n_test": n_test,
            "baseline_hard_indices": sorted(hard),
            "guarded_hard_indices": list(partition.guarded_hard_indices),
            "train_raw_exact_normal_indices": list(partition.train_raw_exact_normal_indices),
            "exact_train_indices_by_candidate": exact_by_candidate,
            "match_definition": "sqlite_values_blob_byte_exact",
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "retrospective_counterfactual": True,
            "upstream_end_to_end_strict_train_only": False,
        }
        validate_prediction_row(row)
        output.append(row)
    if total_hard != expected_hard:
        raise ValueError(f"Exp253 Hard parity mismatch: {total_hard}/{expected_hard}")
    return output


__all__ = [
    "FORBIDDEN_PREDICTION_FIELDS",
    "HardPartition",
    "build_prediction_rows",
    "load_blob_bundle_for_prediction",
    "partition_hard_indices",
    "validate_prediction_row",
]
