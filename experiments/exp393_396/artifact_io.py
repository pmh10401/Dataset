from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from experiments.exp393_396.contracts import (
    GrammarConfig,
    canonical_json,
    config_sha256,
)


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
PREDICTION_NAME = "exp393_predictions.jsonl"
PREDICTION_MANIFEST_NAME = "exp393_prediction_manifest.json"
EXP389_ROOT = DATA_ROOT / "exp389_390_dual_confidence_tiers"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_prediction_inputs(
    dataset_limit: int | None = None,
) -> list[dict[str, Any]]:
    uri = f"file:{DB_PATH}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        query = "SELECT id, name FROM datasets ORDER BY name"
        parameters: tuple[int, ...] = ()
        if dataset_limit is not None:
            query += " LIMIT ?"
            parameters = (int(dataset_limit),)
        datasets = connection.execute(query, parameters).fetchall()
        result: list[dict[str, Any]] = []
        for dataset_id, dataset_name in datasets:
            train_rows = connection.execute(
                """
                SELECT values_blob
                FROM instances
                WHERE dataset_id = ? AND split = 'TRAIN'
                ORDER BY instance_index
                """,
                (dataset_id,),
            ).fetchall()
            test_rows = connection.execute(
                """
                SELECT values_blob
                FROM instances
                WHERE dataset_id = ? AND split = 'TEST'
                ORDER BY instance_index
                """,
                (dataset_id,),
            ).fetchall()
            result.append(
                {
                    "dataset": str(dataset_name),
                    "train": [
                        np.frombuffer(row[0], dtype=np.float32).astype(float)
                        for row in train_rows
                    ],
                    "test": [
                        np.frombuffer(row[0], dtype=np.float32).astype(float)
                        for row in test_rows
                    ],
                }
            )
        return result
    finally:
        connection.close()


def write_prediction_artifact(
    output_root: Path,
    rows: list[dict[str, Any]],
    config: GrammarConfig,
) -> tuple[Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / PREDICTION_NAME
    manifest_path = output_root / PREDICTION_MANIFEST_NAME
    ordered = sorted(rows, key=lambda row: str(row["dataset"]))
    temporary = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(canonical_json(_json_value(row)) + "\n")
    temporary.replace(prediction_path)
    dataset_hash = hashlib.sha256(
        "\n".join(str(row["dataset"]) for row in ordered).encode("utf-8")
    ).hexdigest()
    manifest = {
        "experiment_id": "experiment_393_normal_process_grammar_prediction",
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "rows": len(ordered),
        "datasets": len(ordered),
        "dataset_list_sha256": dataset_hash,
        "config": asdict(config),
        "config_sha256": config_sha256(config),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_family_name": False,
        "uses_dataset_name_for_policy": False,
        "uses_test_length_for_budget": False,
        "test_metrics_select_policy": False,
        "eligible_for_deployment": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prediction_path, manifest_path


def load_verified_prediction(
    prediction_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("prediction manifest contains TEST labels")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction manifest contains evaluation metrics")
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("prediction SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prediction row count mismatch")
    names = [str(row.get("dataset", "")) for row in rows]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("prediction datasets are missing or duplicated")
    dataset_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if dataset_hash != manifest.get("dataset_list_sha256"):
        raise ValueError("prediction dataset list SHA256 mismatch")
    return rows, manifest


def _validated_indices(values: Any, n_test: int, field: str) -> set[int]:
    result: set[int] = set()
    for raw in values or []:
        index = int(raw)
        if index < 0 or index >= n_test:
            raise ValueError(f"{field} index is out of range")
        result.add(index)
    return result


def load_baseline_lanes() -> tuple[dict[str, dict[str, set[int]]], dict[str, Any]]:
    prediction_path = EXP389_ROOT / "exp389_frozen_predictions.jsonl"
    manifest_path = EXP389_ROOT / "exp389_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("baseline manifest contains TEST labels")
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("baseline prediction SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("baseline row count mismatch")
    result: dict[str, dict[str, set[int]]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        n_test = int(row["n_test"])
        if dataset in result:
            raise ValueError("baseline contains duplicate datasets")
        result[dataset] = {
            "hard": _validated_indices(
                row.get("automatic_hard_indices"),
                n_test,
                "baseline hard",
            ),
            "standard": _validated_indices(
                row.get("standard_review_indices"),
                n_test,
                "baseline standard",
            ),
            "priority": _validated_indices(
                row.get("priority_review_indices"),
                n_test,
                "baseline priority",
            ),
        }
    return result, {
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def load_test_labels(dataset_names: set[str]) -> dict[str, list[str]]:
    uri = f"file:{DB_PATH}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        result: dict[str, list[str]] = {}
        for dataset in sorted(dataset_names):
            rows = connection.execute(
                """
                SELECT i.label
                FROM instances AS i
                JOIN datasets AS d ON d.id = i.dataset_id
                WHERE d.name = ? AND i.split = 'TEST'
                ORDER BY i.instance_index
                """,
                (dataset,),
            ).fetchall()
            result[dataset] = [str(row[0]) for row in rows]
        return result
    finally:
        connection.close()
