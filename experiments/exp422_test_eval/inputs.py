"""Prediction-first input validation for Exp422-TEST-EVAL."""

from collections.abc import Sequence
import csv
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from .config import Exp422TestEvalConfig, FROZEN_CONFIG
from .contracts import Coordinate, FrozenLabels, FrozenPrediction, LabelRow


_LABEL_FIELDS = ("dataset", "candidate_index", "is_true_anomaly")
_FORBIDDEN = frozenset(
    {
        "ground_truth", "actual_anomaly_location", "actual_anomaly_locations",
        "is_true_anomaly", "label", "labels", "precision", "recall", "f1",
        "mean_combined_f1", "confusion_matrix", "evaluation_metrics",
    }
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _identity(coordinates: Sequence[Coordinate]) -> str:
    return sha256(canonical_json_bytes([[dataset, index] for dataset, index in coordinates])).hexdigest()


def _dataset(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"Exp422-TEST-EVAL {field} is invalid")
    return value


def _index(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Exp422-TEST-EVAL {field} must be a nonnegative exact integer")
    return value


def _object(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"Exp422-TEST-EVAL {field} must be an object")
    return value


def _json_no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Exp422-TEST-EVAL JSON contains a duplicate key")
        result[key] = value
    return result


def verify_database_reference(
    database_path: Path,
    *,
    config: Exp422TestEvalConfig = FROZEN_CONFIG,
) -> str:
    """Verify the frozen DB bytes without opening SQLite tables."""
    config.validate()
    observed = sha256_file(Path(database_path))
    if observed != config.database_sha256:
        raise ValueError("Exp422-TEST-EVAL database SHA256 mismatch")
    return observed


def sqlite_quick_check(
    database_path: Path,
    *,
    config: Exp422TestEvalConfig = FROZEN_CONFIG,
) -> str:
    """Run the read-only SQLite integrity check after DB bytes are frozen."""
    verify_database_reference(database_path, config=config)
    resolved = Path(database_path).expanduser().resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        row = connection.execute("PRAGMA quick_check").fetchone()
    finally:
        connection.close()
    if row != ("ok",):
        raise ValueError("SQLite quick_check did not return ok")
    return "ok"


def _prediction_row(row: object, line_number: int) -> Coordinate:
    body = _object(row, f"prediction row {line_number}")
    forbidden = _FORBIDDEN & set(body)
    if forbidden:
        raise ValueError(f"prediction row contains forbidden label/evaluation field: {sorted(forbidden)[0]}")
    if body.get("contains_test_labels") is True or body.get("contains_evaluation_metrics") is True:
        raise ValueError("prediction row contains label or evaluation material")
    dataset = _dataset(body.get("dataset"), "prediction dataset")
    if "instance_index" in body:
        index = _index(body.get("instance_index"), "prediction instance_index")
    elif "candidate_index" in body:
        index = _index(body.get("candidate_index"), "prediction candidate_index")
    else:
        raise ValueError("prediction row must contain instance_index")
    return dataset, index


def freeze_prediction_inputs(
    prediction_path: Path,
    manifest_path: Path,
    *,
    config: Exp422TestEvalConfig = FROZEN_CONFIG,
) -> FrozenPrediction:
    """Freeze prediction coordinates before the TEST label file is opened."""
    config.validate()
    prediction = Path(prediction_path)
    manifest_file = Path(manifest_path)
    prediction_sha256 = sha256_file(prediction)
    manifest_sha256 = sha256_file(manifest_file)
    coordinates: set[Coordinate] = set()
    try:
        with prediction.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=_json_no_duplicate_keys)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"invalid prediction JSONL row {line_number}") from error
                coordinate = _prediction_row(row, line_number)
                if coordinate in coordinates:
                    raise ValueError("prediction contains duplicate coordinate")
                coordinates.add(coordinate)
    except OSError as error:
        raise ValueError("prediction file cannot be read") from error

    try:
        with manifest_file.open(encoding="utf-8") as source:
            manifest = json.load(source, object_pairs_hook=_json_no_duplicate_keys)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("prediction manifest cannot be read") from error
    manifest = _object(manifest, "prediction manifest")
    if manifest.get("experiment_id") != config.experiment_id:
        raise ValueError("prediction manifest experiment_id mismatch")
    if manifest.get("source_experiment_id") != config.source_experiment_id:
        raise ValueError("prediction manifest source_experiment_id mismatch")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("prediction manifest contains_test_labels must be false")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction manifest contains_evaluation_metrics must be false")
    for field in ("test_informed_reselection", "operational_policy_changed"):
        if manifest.get(field) is not False:
            raise ValueError(f"prediction manifest {field} must be false")
    if manifest.get("candidate_semantics") != "predicted_positive":
        raise ValueError("prediction manifest candidate_semantics is not frozen")
    raw_scope = manifest.get("dataset_scope")
    if type(raw_scope) is not list or not raw_scope:
        raise ValueError("prediction manifest dataset_scope must be a nonempty list")
    scope = tuple(_dataset(item, "manifest dataset") for item in raw_scope)
    if scope != tuple(sorted(scope)) or len(scope) != len(set(scope)):
        raise ValueError("prediction manifest dataset_scope must be sorted and unique")
    if len(scope) != config.expected_datasets:
        raise ValueError("prediction manifest dataset count mismatch")
    expected_rows = _index(manifest.get("prediction_coordinate_count"), "manifest prediction_coordinate_count")
    expected_datasets = _index(manifest.get("prediction_dataset_count"), "manifest prediction_dataset_count")
    coordinate_tuple = tuple(sorted(coordinates))
    if expected_rows != len(coordinate_tuple):
        raise ValueError("prediction manifest coordinate count mismatch")
    if expected_datasets != len({dataset for dataset, _ in coordinate_tuple}):
        raise ValueError("prediction manifest prediction dataset count mismatch")
    if not set(dataset for dataset, _ in coordinate_tuple) <= set(scope):
        raise ValueError("prediction coordinate is outside manifest dataset_scope")
    if sha256_file(prediction) != prediction_sha256 or sha256_file(manifest_file) != manifest_sha256:
        raise ValueError("prediction input changed while being frozen")
    return FrozenPrediction(
        coordinates=coordinate_tuple,
        dataset_scope=scope,
        prediction_dataset_count=expected_datasets,
        identity_sha256=_identity(coordinate_tuple),
        prediction_sha256=prediction_sha256,
        manifest_sha256=manifest_sha256,
    )


def _label_index(raw: object) -> int:
    if type(raw) is not str or not raw or not raw.isdecimal():
        raise ValueError("TEST label candidate_index must be a nonnegative decimal string")
    return int(raw)


def _label_value(raw: object) -> int:
    if raw not in {"0", "1"}:
        raise ValueError("TEST label must be literal 0 or 1")
    return int(raw)


def load_frozen_test_labels(
    labels_path: Path,
    frozen_prediction: FrozenPrediction,
    *,
    config: Exp422TestEvalConfig = FROZEN_CONFIG,
) -> FrozenLabels:
    """Read TEST labels only after a validated FrozenPrediction exists."""
    if type(frozen_prediction) is not FrozenPrediction:
        raise ValueError("TEST labels require a FrozenPrediction")
    config.validate()
    labels = Path(labels_path)
    labels_sha256 = sha256_file(labels)
    if labels_sha256 != config.labels_sha256:
        raise ValueError("TEST label SHA256 mismatch")
    rows: list[LabelRow] = []
    seen: set[Coordinate] = set()
    try:
        with labels.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != _LABEL_FIELDS:
                raise ValueError("TEST label header mismatch")
            for csv_row in reader:
                if set(csv_row) != set(_LABEL_FIELDS) or any(value is None for value in csv_row.values()):
                    raise ValueError("TEST label row fields mismatch")
                dataset = _dataset(csv_row["dataset"], "TEST label dataset")
                index = _label_index(csv_row["candidate_index"])
                label = _label_value(csv_row["is_true_anomaly"])
                coordinate = (dataset, index)
                if coordinate in seen:
                    raise ValueError("TEST labels contain duplicate coordinate")
                seen.add(coordinate)
                rows.append(LabelRow(dataset, index, label))
    except OSError as error:
        raise ValueError("TEST label file cannot be read") from error
    if sha256_file(labels) != labels_sha256:
        raise ValueError("TEST labels changed while being read")
    sorted_rows = tuple(sorted(rows, key=lambda row: row.coordinate))
    scope = tuple(sorted({row.dataset for row in sorted_rows}))
    if len(sorted_rows) != config.expected_label_rows:
        raise ValueError("TEST label row count mismatch")
    if len(scope) != config.expected_datasets:
        raise ValueError("TEST label dataset count mismatch")
    if scope != frozen_prediction.dataset_scope:
        raise ValueError("TEST label scope differs from prediction scope")
    if not set(frozen_prediction.coordinates) <= seen:
        raise ValueError("prediction coordinate is outside TEST label universe")
    return FrozenLabels(rows=sorted_rows, dataset_scope=scope, labels_sha256=labels_sha256)


def verify_input_snapshots(
    database_path: Path,
    prediction_path: Path,
    manifest_path: Path,
    labels_path: Path,
    database_sha256: str,
    frozen_prediction: FrozenPrediction,
    frozen_labels: FrozenLabels,
    *,
    config: Exp422TestEvalConfig = FROZEN_CONFIG,
) -> None:
    config.validate()
    if verify_database_reference(database_path, config=config) != database_sha256:
        raise ValueError("database snapshot changed")
    if sha256_file(prediction_path) != frozen_prediction.prediction_sha256:
        raise ValueError("prediction snapshot changed")
    if sha256_file(manifest_path) != frozen_prediction.manifest_sha256:
        raise ValueError("prediction manifest snapshot changed")
    if sha256_file(labels_path) != frozen_labels.labels_sha256:
        raise ValueError("label snapshot changed")


__all__ = [
    "canonical_json_bytes", "freeze_prediction_inputs", "load_frozen_test_labels",
    "sha256_file", "sqlite_quick_check", "verify_database_reference", "verify_input_snapshots",
]
