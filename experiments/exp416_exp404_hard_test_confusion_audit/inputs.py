"""Prediction-first input validation for the Exp416 TEST-label audit."""

from collections.abc import Sequence
import csv
from hashlib import sha256
import json
from pathlib import Path

from .config import Exp416FrozenConfig, FROZEN_CONFIG
from .contracts import Coordinate, FrozenLabels, FrozenPrediction, LabelRow


_PREDICTION_FLAGS = ("contains_test_labels", "contains_evaluation_metrics")
_MANIFEST_COUNTS = (
    "datasets",
    "rows",
)
_LABEL_FIELDS = ("dataset", "candidate_index", "is_true_anomaly")


def sha256_file(path: Path) -> str:
    """Return the digest of *path* without interpreting its contents."""
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _coordinate_identity(coordinates: Sequence[Coordinate]) -> str:
    payload = canonical_json_bytes([[dataset, index] for dataset, index in coordinates])
    return sha256(payload).hexdigest()


def _exact_index(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Exp416 {field} must be a nonnegative exact integer")
    return value


def _dataset(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"Exp416 {field} must be a non-empty string")
    return value


def _require_false(mapping: dict[str, object], field: str) -> None:
    if mapping.get(field) is not False:
        raise ValueError(f"Exp416 {field} must be exactly false")


def _manifest_lanes(value: object) -> tuple[tuple[str, ...], set[Coordinate]]:
    if type(value) is not dict:
        raise ValueError("Exp416 manifest frozen_hard_lanes must be an object")
    lane_scope = tuple(sorted(_dataset(dataset, field="frozen_hard_lanes dataset") for dataset in value))
    coordinates: set[Coordinate] = set()
    for dataset, values in value.items():
        if type(values) is not list:
            raise ValueError("Exp416 frozen_hard_lanes values must be lists")
        for raw_index in values:
            coordinate = (dataset, _exact_index(raw_index, field="frozen_hard_lanes candidate_index"))
            if coordinate in coordinates:
                raise ValueError("Exp416 frozen_hard_lanes contains duplicate coordinate")
            coordinates.add(coordinate)
    return lane_scope, coordinates


def _json_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"Exp416 manifest contains duplicate key: {key}")
        value[key] = child
    return value


def _verified_prediction_hashes(
    prediction_path: Path, manifest_path: Path, config: Exp416FrozenConfig
) -> tuple[str, str]:
    prediction_sha256 = sha256_file(prediction_path)
    if prediction_sha256 != config.prediction_sha256:
        raise ValueError("Exp416 prediction SHA256 mismatch")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != config.manifest_sha256:
        raise ValueError("Exp416 manifest SHA256 mismatch")
    return prediction_sha256, manifest_sha256


def verify_database_reference(
    database_path: Path, *, config: Exp416FrozenConfig = FROZEN_CONFIG
) -> str:
    """Verify only the frozen database bytes; never connect to SQLite."""
    config.validate()
    observed = sha256_file(database_path)
    if observed != config.database_sha256:
        raise ValueError("Exp416 database reference SHA256 mismatch")
    return observed


def freeze_prediction_inputs(
    prediction_path: Path,
    manifest_path: Path,
    *,
    config: Exp416FrozenConfig = FROZEN_CONFIG,
) -> FrozenPrediction:
    """Freeze validated Exp404 prediction coordinates before TEST labels are opened."""
    config.validate()
    prediction_sha256, manifest_sha256 = _verified_prediction_hashes(
        prediction_path, manifest_path, config
    )

    coordinates: set[Coordinate] = set()
    with prediction_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Exp416 invalid prediction JSONL row {line_number}") from error
            if type(row) is not dict:
                raise ValueError(f"Exp416 prediction row {line_number} must be an object")
            for flag in _PREDICTION_FLAGS:
                _require_false(row, flag)
            coordinate = (
                _dataset(row.get("dataset"), field="prediction dataset"),
                _exact_index(row.get("candidate_index"), field="prediction candidate_index"),
            )
            if coordinate in coordinates:
                raise ValueError("Exp416 prediction contains duplicate coordinate")
            coordinates.add(coordinate)

    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source, object_pairs_hook=_json_object_without_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("Exp416 invalid prediction manifest JSON") from error
    if type(manifest) is not dict:
        raise ValueError("Exp416 prediction manifest must be an object")
    for flag in _PREDICTION_FLAGS:
        _require_false(manifest, flag)
    counts = {name: _exact_index(manifest.get(name), field=f"manifest {name}") for name in _MANIFEST_COUNTS}
    scope, lane_coordinates = _manifest_lanes(manifest.get("frozen_hard_lanes"))

    sorted_coordinates = tuple(sorted(coordinates))
    prediction_scope = tuple(sorted({dataset for dataset, _ in sorted_coordinates}))
    if len(scope) != config.expected_datasets:
        raise ValueError("Exp416 manifest population dataset count mismatch")
    if counts["datasets"] != config.expected_prediction_datasets or len(prediction_scope) != config.expected_prediction_datasets:
        raise ValueError("Exp416 prediction dataset count mismatch")
    if (
        counts["rows"] != config.expected_prediction_coordinates
        or len(sorted_coordinates) != config.expected_prediction_coordinates
    ):
        raise ValueError("Exp416 prediction coordinate count mismatch")
    if lane_coordinates != coordinates:
        raise ValueError("Exp416 frozen_hard_lanes coordinate mismatch")
    if not set(prediction_scope) <= set(scope):
        raise ValueError("Exp416 prediction dataset scope is outside manifest population")

    if sha256_file(prediction_path) != prediction_sha256:
        raise ValueError("Exp416 prediction changed after parsing")
    if sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("Exp416 manifest changed after parsing")
    return FrozenPrediction(
        coordinates=sorted_coordinates,
        dataset_scope=prediction_scope,
        prediction_dataset_count=len(prediction_scope),
        identity_sha256=_coordinate_identity(sorted_coordinates),
        prediction_sha256=prediction_sha256,
        manifest_sha256=manifest_sha256,
    )


def _parse_binary_label(raw: str) -> int:
    if raw not in {"0", "1"}:
        raise ValueError("Exp416 TEST label must be literal 0 or 1")
    return int(raw)


def _parse_label_index(raw: object) -> int:
    if type(raw) is not str or not raw or not raw.isdecimal():
        raise ValueError("Exp416 label candidate_index must be a nonnegative exact integer")
    return _exact_index(int(raw), field="label candidate_index")


def load_frozen_test_labels(
    labels_path: Path,
    frozen_prediction: FrozenPrediction,
    *,
    config: Exp416FrozenConfig = FROZEN_CONFIG,
) -> FrozenLabels:
    """Load frozen TEST labels only after a valid prediction freeze exists."""
    if not isinstance(frozen_prediction, FrozenPrediction):
        raise ValueError("Exp416 labels require a frozen prediction")
    config.validate()
    labels_sha256 = sha256_file(labels_path)
    if labels_sha256 != config.labels_sha256:
        raise ValueError("Exp416 TEST label SHA256 mismatch")

    rows: list[LabelRow] = []
    coordinates: set[Coordinate] = set()
    with labels_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != _LABEL_FIELDS:
            raise ValueError("Exp416 TEST label header mismatch")
        for csv_row in reader:
            if set(csv_row) != set(_LABEL_FIELDS) or any(value is None for value in csv_row.values()):
                raise ValueError("Exp416 TEST label row fields mismatch")
            dataset = _dataset(csv_row["dataset"], field="label dataset")
            candidate_index = _parse_label_index(csv_row["candidate_index"])
            label = _parse_binary_label(csv_row["is_true_anomaly"])
            coordinate = (dataset, candidate_index)
            if coordinate in coordinates:
                raise ValueError("Exp416 TEST labels contain duplicate coordinate")
            coordinates.add(coordinate)
            rows.append(LabelRow(dataset, candidate_index, label))

    if sha256_file(labels_path) != labels_sha256:
        raise ValueError("Exp416 TEST labels changed after streaming")
    sorted_rows = tuple(sorted(rows, key=lambda row: row.coordinate))
    scope = tuple(sorted({row.dataset for row in sorted_rows}))
    if len(sorted_rows) != config.expected_label_rows:
        raise ValueError("Exp416 TEST label row count mismatch")
    if len(scope) != config.expected_datasets:
        raise ValueError("Exp416 TEST label dataset count mismatch")
    if not set(frozen_prediction.dataset_scope) <= set(scope):
        raise ValueError("Exp416 TEST label scope does not cover prediction scope")
    if not set(frozen_prediction.coordinates) <= coordinates:
        raise ValueError("Exp416 TEST labels do not cover prediction coordinates")
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
    config: Exp416FrozenConfig = FROZEN_CONFIG,
) -> None:
    """Rehash all inputs without reparsing or replacing the frozen objects."""
    config.validate()
    observed_database_sha256 = verify_database_reference(database_path, config=config)
    if database_sha256 != config.database_sha256 or observed_database_sha256 != database_sha256:
        raise ValueError("Exp416 database reference snapshot mismatch")
    if not isinstance(frozen_prediction, FrozenPrediction):
        raise ValueError("Exp416 frozen prediction snapshot requires FrozenPrediction")
    if not isinstance(frozen_labels, FrozenLabels):
        raise ValueError("Exp416 frozen label snapshot requires FrozenLabels")
    prediction_sha256 = sha256_file(prediction_path)
    manifest_sha256 = sha256_file(manifest_path)
    labels_sha256 = sha256_file(labels_path)
    if (
        frozen_prediction.prediction_sha256 != config.prediction_sha256
        or prediction_sha256 != config.prediction_sha256
    ):
        raise ValueError("Exp416 frozen prediction snapshot mismatch")
    if frozen_prediction.manifest_sha256 != config.manifest_sha256 or manifest_sha256 != config.manifest_sha256:
        raise ValueError("Exp416 frozen manifest snapshot mismatch")
    if frozen_labels.labels_sha256 != config.labels_sha256 or labels_sha256 != config.labels_sha256:
        raise ValueError("Exp416 frozen label snapshot mismatch")


__all__ = [
    "freeze_prediction_inputs",
    "load_frozen_test_labels",
    "verify_database_reference",
    "verify_input_snapshots",
]
