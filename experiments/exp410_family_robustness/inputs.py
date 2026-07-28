"""Hash-first loading and independent metric reconstruction for Exp410."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import csv
import json
import math
from pathlib import Path
from types import MappingProxyType

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_file

from .config import FROZEN_CONFIG, Exp410FrozenConfig
from .contracts import DatasetMetric, FrozenAuditInputs, family_from_dataset


PREDICTION_MANIFEST = "exp404_prediction_manifest.json"
PREDICTION_ROWS = "exp404_hard_confidence.jsonl"
FROZEN_LABELS = "exp403_408_frozen_test_labels.csv"
INPUT_MANIFEST = "exp403_408_evaluation_input_manifest.json"
PUBLICATION_MANIFEST = "exp403_408_evaluation_publication_manifest.json"

_LANE_FIELDS = (
    "automatic_hard_indices",
    "standard_review_indices",
    "priority_review_indices",
)


def load_verified_inputs(
    input_dir: Path,
    exp291_path: Path,
    exp292_path: Path,
    config: Exp410FrozenConfig = FROZEN_CONFIG,
) -> FrozenAuditInputs:
    """Load only publications whose manifests and frozen hashes all agree."""
    publication = Path(input_dir)
    paths = {
        PREDICTION_MANIFEST: publication / PREDICTION_MANIFEST,
        PREDICTION_ROWS: publication / PREDICTION_ROWS,
        FROZEN_LABELS: publication / FROZEN_LABELS,
        INPUT_MANIFEST: publication / INPUT_MANIFEST,
        PUBLICATION_MANIFEST: publication / PUBLICATION_MANIFEST,
    }
    input_manifest = _read_json(paths[INPUT_MANIFEST])
    publication_manifest = _read_json(paths[PUBLICATION_MANIFEST])
    input_hashes = _artifact_hashes(input_manifest, INPUT_MANIFEST)
    publication_hashes = _artifact_hashes(publication_manifest, PUBLICATION_MANIFEST)

    _require_bound_hash(input_hashes, paths[PREDICTION_MANIFEST], INPUT_MANIFEST)
    _require_bound_hash(input_hashes, paths[PREDICTION_ROWS], INPUT_MANIFEST)
    _require_bound_hash(publication_hashes, paths[FROZEN_LABELS], PUBLICATION_MANIFEST)
    _require_bound_hash(publication_hashes, paths[INPUT_MANIFEST], PUBLICATION_MANIFEST)

    observed_prediction = _require_sha256(paths[PREDICTION_ROWS], config.prediction_sha256)
    observed_labels = _require_sha256(paths[FROZEN_LABELS], config.frozen_labels_sha256)
    observed_exp291 = _require_sha256(Path(exp291_path), config.exp291_sha256)
    observed_exp292 = _require_sha256(Path(exp292_path), config.exp292_sha256)

    prediction_manifest = _read_json(paths[PREDICTION_MANIFEST])
    if prediction_manifest.get("hard_confidence_file") != PREDICTION_ROWS:
        raise ValueError("Exp410 prediction manifest filename mismatch")
    artifact_hashes = _mapping(
        prediction_manifest.get("artifact_hashes"),
        "Exp410 prediction manifest artifact_hashes",
    )
    if artifact_hashes.get("prediction") != observed_prediction:
        raise ValueError("Exp410 prediction manifest SHA256 mismatch")

    labels = _load_labels(paths[FROZEN_LABELS])
    hard_from_rows = _load_hard_rows(paths[PREDICTION_ROWS])
    manifest_hard = _manifest_hard_lanes(prediction_manifest)
    hard_from_rows = {
        dataset: hard_from_rows.get(dataset, frozenset())
        for dataset in set(hard_from_rows) | set(manifest_hard)
    }
    exp291_lanes = _load_lane_rows(Path(exp291_path))
    exp292_lanes = _load_lane_rows(Path(exp292_path))

    if _canonical_lanes(exp291_lanes) != _canonical_lanes(exp292_lanes):
        raise ValueError("Exp410 Exp291/292 lane parity mismatch")
    exp291_hard = {
        dataset: lanes["automatic_hard_indices"]
        for dataset, lanes in exp291_lanes.items()
    }
    if _canonical_hard(exp291_hard) != _canonical_hard(manifest_hard):
        raise ValueError("Exp410 Exp291/292 and prediction Hard lane parity mismatch")
    if _canonical_hard(hard_from_rows) != _canonical_hard(manifest_hard):
        raise ValueError("Exp410 prediction row Hard lane parity mismatch")

    datasets = set(labels)
    if set(manifest_hard) != datasets:
        raise ValueError("Exp410 frozen label and Hard dataset coverage mismatch")
    if len(datasets) != config.expected_datasets:
        raise ValueError("Exp410 frozen dataset count mismatch")
    families = {family_from_dataset(dataset) for dataset in datasets}
    if len(families) != config.expected_families:
        raise ValueError("Exp410 frozen family count mismatch")

    return FrozenAuditInputs(
        labels_by_dataset=MappingProxyType(
            {dataset: labels[dataset] for dataset in sorted(labels)}
        ),
        hard_by_dataset=MappingProxyType(
            {dataset: manifest_hard[dataset] for dataset in sorted(manifest_hard)}
        ),
        input_sha256=MappingProxyType(
            {
                PREDICTION_ROWS: observed_prediction,
                FROZEN_LABELS: observed_labels,
                "exp291_frozen_lanes": observed_exp291,
                "exp292_frozen_lanes": observed_exp292,
                PREDICTION_MANIFEST: sha256_file(paths[PREDICTION_MANIFEST]),
                INPUT_MANIFEST: sha256_file(paths[INPUT_MANIFEST]),
                PUBLICATION_MANIFEST: sha256_file(paths[PUBLICATION_MANIFEST]),
            }
        ),
    )


def recompute_dataset_metrics(
    inputs: FrozenAuditInputs,
    config: Exp410FrozenConfig = FROZEN_CONFIG,
) -> tuple[DatasetMetric, ...]:
    """Recompute every Hard metric without importing official evaluation code."""
    if set(inputs.labels_by_dataset) != set(inputs.hard_by_dataset):
        raise ValueError("Exp410 labels and Hard lanes cover different datasets")
    rows = tuple(
        _metric(
            dataset,
            inputs.labels_by_dataset[dataset],
            inputs.hard_by_dataset[dataset],
        )
        for dataset in sorted(inputs.labels_by_dataset)
    )
    if len(rows) != config.expected_datasets:
        raise ValueError("Exp410 dataset metric count mismatch")
    if len({row.family for row in rows}) != config.expected_families:
        raise ValueError("Exp410 dataset metric family count mismatch")
    totals = (
        sum(row.hard_count for row in rows),
        sum(row.tp for row in rows),
        sum(row.fp for row in rows),
        sum(row.fn for row in rows),
    )
    expected = (
        config.expected_hard,
        config.expected_tp,
        config.expected_fp,
        config.expected_fn,
    )
    if totals != expected:
        raise ValueError(f"Exp410 frozen Hard TP/FP/FN count mismatch: {totals!r}")
    mean_f1 = math.fsum(row.f1 for row in rows) / len(rows)
    if abs(mean_f1 - config.expected_dataset_macro_f1) > config.float_tolerance:
        raise ValueError("Exp410 exact dataset-macro mean F1 mismatch")
    if f"{mean_f1:.6f}" != config.display_dataset_macro_f1:
        raise ValueError("Exp410 dataset-macro mean F1 display mismatch")
    return rows


def _metric(
    dataset: str,
    labels: tuple[int, ...],
    hard: frozenset[int],
) -> DatasetMetric:
    if not labels or any(value not in (0, 1) for value in labels):
        raise ValueError("Exp410 frozen labels must be nonempty binary values")
    if not all(0 <= index < len(labels) for index in hard):
        raise ValueError("Exp410 Hard index is outside frozen TEST range")
    positives = sum(labels)
    if positives == 0:
        raise ValueError("Exp410 pinned dataset must contain an anomaly")
    tp = sum(labels[index] for index in hard)
    fp = len(hard) - tp
    fn = positives - tp
    precision = None if not hard else tp / len(hard)
    recall = tp / positives
    denominator = 2 * tp + fp + fn
    f1 = 0.0 if denominator == 0 else 2 * tp / denominator
    return DatasetMetric(
        dataset=dataset,
        family=family_from_dataset(dataset),
        hard_count=len(hard),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _load_labels(path: Path) -> dict[str, tuple[int, ...]]:
    rows = _read_csv(path)
    values: dict[str, dict[int, int]] = defaultdict(dict)
    for row in rows:
        dataset = _text(row.get("dataset"), "Exp410 frozen label dataset")
        family_from_dataset(dataset)
        index = _integer(row.get("candidate_index"), "Exp410 frozen label index")
        value = _integer(row.get("is_true_anomaly"), "Exp410 frozen label value")
        if index < 0 or value not in (0, 1):
            raise ValueError("Exp410 frozen labels contain an invalid index or value")
        if index in values[dataset]:
            raise ValueError("Exp410 frozen labels contain duplicate indices")
        values[dataset][index] = value
    if not values:
        raise ValueError("Exp410 frozen labels are empty")
    result: dict[str, tuple[int, ...]] = {}
    for dataset, indexed in values.items():
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Exp410 frozen label indices must be contiguous from zero")
        result[dataset] = tuple(indexed[index] for index in range(len(indexed)))
    return result


def _load_hard_rows(path: Path) -> dict[str, frozenset[int]]:
    values: dict[str, set[int]] = defaultdict(set)
    for row in _read_jsonl(path):
        dataset = _text(row.get("dataset"), "Exp410 prediction dataset")
        family_from_dataset(dataset)
        index = _integer(row.get("candidate_index"), "Exp410 prediction index")
        if index < 0 or index in values[dataset]:
            raise ValueError("Exp410 prediction contains an invalid or duplicate Hard identity")
        values[dataset].add(index)
    if not values:
        raise ValueError("Exp410 prediction rows are empty")
    return {dataset: frozenset(indices) for dataset, indices in values.items()}


def _manifest_hard_lanes(manifest: Mapping[str, object]) -> dict[str, frozenset[int]]:
    raw_lanes = _mapping(
        manifest.get("frozen_hard_lanes"),
        "Exp410 prediction manifest frozen_hard_lanes",
    )
    if not raw_lanes:
        raise ValueError("Exp410 prediction manifest frozen Hard lanes are empty")
    return {
        str(dataset): _indices(indices, f"Exp410 frozen Hard lane {dataset}")
        for dataset, indices in raw_lanes.items()
    }


def _load_lane_rows(
    path: Path,
) -> dict[str, dict[str, frozenset[int]]]:
    result: dict[str, dict[str, frozenset[int]]] = {}
    for row in _read_jsonl(path):
        dataset = _text(row.get("dataset"), "Exp410 frozen lane dataset")
        family_from_dataset(dataset)
        if dataset in result:
            raise ValueError("Exp410 frozen lane file contains duplicate datasets")
        result[dataset] = {
            field: _indices(row.get(field, []), f"Exp410 {field}")
            for field in _LANE_FIELDS
        }
    if not result:
        raise ValueError("Exp410 frozen lane file is empty")
    return result


def _canonical_lanes(lanes: Mapping[str, Mapping[str, frozenset[int]]]) -> bytes:
    return canonical_json_bytes(
        {
            dataset: {
                field: sorted(values[field])
                for field in _LANE_FIELDS
            }
            for dataset, values in sorted(lanes.items())
        }
    )


def _canonical_hard(lanes: Mapping[str, frozenset[int]]) -> bytes:
    return canonical_json_bytes(
        {dataset: sorted(indices) for dataset, indices in sorted(lanes.items())}
    )


def _artifact_hashes(manifest: Mapping[str, object], name: str) -> dict[str, str]:
    raw = _mapping(manifest.get("artifact_sha256"), f"{name} artifact_sha256")
    hashes: dict[str, str] = {}
    for artifact, digest in raw.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} artifact SHA256 is invalid")
        hashes[str(artifact)] = digest
    return hashes


def _require_bound_hash(
    hashes: Mapping[str, str],
    path: Path,
    manifest_name: str,
) -> None:
    expected = hashes.get(path.name)
    if expected is None:
        raise ValueError(f"{manifest_name} does not bind required artifact: {path.name}")
    _require_sha256(path, expected)


def _require_sha256(path: Path, expected: str) -> str:
    try:
        actual = sha256_file(path)
    except OSError as error:
        raise ValueError(f"Exp410 cannot read required artifact: {path}") from error
    if actual != expected:
        raise ValueError(f"Exp410 SHA256 mismatch: {path.name}")
    return actual


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Exp410 cannot read JSON artifact: {path}") from error
    return dict(_mapping(value, f"Exp410 JSON artifact {path.name}"))


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    try:
        rows = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Exp410 cannot read JSONL artifact: {path}") from error
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"Exp410 JSONL artifact is invalid: {path}")
    return tuple(dict(row) for row in rows)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"dataset", "candidate_index", "is_true_anomaly"}
            if reader.fieldnames is None or not required <= set(reader.fieldnames):
                raise ValueError(f"Exp410 CSV columns are invalid: {path}")
            rows = tuple(dict(row) for row in reader)
    except OSError as error:
        raise ValueError(f"Exp410 cannot read CSV artifact: {path}") from error
    if not rows:
        raise ValueError(f"Exp410 CSV artifact is empty: {path}")
    return rows


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(value).strip() != str(integer):
        raise ValueError(f"{name} must be an integer")
    return integer


def _indices(value: object, name: str) -> frozenset[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    indices = tuple(_integer(item, name) for item in value)
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains an invalid or duplicate index")
    return frozenset(indices)
