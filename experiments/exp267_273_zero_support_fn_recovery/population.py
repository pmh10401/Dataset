from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import DB_PATH, OUTPUT_ROOT


EXPECTED_DATASETS = 1117
EXP266_OUTPUT_ROOT = Path(
    "/Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow"
)
PREDICTION_FILENAME = "03_frozen_predictions.jsonl"
MANIFEST_FILENAME = "04_prediction_manifest.json"
RESULTS_FILENAME = "05_dataset_results.csv"
SUMMARY_FILENAME = "06_global_summary.csv"


class PopulationContractError(ValueError):
    """Raised when the frozen Exp266 dataset population is not complete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PopulationContractError(f"cannot read input artifact: {path}") from exc
    return digest.hexdigest()


def validate_population(
    names: Iterable[str], *, expected: int = EXPECTED_DATASETS
) -> tuple[str, ...]:
    values = tuple(names)
    if any(type(name) is not str or not name for name in values):
        raise PopulationContractError("frozen population names must be non-empty strings")
    if len(values) != expected or len(set(values)) != expected:
        raise PopulationContractError(
            f"frozen population must contain {expected:,} unique datasets"
        )
    return tuple(sorted(values))


def _prediction_path(path: Path | None) -> Path:
    candidate = EXP266_OUTPUT_ROOT if path is None else Path(path)
    return candidate / PREDICTION_FILENAME if candidate.is_dir() else candidate


def _validate_prediction_boundary(value: object, path: str = "root") -> None:
    forbidden = {
        "label",
        "labels",
        "y_test",
        "test_label",
        "test_labels",
        "anomaly",
        "anomalies",
        "anomaly_index",
        "anomaly_indices",
        "anomaly_position",
        "anomaly_positions",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in forbidden:
                raise PopulationContractError(
                    f"prediction contains label/evaluation field: {path}.{key}"
                )
            if normalized in {
                "contains_test_labels",
                "contains_evaluation_metrics",
            } and child is not False:
                raise PopulationContractError(
                    f"prediction boundary flag must be false: {path}.{key}"
                )
            _validate_prediction_boundary(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_prediction_boundary(child, f"{path}[{index}]")


def read_frozen_prediction_rows(
    prediction_path: Path | None = None,
    *,
    manifest_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Read and verify Exp266 prediction JSONL without reading labels or metrics."""

    path = _prediction_path(prediction_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PopulationContractError(f"cannot read input artifact: {path}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    resolved_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else path.parent / MANIFEST_FILENAME
    )
    try:
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PopulationContractError("cannot read frozen prediction manifest") from exc
    if type(manifest) is not dict:
        raise PopulationContractError("frozen prediction manifest must be an object")
    if manifest.get("prediction_sha256") != actual_sha256:
        raise PopulationContractError(
            "frozen prediction SHA256 mismatch: "
            f"expected={manifest.get('prediction_sha256')} actual={actual_sha256}"
        )

    try:
        raw_lines = payload.decode("utf-8").splitlines()
        rows = [json.loads(line) for line in raw_lines if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PopulationContractError("frozen prediction JSONL is invalid") from exc
    if not rows or any(type(row) is not dict for row in rows):
        raise PopulationContractError("frozen prediction must contain object rows")
    for row in rows:
        _validate_prediction_boundary(row)
    return rows, actual_sha256, manifest


def load_frozen_population(
    prediction_path: Path | None = None,
    *,
    expected: int = EXPECTED_DATASETS,
) -> tuple[str, ...]:
    """Return the sorted, hash-verified Exp266 population."""

    rows, _, _ = read_frozen_prediction_rows(prediction_path)
    return population_from_prediction_rows(rows, expected=expected)


def population_from_prediction_rows(
    rows: Iterable[Mapping[str, object]], *, expected: int = EXPECTED_DATASETS
) -> tuple[str, ...]:
    """Validate a population from already parsed, hash-verified prediction rows."""

    names = [row.get("dataset") for row in rows]
    if any(type(name) is not str for name in names):
        raise PopulationContractError("frozen prediction dataset field is invalid")
    return validate_population(names, expected=expected)


def frozen_input_paths(
    exp266_output_root: Path = EXP266_OUTPUT_ROOT,
) -> tuple[Path, Path, Path]:
    root = Path(exp266_output_root)
    return (
        root / PREDICTION_FILENAME,
        root / RESULTS_FILENAME,
        root / SUMMARY_FILENAME,
    )


def frozen_input_hashes(
    exp266_output_root: Path = EXP266_OUTPUT_ROOT,
) -> dict[str, str]:
    prediction, results, summary = frozen_input_paths(exp266_output_root)
    return {
        PREDICTION_FILENAME: sha256_file(prediction),
        RESULTS_FILENAME: sha256_file(results),
        SUMMARY_FILENAME: sha256_file(summary),
    }


__all__ = [
    "DB_PATH",
    "EXPECTED_DATASETS",
    "EXP266_OUTPUT_ROOT",
    "MANIFEST_FILENAME",
    "OUTPUT_ROOT",
    "PREDICTION_FILENAME",
    "PopulationContractError",
    "RESULTS_FILENAME",
    "SUMMARY_FILENAME",
    "frozen_input_hashes",
    "frozen_input_paths",
    "load_frozen_population",
    "population_from_prediction_rows",
    "read_frozen_prediction_rows",
    "sha256_file",
    "validate_population",
]
