from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Collection, Mapping
from pathlib import Path

from .config import (
    ALPHA,
    EVIDENCE_GROUPS,
    EXPECTED_DATASETS,
    EXPERIMENT_ID,
    FULL_RUN_WORKERS,
    MINIMUM_TRAIN,
    RISK_TARGET,
    SEED,
)
from .contracts import FrozenDatasetInputs, FrozenRecoveryInputs
from .inputs import load_frozen_inputs
from .routing import RecoveryPrediction, route_recovery


PREDICTION_FILENAME = "03_frozen_predictions.jsonl"
MANIFEST_FILENAME = "04_prediction_manifest.json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(path, _canonical_json(value) + b"\n")


def _config_snapshot() -> dict[str, object]:
    return {
        "alpha": ALPHA,
        "evidence_groups": list(EVIDENCE_GROUPS),
        "expected_datasets": EXPECTED_DATASETS,
        "experiment_id": EXPERIMENT_ID,
        "full_run_workers": FULL_RUN_WORKERS,
        "minimum_train": MINIMUM_TRAIN,
        "risk_target": RISK_TARGET,
        "seed": SEED,
    }


def _sorted_indices(values: Collection[int]) -> list[int]:
    return sorted(values)


def _support_sources(inputs: FrozenDatasetInputs) -> dict[str, list[int]]:
    return {
        group: _sorted_indices(inputs.evidence[group])
        for group in EVIDENCE_GROUPS
    }


def _prediction_row(
    inputs: FrozenDatasetInputs,
    prediction: RecoveryPrediction,
    recovery: FrozenRecoveryInputs,
) -> dict[str, object]:
    config = _config_snapshot()
    return {
        "dataset": inputs.dataset,
        "n_train": inputs.n_train,
        "n_test": inputs.n_test,
        "hard_indices": _sorted_indices(prediction.hard),
        "shadow_indices": _sorted_indices(prediction.shadow),
        "priority_indices": _sorted_indices(prediction.priority),
        "standard_indices": _sorted_indices(prediction.standard),
        "recovery_shadow_indices": _sorted_indices(prediction.recovery_shadow),
        "recovery_priority_indices": _sorted_indices(prediction.recovery_priority),
        "data_quality_review_indices": _sorted_indices(prediction.data_quality_review),
        "support_by_index": {
            str(index): count for index, count in prediction.support_by_index.items()
        },
        "support_sources": _support_sources(inputs),
        "source_availability": {
            group: group in inputs.available_groups for group in EVIDENCE_GROUPS
        },
        "quality_flag_indices": _sorted_indices(inputs.quality_flags),
        "input_sha256": dict(sorted(recovery.input_sha256.items())),
        "config_sha256": hashlib.sha256(_canonical_json(config)).hexdigest(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
    }


def freeze_rows(output_root: Path, rows: Collection[Mapping[str, object]]) -> Path:
    output_root = Path(output_root)
    ordered = sorted(rows, key=lambda row: str(row["dataset"]))
    payload = b"".join(_canonical_json(row) + b"\n" for row in ordered)
    path = output_root / PREDICTION_FILENAME
    _atomic_write(path, payload)
    _atomic_json(
        output_root / MANIFEST_FILENAME,
        {
            "experiment_id": EXPERIMENT_ID,
            "rows": len(ordered),
            "datasets": len(ordered),
            "prediction_sha256": hashlib.sha256(payload).hexdigest(),
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "retrospective": True,
        },
    )
    return path


def predict_all(output_root: Path, dataset_limit: int | None = None) -> Path:
    """Freeze label-free recovery predictions from the immutable Exp266 inputs."""

    if dataset_limit is not None and (type(dataset_limit) is not int or dataset_limit < 0):
        raise ValueError("dataset_limit must be a non-negative integer or None")

    output_root = Path(output_root)
    recovery = load_frozen_inputs(output_root=output_root)
    population = recovery.population if dataset_limit is None else recovery.population[:dataset_limit]
    rows = [
        _prediction_row(recovery.datasets[dataset], route_recovery(recovery.datasets[dataset]), recovery)
        for dataset in population
    ]
    path = freeze_rows(output_root, rows)
    manifest_path = output_root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "config": _config_snapshot(),
            "config_sha256": hashlib.sha256(_canonical_json(_config_snapshot())).hexdigest(),
            "input_sha256": dict(sorted(recovery.input_sha256.items())),
            "coverage": {
                "expected_datasets": EXPECTED_DATASETS,
                "frozen_population": len(recovery.population),
                "predicted_datasets": len(population),
            },
            "checkpoint": PREDICTION_FILENAME,
        }
    )
    _atomic_json(manifest_path, manifest)
    return path
