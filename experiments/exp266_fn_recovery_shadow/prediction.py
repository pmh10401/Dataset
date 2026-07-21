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
    FROZEN_PREDICTION_HASHES,
    FULL_RUN_WORKERS,
    MINIMUM_TRAIN,
    RISK_TARGET,
    SEED,
)
from .contracts import FrozenDatasetInputs, FrozenRecoveryInputs
from .inputs import InputContractError, load_frozen_inputs, validate_prediction_boundary
from .routing import RecoveryPrediction, route_recovery


PREDICTION_FILENAME = "03_frozen_predictions.jsonl"
MANIFEST_FILENAME = "04_prediction_manifest.json"
PENDING_FILENAME = ".03_frozen_predictions.pending.json"


class CheckpointError(ValueError):
    """Raised when a prediction checkpoint cannot be accepted safely."""


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
        _sync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(path, _canonical_json(value) + b"\n")


def _remove_checkpoint_marker(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _sync_directory(path.parent)


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read {description}") from exc
    if type(value) is not dict:
        raise CheckpointError(f"invalid {description}")
    return value


def _checkpoint_generation(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_serialized_prediction_row(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "n_test":
                raise InputContractError(f"TEST length cannot be published: {path}.{key}")
            _validate_serialized_prediction_row(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_serialized_prediction_row(child, f"{path}[{index}]")


def verify_prediction_checkpoint(output_root: Path) -> dict[str, object]:
    """Return a manifest only when its prediction file is complete and hash-bound."""

    output_root = Path(output_root)
    if (output_root / PENDING_FILENAME).exists():
        raise CheckpointError("prediction checkpoint publication is pending")
    manifest = _read_json_object(output_root / MANIFEST_FILENAME, "prediction manifest")
    prediction_path = output_root / PREDICTION_FILENAME
    try:
        actual_sha256 = _checkpoint_generation(prediction_path.read_bytes())
    except OSError as exc:
        raise CheckpointError("cannot read frozen predictions") from exc
    expected_sha256 = manifest.get("prediction_sha256")
    if type(expected_sha256) is not str or expected_sha256 != actual_sha256:
        raise CheckpointError("prediction manifest SHA256 mismatch")
    if manifest.get("generation") != actual_sha256:
        raise CheckpointError("prediction manifest generation mismatch")
    _validate_prediction_manifest_contract(output_root, manifest, prediction_path)
    return manifest


def recover_prediction_checkpoint(output_root: Path) -> dict[str, object]:
    """Finalize an interrupted publish only when both files match the marker."""

    output_root = Path(output_root)
    pending_path = output_root / PENDING_FILENAME
    if not pending_path.exists():
        return verify_prediction_checkpoint(output_root)
    pending = _read_json_object(pending_path, "pending prediction checkpoint")
    manifest = _read_json_object(output_root / MANIFEST_FILENAME, "prediction manifest")
    prediction_path = output_root / PREDICTION_FILENAME
    try:
        actual_sha256 = _checkpoint_generation(prediction_path.read_bytes())
    except OSError as exc:
        raise CheckpointError("cannot recover missing frozen predictions") from exc
    expected_sha256 = pending.get("prediction_sha256")
    if (
        type(expected_sha256) is not str
        or pending.get("generation") != expected_sha256
        or actual_sha256 != expected_sha256
        or manifest.get("generation") != expected_sha256
        or manifest.get("prediction_sha256") != expected_sha256
    ):
        raise CheckpointError("pending prediction checkpoint does not match manifest")
    _remove_checkpoint_marker(pending_path)
    return verify_prediction_checkpoint(output_root)


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


def _lane_policy() -> dict[str, dict[str, bool]]:
    return {
        "hard": {"autonomous": True, "unchanged": True},
        "recovery_shadow": {
            "deployment_eligible": False,
            "human_assisted": True,
            "review_only": True,
        },
        "recovery_priority": {
            "deployment_eligible": False,
            "human_assisted": True,
            "review_only": True,
        },
        "data_quality_review": {
            "deployment_eligible": False,
            "human_assisted": True,
            "review_only": True,
        },
    }


def _read_prediction_rows(path: Path) -> list[Mapping[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError("cannot parse frozen predictions") from exc
    if any(type(row) is not dict for row in rows):
        raise CheckpointError("frozen prediction row is not an object")
    try:
        for row in rows:
            validate_prediction_boundary(row, "frozen prediction row")
            _validate_serialized_prediction_row(row)
    except InputContractError as exc:
        raise CheckpointError("frozen prediction row violates boundary") from exc
    return rows


def _validate_prediction_manifest_contract(
    output_root: Path,
    manifest: Mapping[str, object],
    prediction_path: Path,
) -> None:
    required = {
        "checkpoint",
        "complete_population",
        "config",
        "config_sha256",
        "contains_evaluation_metrics",
        "contains_test_labels",
        "coverage",
        "datasets",
        "deployment_eligible",
        "experiment_id",
        "generation",
        "input_sha256",
        "lane_policy",
        "prediction_sha256",
        "retrospective",
        "rows",
        "smoke",
    }
    if required - set(manifest):
        raise CheckpointError("prediction manifest contract fields are missing")
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("checkpoint") != PREDICTION_FILENAME
        or manifest.get("contains_test_labels") is not False
        or manifest.get("contains_evaluation_metrics") is not False
        or manifest.get("retrospective") is not True
        or manifest.get("deployment_eligible") is not False
    ):
        raise CheckpointError("prediction manifest contract mismatch")
    config = _config_snapshot()
    if (
        manifest.get("config") != config
        or manifest.get("config_sha256") != hashlib.sha256(_canonical_json(config)).hexdigest()
        or manifest.get("input_sha256") != dict(sorted(FROZEN_PREDICTION_HASHES.items()))
        or manifest.get("lane_policy") != _lane_policy()
    ):
        raise CheckpointError("prediction manifest frozen contract mismatch")

    rows = manifest.get("rows")
    datasets = manifest.get("datasets")
    if type(rows) is not int or type(datasets) is not int or rows < 0 or datasets < 0:
        raise CheckpointError("prediction manifest coverage is invalid")
    prediction_rows = _read_prediction_rows(prediction_path)
    names = [row.get("dataset") for row in prediction_rows]
    if (
        len(prediction_rows) != rows
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != datasets
        or rows != datasets
    ):
        raise CheckpointError("prediction manifest row coverage mismatch")

    coverage = manifest.get("coverage")
    if type(coverage) is not dict or coverage != {
        "expected_datasets": EXPECTED_DATASETS,
        "frozen_population": EXPECTED_DATASETS,
        "predicted_datasets": rows,
    }:
        raise CheckpointError("prediction manifest coverage contract mismatch")
    smoke = manifest.get("smoke")
    complete_population = manifest.get("complete_population")
    if type(smoke) is not bool or type(complete_population) is not bool or smoke == complete_population:
        raise CheckpointError("prediction manifest smoke contract mismatch")
    if complete_population:
        if rows != EXPECTED_DATASETS or output_root.parent.name == "smoke":
            raise CheckpointError("full prediction location or coverage mismatch")
        return
    if (
        rows >= EXPECTED_DATASETS
        or output_root.parent.name != "smoke"
        or output_root.name != f"dataset_limit_{rows}"
    ):
        raise CheckpointError("smoke prediction location or coverage mismatch")


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


def freeze_rows(
    output_root: Path,
    rows: Collection[Mapping[str, object]],
    *,
    manifest_extra: Mapping[str, object] | None = None,
) -> Path:
    output_root = Path(output_root)
    for row in rows:
        validate_prediction_boundary(row, "prediction row")
        _validate_serialized_prediction_row(row)
    ordered = sorted(rows, key=lambda row: str(row["dataset"]))
    payload = b"".join(_canonical_json(row) + b"\n" for row in ordered)
    path = output_root / PREDICTION_FILENAME
    generation = _checkpoint_generation(payload)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "rows": len(ordered),
        "datasets": len(ordered),
        "generation": generation,
        "prediction_sha256": generation,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
    }
    if manifest_extra is not None:
        manifest.update(manifest_extra)
    _atomic_json(
        output_root / PENDING_FILENAME,
        {
            "generation": generation,
            "prediction_filename": PREDICTION_FILENAME,
            "prediction_sha256": generation,
        },
    )
    _atomic_write(path, payload)
    _atomic_json(output_root / MANIFEST_FILENAME, manifest)
    _remove_checkpoint_marker(output_root / PENDING_FILENAME)
    return path


def _publication_root(output_root: Path, dataset_limit: int | None) -> Path:
    root = Path(output_root)
    if dataset_limit is None:
        return root
    return root / "smoke" / f"dataset_limit_{dataset_limit}"


def _validate_complete_population(recovery: FrozenRecoveryInputs) -> None:
    population = recovery.population
    if (
        len(population) != EXPECTED_DATASETS
        or len(set(population)) != EXPECTED_DATASETS
        or set(population) != set(recovery.datasets)
    ):
        raise ValueError("full prediction requires exactly 1,117 unique datasets")


def predict_all(output_root: Path, dataset_limit: int | None = None) -> Path:
    """Freeze label-free recovery predictions from the immutable Exp266 inputs."""

    if dataset_limit is not None and (
        type(dataset_limit) is not int or dataset_limit < 0 or dataset_limit >= EXPECTED_DATASETS
    ):
        raise ValueError("dataset_limit must be a non-negative partial integer or None")

    publication_root = _publication_root(output_root, dataset_limit)
    recovery = load_frozen_inputs(output_root=publication_root)
    if dataset_limit is None:
        _validate_complete_population(recovery)
    population = recovery.population if dataset_limit is None else recovery.population[:dataset_limit]
    rows = [
        _prediction_row(recovery.datasets[dataset], route_recovery(recovery.datasets[dataset]), recovery)
        for dataset in population
    ]
    return freeze_rows(
        publication_root,
        rows,
        manifest_extra={
            "checkpoint": PREDICTION_FILENAME,
            "complete_population": dataset_limit is None,
            "config": _config_snapshot(),
            "config_sha256": hashlib.sha256(_canonical_json(_config_snapshot())).hexdigest(),
            "coverage": {
                "expected_datasets": EXPECTED_DATASETS,
                "frozen_population": len(recovery.population),
                "predicted_datasets": len(population),
            },
            "deployment_eligible": False,
            "input_sha256": dict(sorted(recovery.input_sha256.items())),
            "lane_policy": _lane_policy(),
            "smoke": dataset_limit is not None,
        },
    )
