from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .config import (
    ALPHA,
    CACHE_CONTRACTS,
    DB_PATH,
    EXPECTED_DATASETS,
    EXPECTED_LANE_TOTALS,
    EXP253_PREDICTION,
    EXP264_PREDICTION,
    FROZEN_AUDIT_HASHES,
    FROZEN_AUDIT_CONFIGS,
    FROZEN_CACHE_MANIFEST_HASHES,
    FROZEN_DATABASE_EXCLUSIONS,
    FROZEN_DATABASE_SHA256,
    FROZEN_PREDICTION_HASHES,
    FROZEN_PREDICTION_AUDIT_HASHES,
    FROZEN_PREDICTION_MANIFEST_HASHES,
    MINI_VARIANTS,
    MINIROCKET_PREDICTION,
    OUTPUT_ROOT,
    RECON_ARCHIVE_HASH_INVENTORY,
    RECON_CACHE_ROOT,
    FROZEN_RECON_ARCHIVE_HASH_INVENTORY_SHA256,
)
from .contracts import FrozenDatasetInputs, FrozenRecoveryInputs
from .database import DatabaseContractError, verify_database_sha256


class InputContractError(ValueError):
    """Raised when an immutable Exp266 source does not satisfy its contract."""


FORBIDDEN_PREDICTION_FIELDS = frozenset(
    {
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
        "evaluation",
        "evaluation_metric",
        "evaluation_metrics",
        "metric",
        "metrics",
        "auc",
        "f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
    }
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalized_field_name(value: object) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_anomaly_position_presence_flag(normalized: str) -> bool:
    parts = set(normalized.split("_"))
    return (
        bool({"anomaly", "anomalies"} & parts)
        and bool({"position", "positions", "pos", "index", "indices", "indexes"} & parts)
        and bool({"has", "contains", "present", "available"} & parts)
    )


def validate_prediction_boundary(value: object, source: str, path: str = "root") -> None:
    """Reject labels, anomaly positions, and evaluation data before routing."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_field_name(key)
            if normalized in FORBIDDEN_PREDICTION_FIELDS:
                raise InputContractError(
                    f"forbidden prediction field: {source} {path}.{key}"
                )
            if _is_anomaly_position_presence_flag(normalized) and child is not False:
                raise InputContractError(
                    f"anomaly-position presence flag must be false: {source} {path}.{key}"
                )
            if normalized in {"contains_test_labels", "contains_evaluation_metrics"} and child is not False:
                raise InputContractError(
                    f"prediction boundary flag must be false: {source} {path}.{key}"
                )
            validate_prediction_boundary(child, source, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_prediction_boundary(child, source, f"{path}[{index}]")


def validate_cache_archive_boundary(path: Path, source: str, dataset: str) -> None:
    """Reject cached NPZ keys that would expose labels or evaluation results."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                normalized = _normalized_field_name(key)
                if (
                    normalized in FORBIDDEN_PREDICTION_FIELDS
                    or _is_anomaly_position_presence_flag(normalized)
                ):
                    raise InputContractError(
                        f"forbidden cache field: {source} dataset={dataset} key={key}"
                    )
    except InputContractError:
        raise
    except (OSError, ValueError) as exc:
        raise InputContractError(
            f"cannot read cache archive: {source} dataset={dataset}"
        ) from exc


def validate_population(names: Collection[str]) -> None:
    if len(names) != EXPECTED_DATASETS or len(set(names)) != EXPECTED_DATASETS:
        raise InputContractError("frozen population must contain 1,117 unique datasets")


def validate_indices(dataset: str, n_test: int, values: Collection[int]) -> frozenset[int]:
    indices = frozenset(values)
    if any(type(index) is not int for index in indices):
        raise InputContractError(f"non-integral index: dataset={dataset}")
    if any(index < 0 or index >= n_test for index in indices):
        raise InputContractError(f"candidate outside TEST range: dataset={dataset}")
    if len(indices) != len(values):
        raise InputContractError(f"duplicate candidate index: dataset={dataset}")
    return indices


def group_kernel_candidates(
    rows_1024: Mapping[str, Mapping[str, object]],
    rows_4096: Mapping[str, Mapping[str, object]],
) -> dict[str, frozenset[int]]:
    if set(rows_1024) != set(rows_4096):
        raise InputContractError("MiniROCKET population mismatch")
    grouped: dict[str, frozenset[int]] = {}
    for dataset in sorted(rows_1024):
        first = rows_1024[dataset].get("candidate_indices")
        second = rows_4096[dataset].get("candidate_indices")
        if type(first) is not list or type(second) is not list:
            raise InputContractError(f"MiniROCKET candidate schema mismatch: dataset={dataset}")
        grouped[dataset] = frozenset((*first, *second))
    return grouped


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputContractError(f"cannot read {description}: {path}") from exc
    if type(value) is not dict:
        raise InputContractError(f"{description} is not an object: {path}")
    return value


def _read_jsonl(path: Path, source: str) -> list[dict[str, object]]:
    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise InputContractError(f"cannot read frozen source: {source}") from exc
    if any(type(row) is not dict for row in rows):
        raise InputContractError(f"source row is not an object: {source}")
    for row in rows:
        validate_prediction_boundary(row, source)
    return rows


def _validate_prediction_artifact(
    name: str, path: Path, expected_rows: int
) -> list[dict[str, object]]:
    actual_hash = sha256_file(path)
    expected_hash = FROZEN_PREDICTION_HASHES[name]
    if actual_hash != expected_hash:
        raise InputContractError(f"frozen SHA256 mismatch: {name}")
    manifest_path = path.parent / "prediction_manifest.json"
    if sha256_file(manifest_path) != FROZEN_PREDICTION_MANIFEST_HASHES[name]:
        raise InputContractError(f"frozen manifest SHA256 mismatch: {name}")
    manifest = _read_json(manifest_path, f"{name} manifest")
    validate_prediction_boundary(manifest, f"{name} manifest")
    if manifest.get("prediction_sha256") != expected_hash:
        raise InputContractError(f"manifest SHA256 mismatch: {name}")
    if manifest.get("contains_test_labels") is not False:
        raise InputContractError(f"source manifest contains TEST labels: {name}")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise InputContractError(f"source manifest contains evaluation metrics: {name}")
    rows = _read_jsonl(path, name)
    if len(rows) != expected_rows:
        raise InputContractError(f"source row count mismatch: {name}")
    if manifest.get("rows", len(rows)) != len(rows):
        raise InputContractError(f"manifest row count mismatch: {name}")
    audit_path = (
        path.parent / "00_source_contract_audit.json"
        if name == "exp264"
        else path.parent / "audit.json"
    )
    if sha256_file(audit_path) != FROZEN_PREDICTION_AUDIT_HASHES[name]:
        raise InputContractError(f"frozen audit SHA256 mismatch: {name}")
    audit = _read_json(audit_path, f"{name} audit")
    validate_prediction_boundary(audit, f"{name} audit")
    _validate_prediction_provenance(name, audit)
    return rows


def _validate_prediction_provenance(name: str, audit: Mapping[str, object]) -> None:
    if name == "exp253":
        inputs = audit.get("input_sha256")
        if (
            audit.get("alpha") != ALPHA
            or audit.get("datasets") != EXPECTED_DATASETS
            or type(inputs) is not dict
            or inputs.get("exp249") != FROZEN_PREDICTION_HASHES["minirocket"]
        ):
            raise InputContractError("frozen prediction provenance mismatch: exp253")
        return
    if name == "exp264":
        inputs = audit.get("input_sha256")
        source_contract = audit.get("source_contract")
        baseline = source_contract.get("baseline") if type(source_contract) is dict else None
        if (
            audit.get("dataset_count") != EXPECTED_DATASETS
            or audit.get("expected_population") != EXPECTED_DATASETS
            or type(inputs) is not dict
            or inputs.get("baseline") != FROZEN_PREDICTION_HASHES["exp253"]
            or baseline != {
                "experiment": "exp253",
                "variant": "final_lanes",
                "provenance_audit": str(EXP253_PREDICTION.parent / "audit.json"),
            }
        ):
            raise InputContractError("frozen prediction provenance mismatch: exp264")
        return
    config = audit.get("config")
    if type(config) is not dict or any(
        config.get(field) != value
        for field, value in FROZEN_AUDIT_CONFIGS["kernel"].items()
    ):
        raise InputContractError("frozen prediction provenance mismatch: minirocket")


def _by_dataset(rows: Collection[Mapping[str, object]], source: str) -> dict[str, Mapping[str, object]]:
    mapped: dict[str, Mapping[str, object]] = {}
    for row in rows:
        dataset = row.get("dataset")
        if type(dataset) is not str or not dataset:
            raise InputContractError(f"missing dataset name: {source}")
        if dataset in mapped:
            raise InputContractError(f"duplicate source row: {source} dataset={dataset}")
        if row.get("contains_test_labels") is not False:
            raise InputContractError(f"source row contains TEST labels: {source} dataset={dataset}")
        if row.get("contains_evaluation_metrics") is not False:
            raise InputContractError(f"source row contains evaluation metrics: {source} dataset={dataset}")
        mapped[dataset] = row
    validate_population(mapped)
    return mapped


def _select_mini_rows(rows: Collection[Mapping[str, object]], variant: str) -> dict[str, Mapping[str, object]]:
    selected: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row.get("variant") != variant or row.get("alpha") != ALPHA:
            continue
        dataset = row.get("dataset")
        if type(dataset) is not str or not dataset:
            raise InputContractError(f"missing MiniROCKET dataset: variant={variant}")
        if dataset in selected:
            raise InputContractError(f"duplicate MiniROCKET row: variant={variant} dataset={dataset}")
        selected[dataset] = row
    validate_population(selected)
    return selected


def _database_metadata(population: Collection[str]) -> dict[str, tuple[int, int, int]]:
    try:
        verify_database_sha256(DB_PATH, FROZEN_DATABASE_SHA256)
    except DatabaseContractError as exc:
        raise InputContractError(str(exc)) from exc
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT name, train_normal_count, test_total_count, series_length FROM datasets ORDER BY name"
        ).fetchall()
    except sqlite3.Error as exc:
        raise InputContractError(f"cannot read frozen database: {DB_PATH}") from exc
    finally:
        if connection is not None:
            connection.close()
    metadata = {str(name): (n_train, n_test, target_length) for name, n_train, n_test, target_length in rows}
    if len(metadata) != len(rows) or any(
        type(n_train) is not int or type(n_test) is not int or type(target_length) is not int or target_length <= 0
        for n_train, n_test, target_length in metadata.values()
    ):
        raise InputContractError("database metadata is malformed")
    missing = sorted(set(population) - set(metadata))
    if missing:
        raise InputContractError(f"frozen dataset absent from database: {missing[0]}")
    if set(metadata) - set(population) != FROZEN_DATABASE_EXCLUSIONS:
        raise InputContractError("frozen database population mismatch")
    return {dataset: metadata[dataset] for dataset in population}


def _indices(row: Mapping[str, object], field: str, dataset: str, n_test: int) -> frozenset[int]:
    raw = row.get(field)
    if type(raw) is not list:
        raise InputContractError(f"missing or invalid {field}: dataset={dataset}")
    return validate_indices(dataset, n_test, raw)


def _validate_cache_contract(
    population: Collection[str], metadata: Mapping[str, tuple[int, int, int]]
) -> dict[str, dict[str, bool]]:
    available = {dataset: {} for dataset in population}
    for name, experiment, source, cache_root, config_hash, calibration in CACHE_CONTRACTS:
        root = Path(cache_root)
        audit_path = root.parent / "audit.json"
        if sha256_file(audit_path) != FROZEN_AUDIT_HASHES[name]:
            raise InputContractError(f"frozen audit SHA256 mismatch: {name}")
        audit = _read_json(audit_path, f"{name} audit")
        validate_prediction_boundary(audit, f"{name} audit")
        config = audit.get("config")
        if type(config) is not dict or config.get("train_calibration") != calibration:
            raise InputContractError(f"query-excluded cross-fitting contract mismatch: {name}")
        if config.get("experiment") != experiment or config.get("source") != source:
            raise InputContractError(f"frozen source config mismatch: {name}")
        if any(config.get(field) != value for field, value in FROZEN_AUDIT_CONFIGS[name].items()):
            raise InputContractError(f"frozen source seed/config mismatch: {name}")
        manifest_path = root.parent / "cache_manifest.csv"
        if sha256_file(manifest_path) != FROZEN_CACHE_MANIFEST_HASHES[name]:
            raise InputContractError(f"frozen cache manifest SHA256 mismatch: {name}")
        try:
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise InputContractError(f"cannot read cache manifest: {name}") from exc
        by_name = {row.get("dataset", ""): row for row in rows}
        if len(by_name) != len(rows) or set(by_name) != set(population):
            raise InputContractError(f"cache population mismatch: {name}")
        for dataset, row in by_name.items():
            n_train, n_test, _target_length = metadata[dataset]
            expected_cache = f"cache/{hashlib.sha256(dataset.encode('utf-8')).hexdigest()[:20]}.npz"
            if (
                row.get("experiment") != str(experiment)
                or row.get("source") != source
                or row.get("config_hash") != config_hash
                or row.get("contains_test_labels") != "False"
                or row.get("n_train") != str(n_train)
                or row.get("n_test") != str(n_test)
                or row.get("cache_file") != expected_cache
            ):
                raise InputContractError(f"cache manifest contract mismatch: {name} dataset={dataset}")
            cache_path = root / f"{hashlib.sha256(dataset.encode('utf-8')).hexdigest()[:20]}.npz"
            metadata_path = cache_path.with_suffix(".json")
            if not cache_path.is_file() or not metadata_path.is_file():
                raise InputContractError(f"unexpected missing cache: {name} dataset={dataset}")
            if sha256_file(cache_path) != row.get("cache_sha256"):
                raise InputContractError(f"cache SHA256 mismatch: {name} dataset={dataset}")
            if sha256_file(metadata_path) != row.get("metadata_sha256"):
                raise InputContractError(f"cache metadata SHA256 mismatch: {name} dataset={dataset}")
            cache_metadata = _read_json(metadata_path, f"{name} cache metadata")
            validate_prediction_boundary(cache_metadata, f"{name} cache metadata")
            if (
                cache_metadata.get("dataset") != dataset
                or cache_metadata.get("experiment") != experiment
                or cache_metadata.get("source") != source
                or cache_metadata.get("n_train") != n_train
                or cache_metadata.get("n_test") != n_test
                or cache_metadata.get("config_hash") != config_hash
            ):
                raise InputContractError(f"cache metadata contract mismatch: {name} dataset={dataset}")
            validate_cache_archive_boundary(cache_path, name, dataset)
            available[dataset][name] = True
    return available


def validate_reconstruction_archive(
    path: Path,
    dataset: str,
    *,
    n_train: int,
    n_test: int,
    expected_sha256: str | None = None,
) -> None:
    validate_cache_archive_boundary(path, "reconstruction", dataset)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise InputContractError(f"reconstruction archive SHA256 mismatch: dataset={dataset}")
    required = {
        "mask_025__train_oof",
        "mask_025__test_scores",
        "mask_025__train_by_seed",
        "mask_025__test_by_seed",
        "n_test",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise InputContractError(
                    f"reconstruction archive missing key: dataset={dataset} key={missing[0]}"
                )
            train_oof = np.asarray(archive["mask_025__train_oof"])
            test_scores = np.asarray(archive["mask_025__test_scores"])
            train_by_seed = np.asarray(archive["mask_025__train_by_seed"])
            test_by_seed = np.asarray(archive["mask_025__test_by_seed"])
            raw_n_test = np.asarray(archive["n_test"])
    except InputContractError:
        raise
    except (OSError, ValueError) as exc:
        raise InputContractError(f"cannot read reconstruction archive: dataset={dataset}") from exc
    if train_oof.shape != (n_train,) or test_scores.shape != (n_test,):
        raise InputContractError(f"reconstruction archive score length mismatch: dataset={dataset}")
    if (
        train_by_seed.ndim != 2
        or test_by_seed.ndim != 2
        or train_by_seed.shape[0] < 1
        or train_by_seed.shape[0] != test_by_seed.shape[0]
        or train_by_seed.shape[1] != n_train
        or test_by_seed.shape[1] != n_test
    ):
        raise InputContractError(f"reconstruction archive seed shape mismatch: dataset={dataset}")
    if raw_n_test.shape != (1,) or type(raw_n_test.item()) is not int or raw_n_test.item() != n_test:
        raise InputContractError(f"reconstruction archive n_test mismatch: dataset={dataset}")
    arrays = (train_oof, test_scores, train_by_seed, test_by_seed)
    if any(not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all() for array in arrays):
        raise InputContractError(f"reconstruction archive is not finite numeric data: dataset={dataset}")


def _load_reconstruction_hash_inventory(
    population: Collection[str], metadata: Mapping[str, tuple[int, int, int]]
) -> Mapping[str, str]:
    if sha256_file(RECON_ARCHIVE_HASH_INVENTORY) != FROZEN_RECON_ARCHIVE_HASH_INVENTORY_SHA256:
        raise InputContractError("frozen reconstruction archive inventory SHA256 mismatch")
    inventory = _read_json(RECON_ARCHIVE_HASH_INVENTORY, "reconstruction archive inventory")
    archives = inventory.get("archives")
    expected_names = {
        f"{hashlib.sha256(dataset.encode('utf-8')).hexdigest()[:20]}.npz"
        for dataset in population
        if metadata[dataset][0] >= 100
    }
    if (
        set(inventory) != {"algorithm", "archives", "cache_root", "source_audit_sha256", "version"}
        or inventory.get("algorithm") != "sha256"
        or inventory.get("version") != 1
        or inventory.get("cache_root") != str(RECON_CACHE_ROOT)
        or inventory.get("source_audit_sha256") != FROZEN_AUDIT_HASHES["reconstruction"]
        or type(archives) is not dict
        or set(archives) != expected_names
        or any(type(value) is not str or len(value) != 64 for value in archives.values())
    ):
        raise InputContractError("frozen reconstruction archive inventory contract mismatch")
    return MappingProxyType(dict(archives))


def _validate_reconstruction_cache(
    population: Collection[str], metadata: Mapping[str, tuple[int, int, int]], available: dict[str, dict[str, bool]]
) -> None:
    audit_path = RECON_CACHE_ROOT.parent / "audit.json"
    if sha256_file(audit_path) != FROZEN_AUDIT_HASHES["reconstruction"]:
        raise InputContractError("frozen audit SHA256 mismatch: reconstruction")
    audit = _read_json(audit_path, "reconstruction audit")
    validate_prediction_boundary(audit, "reconstruction audit")
    if audit.get("population_rule") != "train_normal_count >= 100" or audit.get("datasets") != 251:
        raise InputContractError("frozen reconstruction availability contract mismatch")
    inventory = _load_reconstruction_hash_inventory(population, metadata)
    for dataset in population:
        n_train, _n_test, _target_length = metadata[dataset]
        cache_path = RECON_CACHE_ROOT / f"{hashlib.sha256(dataset.encode('utf-8')).hexdigest()[:20]}.npz"
        should_be_available = n_train >= 100
        if should_be_available and not cache_path.is_file():
            raise InputContractError(f"unexpected missing cache: reconstruction dataset={dataset}")
        if not should_be_available and cache_path.exists():
            raise InputContractError(f"unexpected reconstruction cache: dataset={dataset}")
        if should_be_available:
            validate_reconstruction_archive(
                cache_path,
                dataset,
                n_train=n_train,
                n_test=metadata[dataset][1],
                expected_sha256=inventory[cache_path.name],
            )
        available[dataset]["reconstruction"] = should_be_available


def _write_audit(output_root: Path, recovery: FrozenRecoveryInputs) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    audit = {
        "experiment": 266,
        "experiment_id": "experiment_266_outside_lane_fn_recovery_shadow",
        "datasets": len(recovery.population),
        "lane_totals": dict(recovery.lane_totals),
        "input_sha256": dict(recovery.input_sha256),
        "database_sha256": FROZEN_DATABASE_SHA256,
        "immutable_hash_pins": {
            "database": FROZEN_DATABASE_SHA256,
            "prediction_artifacts": dict(FROZEN_PREDICTION_HASHES),
            "prediction_manifests": dict(FROZEN_PREDICTION_MANIFEST_HASHES),
            "prediction_audits": dict(FROZEN_PREDICTION_AUDIT_HASHES),
            "cache_manifests": dict(FROZEN_CACHE_MANIFEST_HASHES),
            "cache_audits": dict(FROZEN_AUDIT_HASHES),
            "reconstruction_archive_inventory": FROZEN_RECON_ARCHIVE_HASH_INVENTORY_SHA256,
        },
        "source_paths": {
            "exp253": str(EXP253_PREDICTION),
            "exp264": str(EXP264_PREDICTION),
            "minirocket": str(MINIROCKET_PREDICTION),
        },
        "reconstruction_unavailable": sum(
            "reconstruction" not in item.available_groups
            for item in recovery.datasets.values()
        ),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
        "deployment_eligible": False,
    }
    (output_root / "00_contract_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "dataset", "n_train", "n_test", "target_length", "hard_count", "shadow_count", "priority_count",
        "standard_count", "kernel_available", "statistical_available", "structural_available",
        "change_point_available", "reconstruction_available",
    ]
    with (output_root / "01_source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in recovery.population:
            item = recovery.datasets[dataset]
            writer.writerow({
                "dataset": dataset,
                "n_train": item.n_train,
                "n_test": item.n_test,
                "target_length": item.target_length,
                "hard_count": len(item.hard),
                "shadow_count": len(item.shadow),
                "priority_count": len(item.priority),
                "standard_count": len(item.standard),
                **{f"{group}_available": group in item.available_groups for group in item.evidence},
            })


def load_frozen_inputs(
    *,
    output_root: Path = OUTPUT_ROOT,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> FrozenRecoveryInputs:
    """Load and audit Exp266's immutable, label-free source population."""

    exp253_rows = _validate_prediction_artifact("exp253", EXP253_PREDICTION, EXPECTED_DATASETS)
    exp264_rows = _validate_prediction_artifact("exp264", EXP264_PREDICTION, EXPECTED_DATASETS)
    mini_rows = _validate_prediction_artifact("minirocket", MINIROCKET_PREDICTION, 17872)
    exp253 = _by_dataset(exp253_rows, "exp253")
    exp264 = _by_dataset(exp264_rows, "exp264")
    if set(exp253) != set(exp264):
        raise InputContractError("Exp253 and Exp264 population mismatch")
    population = tuple(sorted(exp253))
    metadata = _database_metadata(population)
    mini = {variant: _select_mini_rows(mini_rows, variant) for variant in MINI_VARIANTS}
    if any(set(rows) != set(exp253) for rows in mini.values()):
        raise InputContractError("MiniROCKET frozen population mismatch")
    kernel = group_kernel_candidates(mini[MINI_VARIANTS[0]], mini[MINI_VARIANTS[1]])
    cache_available = _validate_cache_contract(population, metadata)
    _validate_reconstruction_cache(population, metadata, cache_available)

    datasets: dict[str, FrozenDatasetInputs] = {}
    for dataset in population:
        baseline = exp253[dataset]
        integration = exp264[dataset]
        n_train, n_test, target_length = metadata[dataset]
        if integration.get("n_train") != n_train or integration.get("n_test") != n_test:
            raise InputContractError(f"Exp264 database metadata mismatch: dataset={dataset}")
        if baseline.get("n_test") != n_test:
            raise InputContractError(f"Exp253 database metadata mismatch: dataset={dataset}")
        source_indices = integration.get("source_indices")
        if type(source_indices) is not dict or set(source_indices) != {
            "statistical", "structural", "change_point", "reconstruction"
        }:
            raise InputContractError(f"invalid Exp264 evidence schema: dataset={dataset}")
        evidence = {
            "kernel": validate_indices(dataset, n_test, kernel[dataset]),
            **{
                group: validate_indices(dataset, n_test, source_indices[group])
                for group in ("statistical", "structural", "change_point", "reconstruction")
                if type(source_indices[group]) is list
            },
        }
        if len(evidence) != 5:
            raise InputContractError(f"invalid Exp264 evidence indices: dataset={dataset}")
        datasets[dataset] = FrozenDatasetInputs(
            dataset=dataset,
            n_train=n_train,
            n_test=n_test,
            hard=_indices(baseline, "final_hard_indices", dataset, n_test),
            shadow=_indices(integration, "promoted_indices", dataset, n_test),
            priority=_indices(integration, "final_priority_indices", dataset, n_test),
            standard=_indices(integration, "final_standard_indices", dataset, n_test),
            evidence=MappingProxyType(evidence),
            available_groups=frozenset(
                group for group, is_available in cache_available[dataset].items() if is_available
            ),
            quality_flags=_indices(integration, "quality_flag_indices", dataset, n_test),
            target_length=target_length,
        )
        if progress_callback is not None:
            progress_callback(dataset, len(datasets), len(population))
    totals = {
        lane: sum(len(getattr(item, lane)) for item in datasets.values())
        for lane in ("hard", "shadow", "priority", "standard")
    }
    totals["union"] = sum(
        len(item.hard | item.shadow | item.priority | item.standard)
        for item in datasets.values()
    )
    if totals != EXPECTED_LANE_TOTALS:
        raise InputContractError(f"frozen lane totals mismatch: {totals}")
    recovery = FrozenRecoveryInputs(
        population=population,
        datasets=MappingProxyType(datasets),
        input_sha256=MappingProxyType({
            "exp253": FROZEN_PREDICTION_HASHES["exp253"],
            "exp264": FROZEN_PREDICTION_HASHES["exp264"],
            "minirocket": FROZEN_PREDICTION_HASHES["minirocket"],
        }),
        lane_totals=MappingProxyType(totals),
    )
    _write_audit(Path(output_root), recovery)
    return recovery
