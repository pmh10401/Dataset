from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import sha256_file
from experiments.exp239_248_diverse_score_sources.foundation import load_bundle, stable_json_hash
from experiments.exp249_250_official_mini_multirocket.config import (
    MINIMUM_TRAIN,
    MODEL_CONFIGS,
    MAX_TRAIN_REFERENCE,
    QUERY_CHUNK_SIZE,
    SEEDS,
)
from experiments.exp249_250_official_mini_multirocket.scoring import score_configuration
from experiments.exp249_250_official_mini_multirocket.transforms import expected_feature_dimension


EXP236_PATH = Path("/Users/minho/Documents/Dataset/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl")


def dataset_names() -> tuple[str, ...]:
    rows = [json.loads(line) for line in EXP236_PATH.read_text().splitlines()]
    names = tuple(sorted(str(row["dataset"]) for row in rows))
    if len(names) != 1117 or len(set(names)) != 1117:
        raise ValueError("frozen Exp236 population must contain 1,117 unique datasets")
    return names


def source_root(root: Path, experiment: int) -> Path:
    config = MODEL_CONFIGS[int(experiment)]
    return Path(root) / f"experiment_{experiment}_{config.source}"


def _stem(dataset: str) -> str:
    return hashlib.sha256(dataset.encode()).hexdigest()[:20]


def cache_paths(root: Path, experiment: int, dataset: str) -> tuple[Path, Path]:
    base = source_root(root, experiment) / "cache" / _stem(dataset)
    return base.with_suffix(".npz"), base.with_suffix(".json")


def source_config(experiment: int) -> dict[str, object]:
    config = MODEL_CONFIGS[int(experiment)]
    return {
        "experiment": config.experiment,
        "source": config.source,
        "transformer": config.transformer,
        "requested_kernels": list(config.requested_kernels),
        "actual_feature_dimensions": {
            str(value): expected_feature_dimension(config, value)
            for value in config.requested_kernels
        },
        "max_dilations_per_kernel": config.max_dilations_per_kernel,
        "n_features_per_kernel": config.n_features_per_kernel,
        "normalise": config.normalise,
        "seeds": list(SEEDS),
        "inner_n_jobs": 1,
        "minimum_train": MINIMUM_TRAIN,
        "score": "knn3_local_gap",
        "train_calibration": "five_fold_transformer_crossfit",
        "maximum_train_reference": MAX_TRAIN_REFERENCE,
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "contains_test_labels": False,
        "package_versions": {
            "aeon": importlib.metadata.version("aeon"),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def build_cache_unit(
    root_text: str, experiment: int, dataset: str, reset: bool = False
) -> dict[str, object]:
    root = Path(root_text)
    archive_path, metadata_path = cache_paths(root, experiment, dataset)
    frozen_config = source_config(experiment)
    config_hash = stable_json_hash(frozen_config)
    if not reset and archive_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("config_hash") == config_hash
            and sha256_file(archive_path) == metadata.get("cache_sha256")
        ):
            return metadata

    bundle = load_bundle(dataset)
    config = MODEL_CONFIGS[int(experiment)]
    abstained = len(bundle.train_normal) < MINIMUM_TRAIN
    arrays: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    if not abstained:
        for requested in config.requested_kernels:
            train_by_seed = []
            test_by_seed = []
            for seed in SEEDS:
                result = score_configuration(
                    bundle.train_normal,
                    bundle.test_values,
                    config=config,
                    requested_kernels=requested,
                    seed=seed,
                    expected_dimension=expected_feature_dimension(config, requested),
                )
                train_by_seed.append(result.train_oof)
                test_by_seed.append(result.test_scores)
                dimensions[str(requested)] = result.feature_dimension
            arrays[f"kernels_{requested}__train_by_seed"] = np.asarray(train_by_seed)
            arrays[f"kernels_{requested}__test_by_seed"] = np.asarray(test_by_seed)
    else:
        for requested in config.requested_kernels:
            arrays[f"kernels_{requested}__train_by_seed"] = np.empty((len(SEEDS), 0))
            arrays[f"kernels_{requested}__test_by_seed"] = np.empty((len(SEEDS), 0))
            dimensions[str(requested)] = expected_feature_dimension(config, requested)

    _atomic_npz(archive_path, arrays)
    metadata = {
        "dataset": dataset,
        "experiment": int(experiment),
        "source": config.source,
        "n_train": len(bundle.train_normal),
        "n_test": len(bundle.test_values),
        "target_length": bundle.target_length,
        "abstained": abstained,
        "abstention_reason": f"n_train_below_{MINIMUM_TRAIN}" if abstained else None,
        "feature_dimensions": dimensions,
        "config": frozen_config,
        "config_hash": config_hash,
        "contains_test_labels": False,
        "cache_file": str(archive_path.relative_to(source_root(root, experiment))),
        "cache_sha256": sha256_file(archive_path),
    }
    _atomic_json(metadata_path, metadata)
    metadata["metadata_sha256"] = sha256_file(metadata_path)
    return metadata


def cache_experiment(
    root: Path,
    experiment: int,
    *,
    workers: int,
    names: tuple[str, ...],
    reset: bool = False,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    if workers == 1:
        for done, dataset in enumerate(names, 1):
            try:
                rows.append(build_cache_unit(str(root), experiment, dataset, reset))
            except Exception as error:
                errors.append({"dataset": dataset, "error": repr(error)})
            if progress:
                progress(done, len(names), dataset, len(errors))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(build_cache_unit, str(root), experiment, dataset, reset): dataset
                for dataset in names
            }
            for done, future in enumerate(as_completed(futures), 1):
                dataset = futures[future]
                try:
                    rows.append(future.result())
                except Exception as error:
                    errors.append({"dataset": dataset, "error": repr(error)})
                if progress:
                    progress(done, len(names), dataset, len(errors))

    base = source_root(root, experiment)
    base.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key not in {"config"}})
    with (base / "cache_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in fields}
            for row in sorted(rows, key=lambda item: str(item["dataset"]))
        )
    error_path = base / "cache_errors.csv"
    if errors:
        with error_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("dataset", "error"))
            writer.writeheader()
            writer.writerows(errors)
        raise RuntimeError(f"cache errors: experiment={experiment} count={len(errors)}")
    error_path.unlink(missing_ok=True)
    if len(rows) != len(names):
        raise RuntimeError("cache coverage mismatch")
    return rows
