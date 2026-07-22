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
from experiments.exp251_sktime_statistical_features.config import MINIMUM_TRAIN, SEED, SOURCE
from experiments.exp251_sktime_statistical_features.features import (
    FEATURE_DIMENSIONS,
    extract_feature_blocks,
    fit_train_sanitizer,
)
from experiments.exp251_sktime_statistical_features.scoring import score_feature_blocks


EXP236_PATH = Path("/Users/minho/Documents/Dataset/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl")


def dataset_names() -> tuple[str, ...]:
    rows = [json.loads(line) for line in EXP236_PATH.read_text().splitlines()]
    names = tuple(sorted(str(row["dataset"]) for row in rows))
    if len(names) != 1117 or len(set(names)) != 1117:
        raise ValueError("frozen Exp236 population must contain 1,117 unique datasets")
    return names


def source_config() -> dict[str, object]:
    return {
        "experiment": 251,
        "source": SOURCE,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "transformers": {
            "catch22_22": {"catch24": False, "replace_nans": True},
            "summary_9": {"defaults": True},
            "hurst_3": {"method": "rs", "min_lag": 2, "max_lag": "train_length_bounded"},
            "signature_6": {"degree": 2, "use_index": True},
        },
        "seed": SEED,
        "minimum_train": MINIMUM_TRAIN,
        "score": "knn3_local_gap",
        "train_calibration": "five_fold_oof",
        "contains_test_labels": False,
        "package_versions": {
            "sktime": importlib.metadata.version("sktime"),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }


def _stem(dataset: str) -> str:
    return hashlib.sha256(dataset.encode()).hexdigest()[:20]


def cache_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    base = Path(root) / "cache" / _stem(dataset)
    return base.with_suffix(".npz"), base.with_suffix(".json")


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


def build_cache_unit(root_text: str, dataset: str, reset: bool = False) -> dict[str, object]:
    root = Path(root_text)
    archive_path, metadata_path = cache_paths(root, dataset)
    config = source_config()
    config_hash = stable_json_hash(config)
    if not reset and archive_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_hash") == config_hash and sha256_file(archive_path) == metadata.get("cache_sha256"):
            return metadata

    bundle = load_bundle(dataset)
    abstained = len(bundle.train_normal) < MINIMUM_TRAIN
    arrays: dict[str, np.ndarray] = {}
    replacements: dict[str, list[float]] = {}
    if abstained:
        for variant in FEATURE_DIMENSIONS:
            arrays[f"{variant}__train_oof"] = np.asarray([], dtype=float)
            arrays[f"{variant}__test_scores"] = np.asarray([], dtype=float)
    else:
        raw_train = extract_feature_blocks(bundle.train_normal, target_length=bundle.target_length)
        raw_test = extract_feature_blocks(bundle.test_values, target_length=bundle.target_length)
        sanitizer = fit_train_sanitizer(raw_train)
        train_blocks = sanitizer.transform(raw_train)
        test_blocks = sanitizer.transform(raw_test)
        replacements = {name: values.tolist() for name, values in sanitizer.replacements.items()}
        scores = score_feature_blocks(train_blocks, test_blocks, seed=SEED)
        for variant, result in scores.items():
            arrays[f"{variant}__train_oof"] = result.train_oof
            arrays[f"{variant}__test_scores"] = result.test_scores

    _atomic_npz(archive_path, arrays)
    metadata = {
        "dataset": dataset,
        "experiment": 251,
        "source": SOURCE,
        "n_train": len(bundle.train_normal),
        "n_test": len(bundle.test_values),
        "target_length": bundle.target_length,
        "abstained": abstained,
        "abstention_reason": f"n_train_below_{MINIMUM_TRAIN}" if abstained else None,
        "train_feature_replacements": replacements,
        "config": config,
        "config_hash": config_hash,
        "contains_test_labels": False,
        "cache_file": str(archive_path.relative_to(root)),
        "cache_sha256": sha256_file(archive_path),
    }
    _atomic_json(metadata_path, metadata)
    metadata["metadata_sha256"] = sha256_file(metadata_path)
    return metadata


def cache_all(
    root: Path,
    *,
    workers: int,
    names: tuple[str, ...],
    reset: bool = False,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(build_cache_unit, str(root), dataset, reset): dataset
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
    root.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key not in {"config", "train_feature_replacements"}})
    with (root / "cache_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in fields}
            for row in sorted(rows, key=lambda item: str(item["dataset"]))
        )
    error_path = root / "cache_errors.csv"
    if errors:
        with error_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("dataset", "error"))
            writer.writeheader()
            writer.writerows(errors)
        raise RuntimeError(f"Exp251 cache errors: {len(errors)}")
    error_path.unlink(missing_ok=True)
    if len(rows) != len(names):
        raise RuntimeError("Exp251 cache coverage mismatch")
    return rows
