"""Repository integration layer for Exp153--157.

This module is the only place where the new experiments import the historical
research code.  The public mathematical policy remains in ``exp153_157_core``.
The local database path is intentionally unchanged because the user's 1.6 GB
SQLite database already lives there.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP151_DIR = REPO_ROOT / "experiments" / "exp151_152"
for import_path in (HERE, EXP151_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import exp153_157_core as core
import run_experiment_151_152_virtual_run_conformal_policy_v2 as exp152
import virtual_run_policy_core as exp152_core

DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
BASELINE_PATH = DATA_DIR / "experiment_137_operational_triage_results.csv"
SOURCE_NAMES = tuple(exp152.SOURCE_CONFIGS)
LEGACY_BLOCK_B_RATE = float(exp152.BLOCK_B_RATE)
LEGACY_BLOCK_C_RATE = float(exp152.BLOCK_C_RATE)


def evaluation_dataset_names(dataset_limit: int | None = None) -> list[str]:
    return list(exp152.evaluation_dataset_names(dataset_limit))


def load_dataset_series_only(dataset_name: str):
    return exp152.load_dataset_series_only(dataset_name, DB_PATH)


def load_test_labels(dataset_name: str) -> np.ndarray:
    return exp152.load_test_labels(dataset_name, DB_PATH)


def cache_file_name(dataset_name: str) -> str:
    digest = hashlib.sha1(dataset_name.encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def _block_b_score_pair(x_fit: np.ndarray, x_query: np.ndarray):
    """Return Block-B TRAIN/query scores without using labels."""

    fit_z = exp152.z_normalize(x_fit).astype(np.float32)
    query_z = exp152.z_normalize(x_query).astype(np.float32)
    _, _, fit_features, query_features = exp152.exp131.score_block_b(fit_z, query_z)
    fit_scores, query_scores = exp152.density_knn_score_pair(
        fit_features,
        query_features,
        exp152.BLOCK_B_NEIGHBORS,
        "local_gap",
    )
    return np.asarray(fit_scores, dtype=float), np.asarray(query_scores, dtype=float)


def _block_c_query_scores(x_fit: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Return raw Block-C query scores using the frozen Exp135 implementation."""

    fit_z = exp152.z_normalize(x_fit).astype(np.float32)
    query_z = exp152.z_normalize(x_query).astype(np.float32)
    _, query_scores, *_ = exp152.exp135.block_c_candidates(fit_z, query_z)
    return np.asarray(query_scores, dtype=float)


def leave_one_out_historical_countcap_flags(scores: np.ndarray, rate: float) -> np.ndarray:
    """Apply the repository's exact historical count-cap implementation LOO."""

    values = np.asarray(scores, dtype=float)
    flags = np.zeros(len(values), dtype=bool)
    for index, score in enumerate(values):
        reference = np.delete(values, index)
        threshold, _, _ = exp152.count_cap_threshold(reference, float(rate))
        flags[index] = bool(np.isfinite(score) and score > threshold)
    return flags


def crossfit_confirmation_scores(x_train: np.ndarray, x_test: np.ndarray) -> Mapping[str, np.ndarray]:
    """Create OOF TRAIN and full-fit TEST scores for Block-B and Block-C.

    Exp156 needs raw confirmation scores rather than only the historical
    thresholded indices.  Each TRAIN score is out-of-fold.  TEST scores are
    produced only after all TRAIN OOF scores are available.
    """

    n_train = len(x_train)
    if n_train < 5:
        return {
            "block_b_train_oof": np.asarray([], dtype=float),
            "block_b_test": np.asarray([], dtype=float),
            "block_c_train_oof": np.asarray([], dtype=float),
            "block_c_test": np.asarray([], dtype=float),
        }

    block_b_oof = np.full(n_train, np.nan, dtype=float)
    block_c_oof = np.full(n_train, np.nan, dtype=float)
    for fit_indices, query_indices in exp152_core.fold_plan(n_train, seed=core.SEED):
        _, block_b_query = _block_b_score_pair(
            x_train[fit_indices], x_train[query_indices]
        )
        block_c_query = _block_c_query_scores(
            x_train[fit_indices], x_train[query_indices]
        )
        block_b_oof[query_indices] = block_b_query
        block_c_oof[query_indices] = block_c_query

    _, block_b_test = _block_b_score_pair(x_train, x_test)
    block_c_test = _block_c_query_scores(x_train, x_test)
    if not np.all(np.isfinite(block_b_oof)):
        raise ValueError("incomplete Block-B OOF scores")
    if not np.all(np.isfinite(block_c_oof)):
        raise ValueError("incomplete Block-C OOF scores")
    return {
        "block_b_train_oof": block_b_oof,
        "block_b_test": block_b_test,
        "block_c_train_oof": block_c_oof,
        "block_c_test": block_c_test,
    }


def build_dataset_cache(dataset_name: str, cache_dir: Path) -> dict[str, object]:
    """Compute and persist every label-free score needed by Exp153--157."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, record = load_dataset_series_only(dataset_name)
    n_train, n_test = len(x_train), len(x_test)
    abstain = n_train < 5

    arrays: dict[str, np.ndarray] = {}
    if not abstain:
        for source_name in SOURCE_NAMES:
            result = exp152.crossfit_source(
                source_name,
                x_train,
                x_test,
                record,
            )
            arrays[f"{source_name}__train_oof"] = np.asarray(
                result["train_oof_scores"], dtype=float
            )
            arrays[f"{source_name}__test"] = np.asarray(
                result["test_scores"], dtype=float
            )
        arrays.update(crossfit_confirmation_scores(x_train, x_test))
        legacy = exp152.recompute_independent_evidence(x_train, x_test)
        arrays["legacy_block_b_indices"] = np.asarray(
            sorted(legacy["block_b_indices"]), dtype=np.int64
        )
        arrays["legacy_block_c_indices"] = np.asarray(
            sorted(legacy["block_c_indices"]), dtype=np.int64
        )
    else:
        for source_name in SOURCE_NAMES:
            arrays[f"{source_name}__train_oof"] = np.asarray([], dtype=float)
            arrays[f"{source_name}__test"] = np.asarray([], dtype=float)
        arrays.update(
            {
                "block_b_train_oof": np.asarray([], dtype=float),
                "block_b_test": np.asarray([], dtype=float),
                "block_c_train_oof": np.asarray([], dtype=float),
                "block_c_test": np.asarray([], dtype=float),
                "legacy_block_b_indices": np.asarray([], dtype=np.int64),
                "legacy_block_c_indices": np.asarray([], dtype=np.int64),
            }
        )

    metadata = {
        "dataset_name": dataset_name,
        "n_train": n_train,
        "n_test": n_test,
        "train_target_len": int(record["train_target_len"]),
        "abstain": int(abstain),
        "source_names": list(SOURCE_NAMES),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    path = cache_dir / cache_file_name(dataset_name)
    np.savez_compressed(path, **arrays)
    return {
        **metadata,
        "cache_file": path.name,
        "cache_sha256": core.sha256_file(path),
    }


def load_cache_entry(cache_dir: Path, manifest_row: Mapping[str, object]) -> dict[str, object]:
    path = Path(cache_dir) / str(manifest_row["cache_file"])
    expected = str(manifest_row.get("cache_sha256", ""))
    if expected:
        core.verify_frozen_file(path, expected)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = json.loads(str(result.pop("metadata_json").item()))
    result["metadata"] = metadata
    return result



def validate_b2_contract(dataset_names):
    """Validate the same B2 configuration/seed/coverage contract used by Exp152."""

    manifest_path, selected = exp152.load_and_validate_b2_contract(dataset_names)
    return manifest_path, selected


def resolve_exp152_prediction_paths() -> tuple[Path, Path]:
    """Return frozen Exp152 prediction and manifest paths from results or outputs."""

    result_dir_candidates = [
        REPO_ROOT / "results/exp151_152_virtual_run_conformal_v2",
        REPO_ROOT / "outputs/exp151_152_virtual_run_conformal_v2",
    ]
    for directory in result_dir_candidates:
        prediction = directory / "04_exp152_b2_source_predictions.jsonl"
        manifest = directory / "05_prediction_manifest.json"
        if prediction.exists() and manifest.exists():
            return prediction, manifest
    raise FileNotFoundError(
        "Exp152 frozen prediction artifacts not found under results/ or outputs/."
    )
