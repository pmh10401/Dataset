from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import sqlite3
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from experiments.exp177_185.core import canonical_features, crossfit_scores
from experiments.exp224.core import ALPHAS, SEED, train_fold_candidate_counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_metadata() -> dict[str, str]:
    def version(distribution: str) -> str:
        try:
            return importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scikit_learn": version("scikit-learn"),
        "torch": version("torch"),
    }


def _align_series(values: np.ndarray, target_length: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(series) == 0 or not np.all(np.isfinite(series)):
        raise ValueError("TRAIN values must be non-empty and finite")
    if len(series) == target_length:
        return series.copy()
    if len(series) == 1:
        return np.full(target_length, series[0], dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, len(series), dtype=float)
    new_x = np.linspace(0.0, 1.0, target_length, dtype=float)
    return np.interp(new_x, old_x, series).astype(np.float32)


def _sanitize_train_series(values: np.ndarray) -> np.ndarray:
    """Match the frozen historical scorer's TRAIN-only interpolation rule."""
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(series) == 0:
        raise ValueError("TRAIN values must be non-empty")
    finite = np.isfinite(series)
    if finite.all():
        return series.copy()
    if not finite.any():
        return np.zeros_like(series, dtype=np.float32)
    indices = np.arange(len(series))
    cleaned = series.copy()
    cleaned[~finite] = np.interp(
        indices[~finite], indices[finite], series[finite]
    )
    return cleaned.astype(np.float32)


def load_train_profile(
    db_path: Path,
    profile_id: str,
    target_length: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and align only TRAIN values for one frozen profile."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT i.values_blob
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = 'TRAIN'
            ORDER BY i.instance_index
            """,
            (str(profile_id),),
        ).fetchall()
    if not rows:
        raise ValueError(f"TRAIN profile not found: {profile_id}")
    raw = [
        _sanitize_train_series(np.frombuffer(row[0], dtype=np.float32))
        for row in rows
    ]
    if target_length is None:
        target_length = int(np.clip(np.median([len(values) for values in raw]), 8, 2048))
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    train = np.vstack([_align_series(values, target_length) for values in raw])
    record = {
        "dataset_name": str(profile_id),
        "family": "__family_neutral__",
        "metadata_len": target_length,
        "actual_len_median": float(np.median([len(values) for values in raw])),
        "actual_len_max": float(max(len(values) for values in raw)),
        "test_actual_len_median": float(target_length),
        "train_series": raw,
        "test_series": [],
        "train_target_len": target_length,
    }
    return train.astype(np.float32), record


def compute_profile_artifact(
    profile_id: str,
    db_path: Path,
) -> dict[str, np.ndarray]:
    """Recompute both frozen TRAIN OOF sources and one deterministic probe."""
    from experiments.exp153_157 import exp153_157_bridge as bridge

    train, record = load_train_profile(db_path, profile_id)
    if len(train) < 5:
        raise ValueError(f"Exp224 requires at least five TRAIN rows: {profile_id}")
    probe = np.median(train, axis=0).astype(np.float32)
    temporal = bridge.exp152.crossfit_source(
        "rocket_exp40", train, probe.reshape(1, -1), record
    )
    temporal_oof = np.asarray(temporal["train_oof_scores"], dtype=np.float64)
    temporal_probe = np.asarray(temporal["test_scores"], dtype=np.float64)

    train_features = canonical_features(train)
    probe_features = canonical_features(probe.reshape(1, -1))
    canonical_oof, canonical_probe = crossfit_scores(
        train_features,
        probe_features,
        backend="knn_local_gap",
        seed=SEED,
    )
    counts, trials = train_fold_candidate_counts(temporal_oof, canonical_oof)
    metadata = {
        "profile_id": str(profile_id),
        "n_train": int(len(train)),
        "train_target_length": int(train.shape[1]),
        "seed": SEED,
        "alphas": list(ALPHAS),
        "temporal_source": "rocket_exp40/rocket_256_knn3_local_gap",
        "canonical_source": "canonical_features/knn_local_gap",
        "train_false_counts": {str(alpha): counts[alpha] for alpha in ALPHAS},
        "train_trials": trials,
        "runtime": _runtime_metadata(),
    }
    return {
        "canonical_oof": np.asarray(canonical_oof, dtype=np.float64),
        "canonical_probe": np.asarray(canonical_probe, dtype=np.float64),
        "metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
        "probe": probe,
        "temporal_oof": temporal_oof,
        "temporal_probe": temporal_probe,
    }


def write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Write sorted NPY members with stable ZIP metadata and return SHA-256."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.save(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue())
    payload = buffer.getvalue()
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    return hashlib.sha256(payload).hexdigest()


def _build_one(arguments: tuple[str, str, str]) -> dict[str, object]:
    profile_id, db_path, artifacts_dir = arguments
    filename = f"{hashlib.sha256(profile_id.encode('utf-8')).hexdigest()[:24]}.npz"
    artifact_path = Path(artifacts_dir) / filename
    if artifact_path.is_file():
        try:
            return _manifest_row_from_artifact(profile_id, artifact_path)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    arrays = compute_profile_artifact(profile_id, Path(db_path))
    artifact_hash = write_deterministic_npz(artifact_path, arrays)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    return {
        "profile_id": profile_id,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_hash,
        "n_train": metadata["n_train"],
        "train_target_length": metadata["train_target_length"],
        "train_trials": metadata["train_trials"],
        "train_false_counts_json": json.dumps(
            metadata["train_false_counts"], sort_keys=True, separators=(",", ":")
        ),
    }


def _manifest_row_from_artifact(
    profile_id: str,
    artifact_path: Path,
) -> dict[str, object]:
    required = {
        "canonical_oof",
        "canonical_probe",
        "metadata_json",
        "probe",
        "temporal_oof",
        "temporal_probe",
    }
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("artifact members differ from frozen schema")
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if metadata.get("profile_id") != profile_id:
        raise ValueError("artifact profile_id mismatch")
    if metadata.get("runtime") != _runtime_metadata():
        raise ValueError("artifact runtime mismatch")
    n_train = int(metadata["n_train"])
    if len(arrays["temporal_oof"]) != n_train or len(arrays["canonical_oof"]) != n_train:
        raise ValueError("artifact OOF length mismatch")
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.dtype.kind not in "US" and not np.all(np.isfinite(array)):
            raise ValueError(f"non-finite artifact array: {name}")
    counts, trials = train_fold_candidate_counts(
        arrays["temporal_oof"], arrays["canonical_oof"]
    )
    expected_counts = {
        str(alpha): int(metadata["train_false_counts"][str(alpha)])
        for alpha in ALPHAS
    }
    actual_counts = {str(alpha): counts[alpha] for alpha in ALPHAS}
    if actual_counts != expected_counts or trials != int(metadata["train_trials"]):
        raise ValueError("artifact TRAIN risk metadata mismatch")
    return {
        "profile_id": profile_id,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "n_train": n_train,
        "train_target_length": int(metadata["train_target_length"]),
        "train_trials": trials,
        "train_false_counts_json": json.dumps(
            actual_counts, sort_keys=True, separators=(",", ":")
        ),
    }


def build_artifacts(
    profile_ids: Iterable[str],
    db_path: Path,
    output_root: Path,
    workers: int,
) -> list[dict[str, object]]:
    """Build deterministic artifacts and a stable CSV manifest."""
    profiles = sorted({str(profile_id) for profile_id in profile_ids})
    if not profiles:
        raise ValueError("at least one profile is required")
    root = Path(output_root)
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    arguments = [(profile, str(Path(db_path)), str(artifacts_dir)) for profile in profiles]
    if int(workers) == 1:
        rows = [_build_one(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            rows = list(executor.map(_build_one, arguments))
    rows = sorted(rows, key=lambda row: str(row["profile_id"]))
    manifest_path = root / "artifact_manifest.csv"
    fieldnames = [
        "profile_id",
        "artifact_path",
        "artifact_sha256",
        "n_train",
        "train_target_length",
        "train_trials",
        "train_false_counts_json",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
