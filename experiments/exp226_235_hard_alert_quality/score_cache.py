from __future__ import annotations

import ast
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.contracts import ScoreBundle
from experiments.exp226_235_hard_alert_quality.database import sha256_file


EXP153_ROOT = Path(
    "/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements"
)
EXP153_MANIFEST = EXP153_ROOT / "01_score_cache_manifest.csv"
EXP153_MANIFEST_SHA256 = "bf73796c046a6102fce53d3b6fc606afa1ed0a5161ebb6e92d94698ab56a68a0"
CANONICAL_ROOT = Path("/Users/minho/Documents/Dataset/exp177_185_unattended")
CANONICAL_MANIFEST = CANONICAL_ROOT / "exp180_canonical_score_manifest.csv"
CANONICAL_MANIFEST_SHA256 = "36810c14a5cca10d9758aa55144bcd785e1bae4d12278039cac83af2b96901ce"
SOURCE_NAMES = {
    "rocket_exp40": "temporal_random_kernel",
    "exp55_best": "frequency_image",
    "exp56_best": "texture_shape",
    "exp84": "auxiliary_texture_shape",
}


def _validated_vectors(values: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, raw in sorted(values.items()):
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"{kind} vector must be finite and one-dimensional: {name}")
        output[str(name)] = array
    if not output:
        raise ValueError(f"{kind} vectors must not be empty")
    return output


def write_score_bundle(
    cache_path: Path, manifest_path: Path, bundle: ScoreBundle
) -> tuple[Path, Path]:
    cache = Path(cache_path)
    manifest = Path(manifest_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    train = _validated_vectors(dict(bundle.train_oof), "TRAIN OOF")
    test = _validated_vectors(dict(bundle.test_scores), "TEST score")
    if set(train) != set(test):
        raise ValueError("TRAIN and TEST score source names must match")
    arrays = {f"train__{name}": values for name, values in train.items()}
    arrays.update({f"test__{name}": values for name, values in test.items()})
    with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as handle:
        temporary_cache = Path(handle.name)
    try:
        np.savez_compressed(temporary_cache, **arrays)
        os.replace(temporary_cache, cache)
    finally:
        temporary_cache.unlink(missing_ok=True)
    payload = {
        "schema": "exp226_235_score_bundle_v1",
        "dataset": bundle.dataset,
        "sources": sorted(train),
        "train_lengths": {name: len(values) for name, values in train.items()},
        "test_lengths": {name: len(values) for name, values in test.items()},
        "provenance": dict(bundle.provenance),
        "cache_sha256": sha256_file(cache),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with tempfile.NamedTemporaryFile(
        dir=manifest.parent, mode="w", encoding="utf-8", delete=False
    ) as handle:
        temporary_manifest = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_manifest, manifest)
    return cache, manifest


def load_verified_score_bundle(cache_path: Path, manifest_path: Path) -> ScoreBundle:
    cache = Path(cache_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual_hash = sha256_file(cache)
    if actual_hash != manifest.get("cache_sha256"):
        raise ValueError(
            f"score cache SHA-256 mismatch: expected={manifest.get('cache_sha256')} actual={actual_hash}"
        )
    sources = tuple(str(name) for name in manifest.get("sources", ()))
    if not sources:
        raise ValueError("score cache manifest has no sources")
    with np.load(cache, allow_pickle=False) as archive:
        train = {name: np.asarray(archive[f"train__{name}"], dtype=float) for name in sources}
        test = {name: np.asarray(archive[f"test__{name}"], dtype=float) for name in sources}
    train = _validated_vectors(train, "TRAIN OOF")
    test = _validated_vectors(test, "TEST score")
    for name in sources:
        if len(train[name]) != int(manifest["train_lengths"][name]):
            raise ValueError(f"TRAIN score length mismatch: {name}")
        if len(test[name]) != int(manifest["test_lengths"][name]):
            raise ValueError(f"TEST score length mismatch: {name}")
    return ScoreBundle(
        dataset=str(manifest["dataset"]),
        train_oof=train,
        test_scores=test,
        provenance=dict(manifest.get("provenance", {})),
    )


def _csv_map(path: Path, key: str) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapped = {str(row[key]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"duplicate manifest key: {path}")
    return mapped


class FrozenScoreRepository:
    def __init__(
        self,
        source_rows: dict[str, dict[str, str]],
        canonical_rows: dict[str, dict[str, str]],
    ) -> None:
        self.source_rows = source_rows
        self.canonical_rows = canonical_rows

    @classmethod
    def from_frozen_manifests(cls) -> "FrozenScoreRepository":
        if sha256_file(EXP153_MANIFEST) != EXP153_MANIFEST_SHA256:
            raise ValueError("Exp153 score manifest SHA-256 mismatch")
        if sha256_file(CANONICAL_MANIFEST) != CANONICAL_MANIFEST_SHA256:
            raise ValueError("canonical score manifest SHA-256 mismatch")
        source = _csv_map(EXP153_MANIFEST, "dataset_name")
        canonical = _csv_map(CANONICAL_MANIFEST, "dataset_name")
        if len(source) != 1117:
            raise ValueError(f"full score source coverage mismatch: {len(source)}/1117")
        return cls(source, canonical)

    def load(self, dataset: str) -> ScoreBundle:
        if dataset not in self.source_rows:
            raise ValueError(f"dataset missing from full score manifest: {dataset}")
        row = self.source_rows[dataset]
        path = EXP153_ROOT / "cache" / row["cache_file"]
        if sha256_file(path) != row["cache_sha256"]:
            raise ValueError(f"full score cache SHA-256 mismatch: {dataset}")
        raw_names = tuple(str(name) for name in ast.literal_eval(row["source_names"]))
        train: dict[str, np.ndarray] = {}
        test: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as archive:
            for raw_name in raw_names:
                if raw_name not in SOURCE_NAMES:
                    continue
                name = SOURCE_NAMES[raw_name]
                train[name] = np.asarray(archive[f"{raw_name}__train_oof"], dtype=float)
                test[name] = np.asarray(archive[f"{raw_name}__test"], dtype=float)

        canonical_available = dataset in self.canonical_rows
        if canonical_available:
            canonical_row = self.canonical_rows[dataset]
            canonical_path = CANONICAL_ROOT / "canonical_score_cache" / canonical_row["score_file"]
            if sha256_file(canonical_path) != canonical_row["score_sha256"]:
                raise ValueError(f"canonical score cache SHA-256 mismatch: {dataset}")
            with np.load(canonical_path, allow_pickle=False) as archive:
                train["canonical_normal_state"] = np.asarray(
                    archive["knn_local_gap__train_oof"], dtype=float
                )
                test["canonical_normal_state"] = np.asarray(
                    archive["knn_local_gap__test"], dtype=float
                )
        train_lengths = {len(values) for values in train.values()}
        test_lengths = {len(values) for values in test.values()}
        if train_lengths != {int(row["n_train"])} or test_lengths != {int(row["n_test"])}:
            raise ValueError(f"frozen score vector length mismatch: {dataset}")
        return ScoreBundle(
            dataset=dataset,
            train_oof=_validated_vectors(train, "TRAIN OOF"),
            test_scores=_validated_vectors(test, "TEST score"),
            provenance={
                "source_cache_sha256": row["cache_sha256"],
                "canonical_available": canonical_available,
                "canonical_cache_sha256": (
                    self.canonical_rows[dataset]["score_sha256"] if canonical_available else None
                ),
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
                "train_target_length": int(row["train_target_len"]),
            },
        )
