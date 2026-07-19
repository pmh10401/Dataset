from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.contracts import ScoreBundle
from experiments.exp226_235_hard_alert_quality.database import sha256_file


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

