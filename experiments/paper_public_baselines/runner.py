"""Resumable values-only scoring and immutable score-root verification."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import time
from typing import Callable, Mapping

import numpy as np

from .artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    validate_score_payload,
    write_create_only_bytes,
    write_score_shard_create_only,
)
from .calibration import conformal_alerts
from .config import ALPHAS, CLASSICAL_METHOD_IDS, METHOD_IDS, SEED
from .contracts import ScoreResult
from .data import load_dataset_values
from .methods import score_published_method
from .preprocessing import prepare_dataset


Scorer = Callable[[str, np.ndarray, np.ndarray, int], ScoreResult]


@dataclass(frozen=True, slots=True)
class ScoreRunConfig:
    db_path: Path
    run_root: Path
    datasets: tuple[str, ...]
    methods: tuple[str, ...]
    workers: int
    database_sha256: str
    seed: int = SEED
    epochs: int = 50

    def __post_init__(self) -> None:
        if not self.datasets or len(self.datasets) != len(set(self.datasets)):
            raise ValueError("datasets must be non-empty and unique")
        if not self.methods or len(self.methods) != len(set(self.methods)):
            raise ValueError("methods must be non-empty and unique")
        if any(method not in METHOD_IDS for method in self.methods):
            raise ValueError("run config contains an unknown method")
        if type(self.workers) is not int or not 1 <= self.workers <= 7:
            raise ValueError("workers must be between one and seven")
        if len(self.database_sha256) != 64:
            raise ValueError("database SHA256 is invalid")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")


@dataclass(frozen=True, slots=True)
class RunSummary:
    completed: int
    reused: int
    abstained: int
    failed: int


@dataclass(frozen=True, slots=True)
class FrozenScoreManifest:
    run_root: Path
    run_config_sha256: str
    database_sha256: str
    pair_count: int
    shards: tuple[Mapping[str, object], ...]


def run_methods(
    config: ScoreRunConfig,
    *,
    selected_methods: tuple[str, ...] | None = None,
    scorer: Scorer | None = None,
) -> RunSummary:
    """Score requested dataset-method pairs, safely reusing verified shards."""
    run_config, run_config_sha = _initialize_run(config)
    active_methods = tuple(selected_methods or config.methods)
    if (
        not active_methods
        or len(active_methods) != len(set(active_methods))
        or any(method not in config.methods for method in active_methods)
    ):
        raise ValueError("selected methods must be a unique subset of run methods")
    pending: list[tuple[str, str]] = []
    reused = 0
    abstained = 0
    for method in active_methods:
        for dataset in config.datasets:
            existing = _reuse_verified_shard(
                config, run_config_sha, method, dataset
            )
            if existing is None:
                pending.append((method, dataset))
            else:
                reused += 1
                abstained += int(existing["status"] == "abstain")

    results: list[dict[str, object]] = []
    if pending and config.workers > 1 and scorer is None:
        if any(method not in CLASSICAL_METHOD_IDS for method, _ in pending):
            raise ValueError("multi-process scoring is allowed only for classical methods")
        with ProcessPoolExecutor(max_workers=min(config.workers, 7)) as executor:
            futures = {
                executor.submit(
                    _score_pair,
                    config.db_path,
                    dataset,
                    method,
                    _pair_seed(config.seed, method, dataset),
                    config.epochs,
                ): (method, dataset)
                for method, dataset in pending
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for method, dataset in pending:
            results.append(
                _score_pair(
                    config.db_path,
                    dataset,
                    method,
                    _pair_seed(config.seed, method, dataset),
                    config.epochs,
                    scorer=scorer,
                )
            )

    for payload in results:
        _publish_shard(config, run_config_sha, payload)
        abstained += int(payload["status"] == "abstain")
    return RunSummary(
        completed=len(config.datasets) * len(active_methods),
        reused=reused,
        abstained=abstained,
        failed=0,
    )


def freeze_score_root(run_root: Path) -> FrozenScoreManifest:
    """Create a no-clobber manifest only after exact pair coverage verifies."""
    root = Path(run_root)
    run_config, run_config_sha = _verified_run_config(root)
    datasets = _string_tuple(run_config.get("datasets"), "datasets")
    methods = _string_tuple(run_config.get("methods"), "methods")
    _assert_exact_shard_inventory(root, methods, datasets)
    entries: list[dict[str, object]] = []
    for method in methods:
        for dataset in datasets:
            shard, digest, relative = _verified_shard(
                root, run_config_sha, str(run_config["database_sha256"]), method, dataset
            )
            entries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "status": shard["status"],
                }
            )
    expected_count = len(datasets) * len(methods)
    if len(entries) != expected_count:
        raise ValueError("score-root coverage is incomplete")
    payload = {
        "schema": "paper-public-baselines.frozen-scores.v1",
        "run_config_sha256": run_config_sha,
        "database_sha256": run_config["database_sha256"],
        "pair_count": expected_count,
        "shards": entries,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    manifest_bytes = canonical_json_bytes(payload)
    write_create_only_bytes(root / "frozen_score_manifest.json", manifest_bytes)
    write_create_only_bytes(
        root / "frozen_score_manifest.sha256",
        (hashlib.sha256(manifest_bytes).hexdigest() + "\n").encode(),
    )
    return verify_score_root(root)


def verify_score_root(run_root: Path) -> FrozenScoreManifest:
    """Verify manifest, run identity, every shard digest, and exact coverage."""
    root = Path(run_root)
    manifest_path = root / "frozen_score_manifest.json"
    digest_path = root / "frozen_score_manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise ValueError("frozen score manifest is absent")
    expected_manifest_digest = digest_path.read_text(encoding="ascii").strip()
    actual_manifest_digest = sha256_file(manifest_path)
    if expected_manifest_digest != actual_manifest_digest:
        raise ValueError("frozen score manifest SHA256 mismatch")
    manifest = read_strict_json(manifest_path)
    expected_fields = {
        "schema", "run_config_sha256", "database_sha256", "pair_count",
        "shards", "contains_test_labels", "contains_evaluation_metrics",
    }
    if set(manifest) != expected_fields or manifest.get("schema") != "paper-public-baselines.frozen-scores.v1":
        raise ValueError("frozen score manifest schema is invalid")
    if manifest.get("contains_test_labels") is not False or manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("frozen score manifest violates the label boundary")
    run_config, run_config_sha = _verified_run_config(root)
    if manifest.get("run_config_sha256") != run_config_sha or manifest.get("database_sha256") != run_config.get("database_sha256"):
        raise ValueError("frozen score manifest run identity mismatch")
    datasets = _string_tuple(run_config.get("datasets"), "datasets")
    methods = _string_tuple(run_config.get("methods"), "methods")
    _assert_exact_shard_inventory(root, methods, datasets)
    expected_pairs = {(method, dataset) for method in methods for dataset in datasets}
    raw_entries = manifest.get("shards")
    if not isinstance(raw_entries, list) or manifest.get("pair_count") != len(expected_pairs) or len(raw_entries) != len(expected_pairs):
        raise ValueError("frozen score manifest coverage is invalid")
    observed_pairs: set[tuple[str, str]] = set()
    verified_entries: list[Mapping[str, object]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {"dataset", "method", "path", "sha256", "status"}:
            raise ValueError("frozen score manifest shard entry is invalid")
        method, dataset = raw_entry.get("method"), raw_entry.get("dataset")
        if type(method) is not str or type(dataset) is not str:
            raise ValueError("frozen score manifest pair is invalid")
        shard, digest, relative = _verified_shard(
            root, run_config_sha, str(run_config["database_sha256"]), method, dataset
        )
        if raw_entry.get("sha256") != digest:
            raise ValueError("score shard SHA256 differs from frozen manifest")
        if raw_entry.get("path") != relative.as_posix() or raw_entry.get("status") != shard.get("status"):
            raise ValueError("score shard metadata differs from frozen manifest")
        observed_pairs.add((method, dataset))
        verified_entries.append(dict(raw_entry))
    if observed_pairs != expected_pairs:
        raise ValueError("frozen score manifest coverage is not exact")
    return FrozenScoreManifest(
        run_root=root,
        run_config_sha256=run_config_sha,
        database_sha256=str(run_config["database_sha256"]),
        pair_count=len(expected_pairs),
        shards=tuple(verified_entries),
    )


def _initialize_run(config: ScoreRunConfig) -> tuple[dict[str, object], str]:
    actual_db_sha = sha256_file(config.db_path)
    if actual_db_sha != config.database_sha256:
        raise ValueError("database SHA256 mismatch")
    payload = _run_config_payload(config)
    encoded = canonical_json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    write_create_only_bytes(config.run_root / "run_config.json", encoded)
    write_create_only_bytes(
        config.run_root / "run_config.sha256", (digest + "\n").encode()
    )
    verified, verified_digest = _verified_run_config(config.run_root)
    if verified != payload or verified_digest != digest:
        raise ValueError("existing run config mismatch")
    return verified, verified_digest


def _verified_run_config(root: Path) -> tuple[dict[str, object], str]:
    config_path = root / "run_config.json"
    digest_path = root / "run_config.sha256"
    if not config_path.is_file() or not digest_path.is_file():
        raise ValueError("run config is absent")
    expected = digest_path.read_text(encoding="ascii").strip()
    actual = sha256_file(config_path)
    if expected != actual:
        raise ValueError("run config SHA256 mismatch")
    payload = read_strict_json(config_path)
    if payload.get("schema") != "paper-public-baselines.score-run.v1":
        raise ValueError("run config schema is invalid")
    return payload, actual


def _run_config_payload(config: ScoreRunConfig) -> dict[str, object]:
    return {
        "schema": "paper-public-baselines.score-run.v1",
        "database_path": str(config.db_path.resolve(strict=True)),
        "database_sha256": config.database_sha256,
        "datasets": list(config.datasets),
        "methods": list(config.methods),
        "alphas": list(ALPHAS),
        "seed": config.seed,
        "epochs": config.epochs,
        "worker_policy": "classical_up_to_7_deep_exactly_1",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _version("scipy"),
            "scikit_learn": _version("scikit-learn"),
            "torch": _version("torch"),
        },
        "source_sha256": _source_hashes(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


def _score_pair(
    db_path: Path,
    dataset: str,
    method: str,
    seed: int,
    epochs: int,
    scorer: Scorer | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    prepared = prepare_dataset(load_dataset_values(db_path, dataset))
    result = (
        scorer(method, prepared.train, prepared.test, seed)
        if scorer is not None
        else score_published_method(
            method, prepared.train, prepared.test, seed, epochs=epochs
        )
    )
    if result.method_id != method:
        raise ValueError("scorer returned a different method identity")
    alerts = {
        format(alpha, ".12g"): list(
            conformal_alerts(
                result.oof_scores,
                result.test_scores,
                prepared.test_indices,
                alpha,
            )
        )
        if result.status == "complete"
        else []
        for alpha in ALPHAS
    }
    payload: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "n_train": len(prepared.train),
        "n_test": len(prepared.test),
        "status": result.status,
        "reason": result.reason,
        "test_indices": list(prepared.test_indices),
        "train_oof_scores": result.oof_scores.tolist(),
        "test_scores": result.test_scores.tolist(),
        "alerts_by_alpha": alerts,
        "runtime_seconds": time.perf_counter() - started,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    return validate_score_payload(payload)


def _publish_shard(
    config: ScoreRunConfig,
    run_config_sha: str,
    payload: Mapping[str, object],
) -> None:
    path = _shard_path(config.run_root, str(payload["method"]), str(payload["dataset"]))
    digest = write_score_shard_create_only(path, payload)
    metadata = {
        "schema": "paper-public-baselines.score-shard-meta.v1",
        "run_config_sha256": run_config_sha,
        "database_sha256": config.database_sha256,
        "shard_sha256": digest,
    }
    write_create_only_bytes(path.with_suffix(".meta"), canonical_json_bytes(metadata))


def _reuse_verified_shard(
    config: ScoreRunConfig,
    run_config_sha: str,
    method: str,
    dataset: str,
) -> dict[str, object] | None:
    path = _shard_path(config.run_root, method, dataset)
    metadata_path = path.with_suffix(".meta")
    if not path.exists() and not metadata_path.exists():
        return None
    shard, _, _ = _verified_shard(
        config.run_root, run_config_sha, config.database_sha256, method, dataset
    )
    values = prepare_dataset(load_dataset_values(config.db_path, dataset))
    if shard["n_train"] != len(values.train) or shard["n_test"] != len(values.test) or shard["test_indices"] != list(values.test_indices):
        raise ValueError("existing score shard population mismatch")
    return shard


def _verified_shard(
    root: Path,
    run_config_sha: str,
    database_sha: str,
    method: str,
    dataset: str,
) -> tuple[dict[str, object], str, Path]:
    path = _shard_path(root, method, dataset)
    metadata_path = path.with_suffix(".meta")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("score-root coverage is incomplete")
    metadata = read_strict_json(metadata_path)
    expected_metadata_fields = {"schema", "run_config_sha256", "database_sha256", "shard_sha256"}
    if set(metadata) != expected_metadata_fields or metadata.get("schema") != "paper-public-baselines.score-shard-meta.v1":
        raise ValueError("score shard metadata is invalid")
    digest = sha256_file(path)
    if metadata.get("shard_sha256") != digest:
        raise ValueError("score shard SHA256 mismatch")
    if metadata.get("run_config_sha256") != run_config_sha or metadata.get("database_sha256") != database_sha:
        raise ValueError("score shard run identity mismatch")
    shard = validate_score_payload(read_strict_json(path))
    if shard["method"] != method or shard["dataset"] != dataset:
        raise ValueError("score shard coordinate mismatch")
    return shard, digest, path.relative_to(root)


def _shard_path(root: Path, method: str, dataset: str) -> Path:
    token = hashlib.sha256(dataset.encode("utf-8")).hexdigest()
    return Path(root) / "shards" / method / f"{token}.json"


def _assert_exact_shard_inventory(
    root: Path, methods: tuple[str, ...], datasets: tuple[str, ...]
) -> None:
    shard_root = Path(root) / "shards"
    expected = {
        _shard_path(root, method, dataset).relative_to(shard_root)
        for method in methods
        for dataset in datasets
    }
    expected |= {path.with_suffix(".meta") for path in tuple(expected)}
    actual = (
        {path.relative_to(shard_root) for path in shard_root.rglob("*") if path.is_file()}
        if shard_root.is_dir()
        else set()
    )
    if actual != expected:
        raise ValueError("score shard inventory is not exact")


def _pair_seed(seed: int, method: str, dataset: str) -> int:
    material = f"{seed}\0{method}\0{dataset}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "acceleration.py", "artifacts.py", "calibration.py", "config.py",
        "contracts.py", "data.py", "methods/classical.py",
        "methods/deep_common.py", "methods/deep_svdd.py", "methods/drocc.py",
        "preprocessing.py", "runner.py",
    )
    return {name: sha256_file(package / name) for name in names}


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(type(item) is not str or not item for item in value):
        raise ValueError(f"run config {name} is invalid")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"run config {name} is not unique")
    return result


__all__ = [
    "FrozenScoreManifest",
    "RunSummary",
    "ScoreRunConfig",
    "freeze_score_root",
    "run_methods",
    "verify_score_root",
]
