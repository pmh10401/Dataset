"""Resumable, label-free cache and prediction freeze for Exp268-270.

The cache boundary is intentionally before evaluation: it may read aligned TEST
values to score them, but it neither opens nor carries TEST labels.  Every
cache unit is self-verifying, so a partial crash or an edited artifact is
rebuilt rather than silently reused.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import multiprocessing
import os
import socket
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates
from experiments.exp239_248_diverse_score_sources.foundation import (
    database_hash,
    load_bundle,
    stable_json_hash,
)

from . import config as experiment_config
from .config import ALPHAS, DB_PATH, SECOND_SEED, SEED
from .contracts import FrozenSourcePrediction, SourceScore
from .source_registry import build_source, registration, source_names, source_variants


CACHE_SCHEMA_VERSION = 2
PREDICTION_SCHEMA_VERSION = 2
_FORBIDDEN_PREDICTION_FIELDS = frozenset(
    {
        "y_test", "label", "labels", "tp", "fp", "fn", "f1", "precision", "metrics",
        "hard_tp", "hard_fp", "hard_fn", "hard_f1", "combined_f1", "evaluation_metrics",
        "true_label", "true_indices", "actual_anomaly_positions", "test_anomaly_positions",
    }
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _json_safe(value: object) -> object:
    """Detach frozen provenance objects into canonical JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(child) for child in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(child) for child in sorted(value, key=repr)]
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _canonical_bytes(payload) + b"\n")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(arrays[key]) for key in sorted(arrays)})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_stem(dataset: str) -> str:
    return hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()[:24]


def source_root(root: Path, source: str) -> Path:
    return Path(root) / "source_cache" / registration(source).source


def cache_paths(root: Path, source: str, dataset: str) -> tuple[Path, Path]:
    base = source_root(root, source) / "cache" / _dataset_stem(dataset)
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _source_code_hash(source: str) -> str:
    instance = build_source(source)
    source_file = inspect.getsourcefile(type(instance))
    if source_file is None:
        raise RuntimeError(f"cannot locate code file for {source}")
    return _file_sha256(Path(source_file))


def _module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    path = getattr(module, "__file__", None)
    if not isinstance(path, str):
        raise RuntimeError(f"cannot locate module file: {module_name}")
    return Path(path)


def _module_hashes(source: str) -> dict[str, str]:
    """Hash every local module that can alter a cached score or calibration."""
    canonical = registration(source).source
    source_module = {
        "scattering": "experiments.exp267_273_zero_support_fn_recovery.scattering_source",
        "signature": "experiments.exp267_273_zero_support_fn_recovery.signature_source",
        "rff_forecast": "experiments.exp267_273_zero_support_fn_recovery.forecast_rff_source",
        "tcn_forecast": "experiments.exp267_273_zero_support_fn_recovery.forecast_tcn_source",
    }[canonical]
    names = (
        "experiments.exp267_273_zero_support_fn_recovery.config",
        "experiments.exp267_273_zero_support_fn_recovery.contracts",
        "experiments.exp267_273_zero_support_fn_recovery.source_registry",
        "experiments.exp267_273_zero_support_fn_recovery.source_pipeline",
        source_module,
        "experiments.exp239_248_diverse_score_sources.contracts",
        "experiments.exp239_248_diverse_score_sources.foundation",
        "experiments.exp239_248_diverse_score_sources.source_base",
        "experiments.exp239_248_diverse_score_sources.calibration",
        "experiments.exp226_235_hard_alert_quality.database",
        "experiments.exp226_235_hard_alert_quality.preprocessing",
    )
    return {name: _file_sha256(_module_path(name)) for name in names}


def _policy_constants() -> dict[str, object]:
    """Freeze all simple uppercase policy values, including seeds and limits."""
    def simple(value: object) -> bool:
        if isinstance(value, (str, int, float, bool, type(None))):
            return True
        return isinstance(value, (tuple, list)) and all(simple(child) for child in value)

    values: dict[str, object] = {}
    for name in dir(experiment_config):
        if not name.isupper():
            continue
        value = getattr(experiment_config, name)
        if simple(value):
            values[name] = _json_safe(value)
    return values


def _runtime_versions() -> dict[str, str]:
    """Capture installed numerical runtime versions without consulting TEST data."""
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for package in ("scipy", "scikit-learn", "torch", "kymatio", "iisignature"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _run_contract(
    source: str,
    *,
    device: str,
    benchmark_record: Mapping[str, object] | None,
    db_sha256: str,
) -> dict[str, object]:
    """Build a complete, label-free fingerprint of score generation semantics."""
    canonical = registration(source).source
    config = _source_config(canonical, device=device, benchmark_record=benchmark_record)
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "source": canonical,
        "variants": tuple(item.variant for item in source_variants(canonical)),
        "seeds": (SEED, SECOND_SEED),
        "runtime_config": config,
        "policy_constants": _policy_constants(),
        "runtime_versions": _runtime_versions(),
        "db_sha256": str(db_sha256),
        "module_hashes": _module_hashes(canonical),
    }
    return {**payload, "run_contract_sha256": _run_contract_hash(payload)}


def _run_contract_hash(contract: Mapping[str, object]) -> str:
    return stable_json_hash(_json_safe({
        key: value for key, value in contract.items() if key != "run_contract_sha256"
    }))


def _variant_run_contract_hashes(contract: Mapping[str, object]) -> dict[str, str]:
    source = str(contract["source"])
    return {
        item.variant: stable_json_hash(_json_safe({
            "run_contract_sha256": contract["run_contract_sha256"],
            "source": source,
            "variant": item.variant,
            "family": item.family,
            "evidence_identity": item.evidence_identity,
        }))
        for item in source_variants(source)
    }


def _source_config(
    source: str,
    *,
    device: str,
    benchmark_record: Mapping[str, object] | None,
) -> dict[str, object]:
    variants = source_variants(source)
    item = variants[0]
    benchmark_hash = None
    if benchmark_record is not None:
        benchmark_hash = stable_json_hash(_json_safe(dict(benchmark_record)))
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source": item.source,
        "algorithm": item.algorithm,
        "source_version": item.source_version,
        "family": item.family,
        "evidence_identity": item.evidence_identity,
        "variants": tuple(variant.variant for variant in variants),
        "minimum_train": item.minimum_train,
        "seed": SEED,
        "device_request": str(device),
        "tcn_benchmark_record_hash": benchmark_hash,
        "contains_test_labels": False,
    }


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    descriptor = {
        "dtype": str(array.dtype),
        "shape": tuple(int(value) for value in array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
    return stable_json_hash(descriptor)


def _input_hash(bundle: object) -> str:
    train = np.asarray(getattr(bundle, "train_normal"))
    test = np.asarray(getattr(bundle, "test_values"))
    return stable_json_hash(
        {
            "dataset": str(getattr(bundle, "dataset")),
            "target_length": int(getattr(bundle, "target_length")),
            "train_hash": _array_hash(train),
            "test_values_hash": _array_hash(test),
        }
    )


def _cache_hash(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    stable_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"cache_sha256", "archive_sha256", "metadata_sha256"}
    }
    return stable_json_hash(_json_safe(
        {
            "metadata": stable_metadata,
            "arrays": {key: _array_hash(np.asarray(value)) for key, value in sorted(arrays.items())},
        }
    ))


def _metadata_hash(metadata: Mapping[str, object]) -> str:
    return stable_json_hash(_json_safe({key: value for key, value in metadata.items() if key != "metadata_sha256"}))


def cache_result_complete(metadata: Mapping[str, object]) -> bool:
    """An explicit calibration/resource abstention is a complete source result."""
    abstained = bool(metadata.get("abstained", False))
    reason = metadata.get("abstention_reason")
    return bool(abstained and isinstance(reason, str) and reason) or bool(
        metadata.get("eligible", False) and not abstained
    )


def _score_arrays(result: SourceScore) -> dict[str, np.ndarray]:
    if result.provenance.get("contains_test_labels") is not False:
        raise ValueError("source provenance must explicitly exclude TEST labels")
    arrays = {
        "train_oof": np.asarray(result.train_oof if result.train_oof is not None else [], dtype=float),
        "test_scores": np.asarray(result.test_scores if result.test_scores is not None else [], dtype=float),
    }
    if not result.abstained:
        if not len(arrays["train_oof"]) or not len(arrays["test_scores"]):
            raise ValueError("eligible source must emit non-empty train OOF and TEST scores")
        if not all(np.isfinite(values).all() for values in arrays.values()):
            raise ValueError("source cache scores must be finite")
    for key, output_name in (
        ("per_seed_train_oof", "train_oof__seed_"),
        ("per_seed_test_scores", "test_scores__seed_"),
    ):
        seed_mapping = result.provenance.get(key)
        if seed_mapping is None:
            continue
        if not isinstance(seed_mapping, Mapping):
            raise TypeError(f"{key} must be a mapping")
        for seed, values in sorted(seed_mapping.items(), key=lambda item: str(item[0])):
            array = np.asarray(values, dtype=float)
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite {key} values")
            arrays[f"{output_name}{seed}"] = array
    return arrays


def _call_factory(
    source: str,
    *,
    device: str,
    benchmark_record: Mapping[str, object] | None,
    source_factory: Callable[..., object] | None,
) -> object:
    if source_factory is None:
        return build_source(source, device=device, benchmark_record=benchmark_record)
    try:
        return source_factory(source, device=device, benchmark_record=benchmark_record)
    except TypeError:
        return source_factory()


def _cache_is_valid(
    archive_path: Path,
    metadata_path: Path,
    *,
    expected: Mapping[str, object],
) -> dict[str, object] | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return None
        required = (
            "cache_schema_version", "config_hash", "db_sha256", "code_sha256",
            "input_sha256", "cache_sha256", "metadata_sha256", "archive_sha256",
            "run_contract", "run_contract_sha256", "variant_run_contract_sha256",
        )
        if any(field not in metadata for field in required):
            return None
        if any(metadata.get(field) != value for field, value in expected.items()):
            return None
        if metadata.get("metadata_sha256") != _metadata_hash(metadata):
            return None
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        if metadata.get("archive_sha256") != _file_sha256(archive_path):
            return None
        if metadata.get("cache_sha256") != _cache_hash(arrays, metadata):
            return None
        if not cache_result_complete(metadata):
            return None
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def build_cache_unit(
    root: Path,
    source: str,
    dataset: str,
    *,
    device: str = "cpu",
    reset: bool = False,
    benchmark_record: Mapping[str, object] | None = None,
    source_factory: Callable[..., object] | None = None,
    db_hash: str | None = None,
    code_hash: str | None = None,
    run_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build or verify one atomic label-free source cache unit."""
    canonical = registration(source).source
    if canonical == "tcn_forecast" and benchmark_record is None:
        raise ValueError("TCN cache requires one frozen verified benchmark record")
    archive_path, metadata_path = cache_paths(Path(root), canonical, dataset)
    config = _source_config(canonical, device=device, benchmark_record=benchmark_record)
    db_sha256 = db_hash or database_hash(DB_PATH)
    contract = dict(run_contract) if run_contract is not None else _run_contract(
        canonical, device=device, benchmark_record=benchmark_record, db_sha256=db_sha256
    )
    if contract.get("run_contract_sha256") != _run_contract_hash(contract):
        raise ValueError("run contract self-hash mismatch")
    if contract.get("source") != canonical or contract.get("db_sha256") != db_sha256:
        raise ValueError("run contract does not match source or DB invocation")
    expected_base = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "config_hash": stable_json_hash(config),
        "db_sha256": db_sha256,
        "code_sha256": code_hash or _source_code_hash(canonical),
        "run_contract_sha256": contract["run_contract_sha256"],
    }
    if not reset and archive_path.exists() and metadata_path.exists():
        # Input hash is embedded in the existing cache; calculate it only if all
        # cheap identity checks pass, keeping reuse safe without label access.
        existing = _cache_is_valid(archive_path, metadata_path, expected=expected_base)
        if existing is not None:
            bundle = load_bundle(dataset)
            if existing.get("input_sha256") == _input_hash(bundle):
                return existing

    bundle = load_bundle(dataset)
    source_object = _call_factory(
        canonical,
        device=device,
        benchmark_record=benchmark_record,
        source_factory=source_factory,
    )
    if canonical == "tcn_forecast" and not bool(
        getattr(source_object, "benchmark_record_verified", False)
    ):
        raise ValueError("TCN cache requires a verified frozen benchmark record")
    result = source_object.score(bundle, SEED, archive_path.parent)
    if not isinstance(result, SourceScore):
        raise TypeError("source factory must return a SourceScore")
    arrays = _score_arrays(result)
    expected_train = len(np.asarray(bundle.train_normal))
    expected_test = len(np.asarray(bundle.test_values))
    if not result.abstained and (
        len(arrays["train_oof"]) != expected_train or len(arrays["test_scores"]) != expected_test
    ):
        raise ValueError("source score array length does not match label-free bundle")
    metadata: dict[str, object] = {
        **expected_base,
        "dataset": str(dataset),
        "source": canonical,
        "source_result_name": result.source,
        "source_result_variant": result.variant,
        "n_train": expected_train,
        "n_test": expected_test,
        "target_length": int(bundle.target_length),
        "eligible": bool(result.eligible),
        "abstained": bool(result.abstained),
        "abstention_reason": result.abstention_reason,
        "config": config,
        "run_contract": contract,
        "run_contract_sha256": contract["run_contract_sha256"],
        "variant_run_contract_sha256": _variant_run_contract_hashes(contract),
        "input_sha256": _input_hash(bundle),
        "provenance": _json_safe({
            key: value
            for key, value in result.provenance.items()
            if key not in {"per_seed_train_oof", "per_seed_test_scores"}
        }),
        "contains_test_labels": False,
        "cache_file": str(archive_path.relative_to(Path(root))),
    }
    if not cache_result_complete(metadata):
        raise ValueError("source result is neither eligible nor explicit abstention")
    _atomic_npz(archive_path, arrays)
    metadata["archive_sha256"] = _file_sha256(archive_path)
    metadata["cache_sha256"] = _cache_hash(arrays, metadata)
    metadata["metadata_sha256"] = _metadata_hash(metadata)
    _atomic_json(metadata_path, metadata)
    return metadata


def _worker_build_cache_unit(arguments: tuple[object, ...]) -> dict[str, object]:
    root, source, dataset, device, reset, benchmark_record, db_hash, code_hash, contract = arguments
    return build_cache_unit(
        Path(str(root)), str(source), str(dataset), device=str(device), reset=bool(reset),
        benchmark_record=benchmark_record if isinstance(benchmark_record, Mapping) else None,
        db_hash=str(db_hash), code_hash=str(code_hash),
        run_contract=contract if isinstance(contract, Mapping) else None,
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: tuple[str, ...]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def tcn_exclusive_lock(root: Path, *, nonblocking: bool = True) -> Iterator[dict[str, object]]:
    """Hold an OS-level exclusive lock for the complete TCN cache stage."""
    path = source_root(Path(root), "tcn_forecast") / "tcn_exclusive.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    try:
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise RuntimeError(f"TCN exclusive lock is already held: {path}") from exc
        owner = {
            "lock_path": str(path),
            "owner_pid": os.getpid(),
            "owner_host": socket.gethostname(),
            "acquired_at_unix": time.time(),
            "nonblocking": bool(nonblocking),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(owner, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield owner
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def cache_source(
    root: Path,
    source: str,
    *,
    datasets: tuple[str, ...],
    workers: int = 7,
    device: str = "cpu",
    reset: bool = False,
    benchmark_record: Mapping[str, object] | None = None,
    expected_dataset_count: int | None = None,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> list[dict[str, object]]:
    """Build one source across the frozen population without silently dropping errors."""
    canonical = registration(source).source
    selected = tuple(sorted(dict.fromkeys(str(item) for item in datasets)))
    if len(selected) != len(tuple(datasets)):
        raise ValueError("dataset list contains duplicates")
    if expected_dataset_count is not None and len(selected) != int(expected_dataset_count):
        raise ValueError("source cache dataset coverage contract mismatch")
    item = registration(source)
    if item.model_based and workers != 1:
        raise ValueError("TCN cache requires exactly one worker and exclusive selected device")
    if item.model_based:
        probe = build_source(canonical, device=device, benchmark_record=benchmark_record)
        if not bool(getattr(probe, "benchmark_record_verified", False)):
            raise ValueError("TCN cache requires one verified frozen benchmark record")
    worker_count = 1 if item.model_based else min(7, max(1, int(workers)))
    base = source_root(Path(root), canonical)
    db_sha = database_hash(DB_PATH)
    code_sha = _source_code_hash(canonical)
    contract = _run_contract(
        canonical, device=device, benchmark_record=benchmark_record, db_sha256=db_sha
    )
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    def finish(done: int, dataset: str) -> None:
        if progress is not None:
            progress(done, len(selected), dataset, len(errors))

    lock_context = tcn_exclusive_lock(Path(root)) if item.model_based else nullcontext(None)
    with lock_context as lock_owner:
        if worker_count == 1:
            for done, dataset in enumerate(selected, 1):
                try:
                    rows.append(build_cache_unit(
                        Path(root), canonical, dataset, device=device, reset=reset,
                        benchmark_record=benchmark_record, db_hash=db_sha, code_hash=code_sha,
                        run_contract=contract,
                    ))
                except Exception as error:  # persist failures and block prediction freeze.
                    errors.append({"dataset": dataset, "error_type": type(error).__name__, "error": str(error)})
                finish(done, dataset)
        else:
            arguments = [
                (str(root), canonical, dataset, device, reset, None, db_sha, code_sha, contract)
                for dataset in selected
            ]
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                futures = {executor.submit(_worker_build_cache_unit, argument): argument[2] for argument in arguments}
                for done, future in enumerate(as_completed(futures), 1):
                    dataset = str(futures[future])
                    try:
                        rows.append(future.result())
                    except Exception as error:
                        errors.append({"dataset": dataset, "error_type": type(error).__name__, "error": str(error)})
                    finish(done, dataset)
    rows.sort(key=lambda row: str(row["dataset"]))
    errors.sort(key=lambda row: str(row["dataset"]))
    manifest_fields = (
        "dataset", "source", "n_train", "n_test", "target_length", "eligible", "abstained",
        "abstention_reason", "config_hash", "db_sha256", "code_sha256", "input_sha256",
        "cache_sha256", "archive_sha256", "metadata_sha256", "cache_file",
    )
    _write_csv(base / "cache_manifest.csv", rows, manifest_fields)
    _write_csv(base / "cache_errors.csv", errors, ("dataset", "error_type", "error"))
    complete = not errors and len(rows) == len(selected)
    _atomic_json(base / "source_manifest.json", {
        "source": canonical,
        "datasets_expected": len(selected),
        "datasets_cached": len(rows),
        "errors": len(errors),
        "complete": complete,
        "device": str(device),
        "workers": worker_count,
        "tcn_exclusive": bool(item.model_based),
        "run_contract_sha256": contract["run_contract_sha256"],
        "runtime_config": contract["runtime_config"],
        "tcn_lock": lock_owner if item.model_based else None,
        "contains_test_labels": False,
    })
    return rows


def _read_cache(
    root: Path,
    source: str,
    dataset: str,
    *,
    bundle_loader: Callable[[str], object] = load_bundle,
    device: str = "cpu",
    benchmark_record: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    archive_path, metadata_path = cache_paths(root, source, dataset)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("cache metadata must be an object")
    config = metadata.get("config")
    item = registration(source)
    if not isinstance(config, dict) or any(
        config.get(field) != expected
        for field, expected in {
            "source": item.source,
            "algorithm": item.algorithm,
            "source_version": item.source_version,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
        }.items()
    ):
        raise ValueError(f"cache configuration is stale or incompatible: source={source}")
    if metadata.get("config_hash") != stable_json_hash(config):
        raise ValueError(f"cache configuration hash mismatch: source={source}")
    current_contract = _run_contract(
        source,
        device=device,
        benchmark_record=benchmark_record,
        db_sha256=database_hash(DB_PATH),
    )
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "config_hash": metadata.get("config_hash"),
        "db_sha256": current_contract["db_sha256"],
        "code_sha256": _source_code_hash(source),
        "run_contract_sha256": current_contract["run_contract_sha256"],
    }
    verified = _cache_is_valid(archive_path, metadata_path, expected=expected)
    if verified is None:
        raise ValueError(f"invalid or stale cache: source={source} dataset={dataset}")
    if verified.get("input_sha256") != _input_hash(bundle_loader(dataset)):
        raise ValueError(f"cache input hash mismatch: source={source} dataset={dataset}")
    stored_contract = verified.get("run_contract")
    if not isinstance(stored_contract, Mapping) or _run_contract_hash(stored_contract) != current_contract["run_contract_sha256"]:
        raise ValueError(f"cache run-contract content mismatch: source={source}")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return verified, arrays


def candidate_indices(train_oof: np.ndarray, test_scores: np.ndarray, alpha: float) -> tuple[int, ...]:
    return tuple(empirical_candidates(train_oof, test_scores, float(alpha)).indices)


def _prediction_for_variant(
    *, metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray], variant: str, alpha: float,
) -> tuple[tuple[int, ...], str, bool, float | None]:
    if bool(metadata["abstained"]):
        return (), stable_json_hash({"abstained": metadata.get("abstention_reason")}), False, None
    if variant == "seed_intersection":
        seeds = sorted(key.removeprefix("train_oof__seed_") for key in arrays if key.startswith("train_oof__seed_"))
        if len(seeds) != 2 or any(f"test_scores__seed_{seed}" not in arrays for seed in seeds):
            raise ValueError("TCN intersection requires exactly two cached seed score pairs")
        results = [empirical_candidates(arrays[f"train_oof__seed_{seed}"], arrays[f"test_scores__seed_{seed}"], alpha) for seed in seeds]
        indices = tuple(sorted(set(results[0].indices).intersection(results[1].indices)))
        return indices, stable_json_hash({"seed_pvalues": [_array_hash(result.pvalues) for result in results]}), any(result.resolution_blocked for result in results), max(result.attainable_p for result in results)
    result = empirical_candidates(arrays["train_oof"], arrays["test_scores"], alpha)
    return tuple(result.indices), _array_hash(result.pvalues), bool(result.resolution_blocked), float(result.attainable_p)


def _prediction_dict(prediction: FrozenSourcePrediction, *, pvalue_hash: str, resolution_blocked: bool, attainable_p: float | None, evidence_family: str, evidence_identity: str) -> dict[str, object]:
    row = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "dataset": prediction.dataset,
        "source": prediction.source,
        "variant": prediction.variant,
        "alpha": prediction.alpha,
        "candidate_indices": list(prediction.candidate_indices),
        "score_hash": prediction.score_hash,
        "pvalue_hash": pvalue_hash,
        "eligibility": prediction.eligibility,
        "abstention_reason": prediction.abstention_reason,
        "config_hash": prediction.config_hash,
        "resolution_blocked": resolution_blocked,
        "attainable_p": attainable_p,
        "calibration_reference": "query_excluded_train_oof",
        "evidence_family": evidence_family,
        "evidence_identity": evidence_identity,
        "contains_test_labels": False,
    }
    if set(row).intersection(_FORBIDDEN_PREDICTION_FIELDS):
        raise AssertionError("prediction row leaked an evaluation field")
    return row


def _assert_source_manifest_complete(
    root: Path, source: str, *, expected_dataset_count: int
) -> None:
    """Do not freeze a partial cache that was known to have worker failures."""
    manifest_path = source_root(root, source) / "source_manifest.json"
    if not manifest_path.exists():
        # Direct per-dataset cache construction is useful in unit tests. Full
        # production cache_source always emits this manifest.
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not bool(manifest.get("complete")):
        raise ValueError(f"source cache is incomplete: {source}")
    if int(manifest.get("datasets_expected", -1)) != int(expected_dataset_count):
        raise ValueError(f"source cache coverage differs from prediction contract: {source}")


def _canonical_sources(sources: Iterable[str]) -> tuple[str, ...]:
    supplied = tuple(str(item) for item in sources)
    canonical = tuple(registration(item).source for item in supplied)
    if len(canonical) != len(set(canonical)):
        raise ValueError("prediction source inputs contain duplicate physical sources")
    return canonical


def _runtime_for_source(
    source: str,
    source_runtime: Mapping[str, Mapping[str, object]] | None,
) -> tuple[str, Mapping[str, object] | None]:
    if source_runtime is None:
        return "cpu", None
    entries = [
        value for key, value in source_runtime.items()
        if registration(str(key)).source == source
    ]
    if len(entries) > 1:
        raise ValueError(f"duplicate runtime entries for source={source}")
    if not entries:
        return "cpu", None
    entry = entries[0]
    device = str(entry.get("device", "cpu"))
    benchmark = entry.get("benchmark_record")
    if benchmark is not None and not isinstance(benchmark, Mapping):
        raise TypeError("benchmark_record must be a mapping")
    return device, benchmark


def _expected_prediction_keys(
    sources: tuple[str, ...], datasets: tuple[str, ...]
) -> set[tuple[str, str, str, float]]:
    return {
        (source, item.variant, dataset, float(alpha))
        for source in sources
        for item in source_variants(source)
        for dataset in datasets
        for alpha in ALPHAS
    }


def _key_universe_hash(keys: Iterable[tuple[str, str, str, float]]) -> str:
    return stable_json_hash(tuple(sorted(keys)))


def freeze_source_predictions(
    root: Path,
    *,
    sources: tuple[str, ...] | None = None,
    datasets: tuple[str, ...],
    expected_dataset_count: int = 1117,
    output_dir: Path | None = None,
    bundle_loader: Callable[[str], object] = load_bundle,
    source_runtime: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[Path, Path]:
    """Freeze deterministic candidates after cache verification and before labels."""
    selected_sources = _canonical_sources(sources or source_names())
    selected_datasets = tuple(sorted(dict.fromkeys(str(item) for item in datasets)))
    if len(selected_datasets) != len(tuple(datasets)) or len(selected_datasets) != expected_dataset_count:
        raise ValueError("prediction population coverage contract mismatch")
    expected_keys = _expected_prediction_keys(selected_sources, selected_datasets)
    rows: list[dict[str, object]] = []
    source_contracts: dict[str, str] = {}
    variant_contracts: dict[str, str] = {}
    for source in selected_sources:
        canonical = source
        device, benchmark_record = _runtime_for_source(canonical, source_runtime)
        _assert_source_manifest_complete(
            Path(root), canonical, expected_dataset_count=expected_dataset_count
        )
        for dataset in selected_datasets:
            metadata, arrays = _read_cache(
                Path(root), canonical, dataset, bundle_loader=bundle_loader,
                device=device, benchmark_record=benchmark_record,
            )
            source_contracts.setdefault(canonical, str(metadata["run_contract_sha256"]))
            if source_contracts[canonical] != str(metadata["run_contract_sha256"]):
                raise ValueError(f"source run-contract differs by dataset: source={canonical}")
            cache_variant_contracts = metadata.get("variant_run_contract_sha256")
            if not isinstance(cache_variant_contracts, Mapping):
                raise ValueError(f"missing variant run contracts: source={canonical}")
            for variant_registration in source_variants(canonical):
                variant_key = f"{canonical}:{variant_registration.variant}"
                current_variant_hash = cache_variant_contracts.get(variant_registration.variant)
                if not isinstance(current_variant_hash, str):
                    raise ValueError(f"missing variant contract hash: {variant_key}")
                variant_contracts.setdefault(variant_key, current_variant_hash)
                if variant_contracts[variant_key] != current_variant_hash:
                    raise ValueError(f"variant run-contract differs by dataset: {variant_key}")
                for alpha in ALPHAS:
                    candidates, pvalue_hash, blocked, attainable = _prediction_for_variant(
                        metadata=metadata, arrays=arrays, variant=variant_registration.variant, alpha=float(alpha)
                    )
                    prediction = FrozenSourcePrediction(
                        dataset=dataset,
                        source=canonical,
                        variant=variant_registration.variant,
                        alpha=float(alpha),
                        candidate_indices=candidates,
                        score_hash=str(metadata["cache_sha256"]),
                        eligibility=bool(metadata["eligible"]),
                        abstention_reason=metadata.get("abstention_reason") if isinstance(metadata.get("abstention_reason"), str) else None,
                        config_hash=str(metadata["config_hash"]),
                    )
                    row = _prediction_dict(
                        prediction, pvalue_hash=pvalue_hash, resolution_blocked=blocked,
                        attainable_p=attainable, evidence_family=variant_registration.family,
                        evidence_identity=variant_registration.evidence_identity,
                    )
                    row["run_contract_sha256"] = current_variant_hash
                    rows.append(row)
    rows.sort(key=lambda row: (str(row["source"]), str(row["variant"]), str(row["dataset"]), float(row["alpha"])))
    actual_keys = [
        (str(row["source"]), str(row["variant"]), str(row["dataset"]), float(row["alpha"]))
        for row in rows
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("prediction freeze attempted duplicate row keys")
    if set(actual_keys) != expected_keys:
        raise ValueError("prediction freeze row universe mismatch")
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    destination = Path(output_dir) if output_dir is not None else Path(root) / "frozen_predictions"
    prediction_path = destination / "frozen_source_predictions.jsonl"
    manifest_path = destination / "source_prediction_manifest.json"
    _atomic_bytes(prediction_path, payload)
    manifest = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_file": prediction_path.name,
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "sources": selected_sources,
        "datasets": len(selected_datasets),
        "dataset_list_sha256": stable_json_hash(selected_datasets),
        "key_universe_sha256": _key_universe_hash(expected_keys),
        "source_run_contract_sha256": source_contracts,
        "variant_run_contract_sha256": variant_contracts,
        "canonical_order": ("source", "variant", "dataset", "alpha"),
        "contains_test_labels": False,
    }
    _atomic_json(manifest_path, manifest)
    return prediction_path, manifest_path


def _default_frozen_population() -> tuple[str, ...]:
    from .population import load_frozen_population

    return tuple(load_frozen_population())


def _validate_prediction_row(
    row: Mapping[str, object],
    *,
    expected_key: tuple[str, str, str, float],
    n_test: int,
    expected_variant_contract: str,
) -> None:
    source, variant, dataset, alpha = expected_key
    if set(row).intersection(_FORBIDDEN_PREDICTION_FIELDS):
        raise ValueError("frozen prediction carries forbidden evaluation fields")
    if (
        row.get("source") != source
        or row.get("variant") != variant
        or row.get("dataset") != dataset
        or type(row.get("alpha")) not in {int, float}
        or float(row["alpha"]) != alpha
        or float(row["alpha"]) not in ALPHAS
        or row.get("contains_test_labels") is not False
    ):
        raise ValueError("prediction row identity or alpha contract mismatch")
    registration_item = next(item for item in source_variants(source) if item.variant == variant)
    if (
        row.get("evidence_family") != registration_item.family
        or row.get("evidence_identity") != registration_item.evidence_identity
        or row.get("run_contract_sha256") != expected_variant_contract
    ):
        raise ValueError("prediction row source/family/run-contract mismatch")
    indices = row.get("candidate_indices")
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise ValueError("candidate indices must be an integer list")
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("candidate indices must be sorted and unique")
    if any(index < 0 or index >= n_test for index in indices):
        raise ValueError("candidate index outside TEST score range")


def verify_prediction_hash(
    prediction_path: Path,
    manifest_path: Path,
    *,
    expected_datasets: tuple[str, ...] | None = None,
    expected_sources: tuple[str, ...] | None = None,
    cache_root: Path | None = None,
    bundle_loader: Callable[[str], object] = load_bundle,
    source_runtime: Mapping[str, Mapping[str, object]] | None = None,
) -> bool:
    """Verify immutable prediction bytes, universe, contracts, and cached candidates.

    The expected population comes from the frozen Exp266 population by default.
    Test fixtures must pass their own population explicitly; a rewritten
    prediction manifest is never accepted as the source of truth.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("prediction manifest must be an object")
    payload = Path(prediction_path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != manifest.get("prediction_sha256"):
        raise ValueError("frozen prediction SHA256 mismatch")
    try:
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen prediction is not valid JSONL") from exc
    canonical = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    if canonical != payload:
        raise ValueError("frozen prediction JSONL is not canonical")

    supplied_datasets = tuple(expected_datasets) if expected_datasets is not None else _default_frozen_population()
    datasets = tuple(sorted(dict.fromkeys(supplied_datasets)))
    if not datasets or len(datasets) != len(supplied_datasets):
        raise ValueError("expected frozen population is empty or contains duplicates")
    sources = _canonical_sources(expected_sources or source_names())
    expected_keys = _expected_prediction_keys(sources, datasets)
    actual_keys: list[tuple[str, str, str, float]] = []
    for row in rows:
        try:
            actual_keys.append((
                str(row["source"]), str(row["variant"]), str(row["dataset"]), float(row["alpha"])
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("prediction row has no valid key") from exc
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("frozen prediction has duplicate row keys")
    if set(actual_keys) != expected_keys:
        raise ValueError("frozen prediction key universe mismatch")
    if len(rows) != len(expected_keys) or len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("frozen prediction row count mismatch")
    if (
        manifest.get("dataset_list_sha256") != stable_json_hash(datasets)
        or manifest.get("key_universe_sha256") != _key_universe_hash(expected_keys)
        or tuple(manifest.get("sources", ())) != sources
        or int(manifest.get("datasets", -1)) != len(datasets)
        or manifest.get("contains_test_labels") is not False
    ):
        raise ValueError("prediction manifest population or universe mismatch")

    root = Path(cache_root) if cache_root is not None else Path(prediction_path).parent.parent
    bundles = {dataset: bundle_loader(dataset) for dataset in datasets}
    source_contracts: dict[str, str] = {}
    variant_contracts: dict[str, str] = {}
    cache_rows: dict[tuple[str, str], tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    for source in sources:
        device, benchmark_record = _runtime_for_source(source, source_runtime)
        db_sha = database_hash(DB_PATH)
        contract = _run_contract(
            source, device=device, benchmark_record=benchmark_record, db_sha256=db_sha
        )
        source_contracts[source] = str(contract["run_contract_sha256"])
        for item in source_variants(source):
            variant_contracts[f"{source}:{item.variant}"] = _variant_run_contract_hashes(contract)[item.variant]
        for dataset in datasets:
            cache_rows[(source, dataset)] = _read_cache(
                root, source, dataset, bundle_loader=bundle_loader,
                device=device, benchmark_record=benchmark_record,
            )
    if (
        manifest.get("source_run_contract_sha256") != source_contracts
        or manifest.get("variant_run_contract_sha256") != variant_contracts
    ):
        raise ValueError("prediction manifest run-contract mismatch")

    for row, key in zip(rows, actual_keys, strict=True):
        source, variant, dataset, alpha = key
        metadata, arrays = cache_rows[(source, dataset)]
        _validate_prediction_row(
            row, expected_key=key, n_test=len(np.asarray(bundles[dataset].test_values)),
            expected_variant_contract=variant_contracts[f"{source}:{variant}"],
        )
        candidates, pvalue_hash, blocked, attainable = _prediction_for_variant(
            metadata=metadata, arrays=arrays, variant=variant, alpha=alpha
        )
        expected_attainable = attainable
        if (
            row.get("candidate_indices") != list(candidates)
            or row.get("pvalue_hash") != pvalue_hash
            or row.get("score_hash") != metadata.get("cache_sha256")
            or row.get("config_hash") != metadata.get("config_hash")
            or row.get("eligibility") != bool(metadata.get("eligible"))
            or row.get("abstention_reason") != metadata.get("abstention_reason")
            or row.get("resolution_blocked") != blocked
            or row.get("attainable_p") != expected_attainable
        ):
            raise ValueError("prediction row disagrees with frozen source cache")
    sorted_rows = sorted(
        rows,
        key=lambda row: (str(row["source"]), str(row["variant"]), str(row["dataset"]), float(row["alpha"])),
    )
    if rows != sorted_rows:
        raise ValueError("frozen prediction rows are not canonically ordered")
    return True
