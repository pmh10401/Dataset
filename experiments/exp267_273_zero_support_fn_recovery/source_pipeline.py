"""Resumable, label-free cache and prediction freeze for Exp268-270.

The cache boundary is intentionally before evaluation: it may read aligned TEST
values to score them, but it neither opens nor carries TEST labels.  Every
cache unit is self-verifying, so a partial crash or an edited artifact is
rebuilt rather than silently reused.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates
from experiments.exp239_248_diverse_score_sources.foundation import (
    database_hash,
    load_bundle,
    stable_json_hash,
)

from .config import ALPHAS, DB_PATH, SEED
from .contracts import FrozenSourcePrediction, SourceScore
from .source_registry import build_source, registration, source_names, source_variants


CACHE_SCHEMA_VERSION = 1
PREDICTION_SCHEMA_VERSION = 1
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
        benchmark_hash = stable_json_hash(dict(benchmark_record))
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
) -> dict[str, object]:
    """Build or verify one atomic label-free source cache unit."""
    canonical = registration(source).source
    if canonical == "tcn_forecast" and benchmark_record is None:
        raise ValueError("TCN cache requires one frozen verified benchmark record")
    archive_path, metadata_path = cache_paths(Path(root), canonical, dataset)
    config = _source_config(canonical, device=device, benchmark_record=benchmark_record)
    expected_base = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "config_hash": stable_json_hash(config),
        "db_sha256": db_hash or database_hash(DB_PATH),
        "code_sha256": code_hash or _source_code_hash(canonical),
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
    root, source, dataset, device, reset, benchmark_record, db_hash, code_hash = arguments
    return build_cache_unit(
        Path(str(root)), str(source), str(dataset), device=str(device), reset=bool(reset),
        benchmark_record=benchmark_record if isinstance(benchmark_record, Mapping) else None,
        db_hash=str(db_hash), code_hash=str(code_hash),
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
    worker_count = 1 if item.model_based else min(7, max(1, int(workers)))
    base = source_root(Path(root), canonical)
    db_sha = database_hash(DB_PATH)
    code_sha = _source_code_hash(canonical)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    def finish(done: int, dataset: str) -> None:
        if progress is not None:
            progress(done, len(selected), dataset, len(errors))

    if worker_count == 1:
        for done, dataset in enumerate(selected, 1):
            try:
                rows.append(build_cache_unit(
                    Path(root), canonical, dataset, device=device, reset=reset,
                    benchmark_record=benchmark_record, db_hash=db_sha, code_hash=code_sha,
                ))
            except Exception as error:  # persist failures and block prediction freeze.
                errors.append({"dataset": dataset, "error_type": type(error).__name__, "error": str(error)})
            finish(done, dataset)
    else:
        arguments = [
            (str(root), canonical, dataset, device, reset, None, db_sha, code_sha)
            for dataset in selected
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
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
        "contains_test_labels": False,
    })
    return rows


def _read_cache(
    root: Path,
    source: str,
    dataset: str,
    *,
    bundle_loader: Callable[[str], object] = load_bundle,
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
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "config_hash": metadata.get("config_hash"),
        "db_sha256": database_hash(DB_PATH),
        "code_sha256": _source_code_hash(source),
    }
    verified = _cache_is_valid(archive_path, metadata_path, expected=expected)
    if verified is None:
        raise ValueError(f"invalid or stale cache: source={source} dataset={dataset}")
    if verified.get("input_sha256") != _input_hash(bundle_loader(dataset)):
        raise ValueError(f"cache input hash mismatch: source={source} dataset={dataset}")
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


def freeze_source_predictions(
    root: Path,
    *,
    sources: tuple[str, ...] | None = None,
    datasets: tuple[str, ...],
    expected_dataset_count: int = 1117,
    output_dir: Path | None = None,
    bundle_loader: Callable[[str], object] = load_bundle,
) -> tuple[Path, Path]:
    """Freeze deterministic candidates after cache verification and before labels."""
    selected_sources = tuple(sources or source_names())
    selected_datasets = tuple(sorted(dict.fromkeys(str(item) for item in datasets)))
    if len(selected_datasets) != len(tuple(datasets)) or len(selected_datasets) != expected_dataset_count:
        raise ValueError("prediction population coverage contract mismatch")
    rows: list[dict[str, object]] = []
    for source in selected_sources:
        canonical = registration(source).source
        _assert_source_manifest_complete(
            Path(root), canonical, expected_dataset_count=expected_dataset_count
        )
        for dataset in selected_datasets:
            metadata, arrays = _read_cache(
                Path(root), canonical, dataset, bundle_loader=bundle_loader
            )
            for variant_registration in source_variants(canonical):
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
                    rows.append(_prediction_dict(
                        prediction, pvalue_hash=pvalue_hash, resolution_blocked=blocked,
                        attainable_p=attainable, evidence_family=variant_registration.family,
                        evidence_identity=variant_registration.evidence_identity,
                    ))
    rows.sort(key=lambda row: (str(row["source"]), str(row["variant"]), str(row["dataset"]), float(row["alpha"])))
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
        "sources": tuple(sorted(registration(item).source for item in selected_sources)),
        "datasets": len(selected_datasets),
        "dataset_list_sha256": stable_json_hash(selected_datasets),
        "canonical_order": ("source", "variant", "dataset", "alpha"),
        "contains_test_labels": False,
    }
    _atomic_json(manifest_path, manifest)
    return prediction_path, manifest_path


def verify_prediction_hash(prediction_path: Path, manifest_path: Path) -> bool:
    """Refuse byte or content tampering before downstream evaluation opens labels."""
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
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("frozen prediction row count mismatch")
    if any(set(row).intersection(_FORBIDDEN_PREDICTION_FIELDS) for row in rows):
        raise ValueError("frozen prediction carries forbidden evaluation fields")
    sorted_rows = sorted(rows, key=lambda row: (str(row["source"]), str(row["variant"]), str(row["dataset"]), float(row["alpha"])))
    if rows != sorted_rows:
        raise ValueError("frozen prediction rows are not canonically ordered")
    return True
