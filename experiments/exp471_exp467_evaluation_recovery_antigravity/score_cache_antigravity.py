"""Label-free, resumable preparation of deterministic Exp471 score shards."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from paper.equipment_anomaly_detection.numba_calibration_antigravity import (
    upper_tail_pvalues_numba_antigravity,
)

from . import cache_artifacts_antigravity as _artifacts_antigravity
from .cache_artifacts_antigravity import (
    publish_cache_manifest_antigravity,
    publish_score_shard_antigravity,
    read_cache_manifest_antigravity,
    read_score_shard_antigravity,
    score_shard_relative_path_antigravity,
)
from .config_antigravity import (
    ACCELERATION_ASSESSMENT_ANTIGRAVITY,
    CPU_WORKERS_ANTIGRAVITY,
    EXP467_IMPORTED_TOTAL_ANTIGRAVITY,
    EXP467_NATIVE_TOTAL_ANTIGRAVITY,
    EXP467_SOURCE_COMMIT_ANTIGRAVITY,
    EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
    EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
    EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
    INNER_NUMERICAL_THREADS_ANTIGRAVITY,
    LOGICAL_CPU_COUNT_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)
from .contracts_antigravity import (
    Exp471ScoreCacheManifestAntigravity,
    Exp471ScoreShardAntigravity,
    Exp471SourceAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from .source_authority_antigravity import (
    Exp471SourceSessionAntigravity,
    _source_session_is_issued_antigravity,
)


_INPUT_SCHEMA_ANTIGRAVITY = "exp471-verified-score-input-antigravity.v2"
_STATUS_SCHEMA_ANTIGRAVITY = "exp471-score-cache-status-antigravity.v1"
_BENCHMARK_SCHEMA_ANTIGRAVITY = "exp471-score-preparation-benchmark-antigravity.v1"
_BASELINE_AUTHORITY_SCHEMA_ANTIGRAVITY = (
    "exp471-verified-baseline-authority-antigravity.v1"
)
_BENCHMARK_NAME_ANTIGRAVITY = "score_preparation_benchmark_antigravity.json"
_SPEEDUP_THRESHOLD_ANTIGRAVITY = 1.1
_monotonic_antigravity = time.perf_counter


@dataclass(frozen=True, slots=True)
class _ScoreTaskAntigravity:
    source_authority_sha256: str
    cache_context_sha256: str
    method: str
    dataset: str
    test_count: int
    raw_scores: tuple[float, ...]
    micro_scores: tuple[float, ...]
    source_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _TaskOutcomeAntigravity:
    task: _ScoreTaskAntigravity
    shard: Exp471ScoreShardAntigravity | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class _TestCoordinatesAntigravity:
    count: int
    indices: tuple[int, ...]
    coordinates_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedScoreMetadataAntigravity:
    source_authority_sha256: str
    database_sha256: str
    cache_context_sha256: str
    dataset_order: tuple[str, ...]
    coordinates: Mapping[str, _TestCoordinatesAntigravity]
    coordinates_sha256: str
    source_manifest_sha256: Mapping[str, str]
    baseline_authority_sha256: Mapping[str, str]
    condition_scores: Mapping[str, Mapping[str, tuple[float, ...]]]
    baseline_scores: Mapping[
        str,
        Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]],
    ]


def _digest_antigravity(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _mapping_antigravity(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _float64_vector_antigravity(
    value: object, field_name: str, *, nonempty: bool = True
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a finite float64 vector")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite float64 vector") from error
    if array.ndim != 1 or (nonempty and array.size == 0) or not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be a finite float64 vector")
    return tuple(float(item) for item in array)


def _cache_context_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    datasets: tuple[str, ...],
    coordinates_sha256: str,
    source_digests: Mapping[str, str],
    baseline_authority_digests: Mapping[str, str],
) -> str:
    record = {
        "schema": "exp471-score-cache-context-antigravity.v1",
        "source_authority_sha256": authority.authority_sha256,
        "database_sha256": authority.database_sha256,
        "method_order": list(METHOD_ORDER_ANTIGRAVITY),
        "dataset_order": list(datasets),
        "test_coordinates_sha256": coordinates_sha256,
        "source_manifest_sha256": dict(source_digests),
        "baseline_authority_sha256": dict(baseline_authority_digests),
        "calibration": {
            "function": "upper_tail_pvalues_numba_antigravity",
            "dtype": "float64",
            "fastmath": False,
            "parallel": False,
            "nogil": True,
        },
    }
    return sha256(canonical_json_bytes_exp471_antigravity(record)).hexdigest()


def _test_coordinates_antigravity(
    value: object, dataset: str
) -> _TestCoordinatesAntigravity:
    record = _mapping_antigravity(value, f"Exp471 {dataset} TEST coordinates")
    if set(record) != {"count", "indices", "coordinates_sha256"}:
        raise ValueError(f"Exp471 {dataset} TEST coordinates schema differs")
    count = record["count"]
    indices_value = record["indices"]
    if (
        type(count) is not int
        or count < 1
        or not isinstance(indices_value, (tuple, list))
        or len(indices_value) != count
        or any(type(index) is not int or index < 0 for index in indices_value)
        or len(set(indices_value)) != count
    ):
        raise ValueError(f"Exp471 {dataset} TEST coordinates are invalid")
    indices = tuple(indices_value)
    digest = _digest_antigravity(
        record["coordinates_sha256"],
        f"Exp471 {dataset} TEST coordinates_sha256",
    )
    expected = sha256(
        canonical_json_bytes_exp471_antigravity(
            {"dataset": dataset, "count": count, "indices": list(indices)}
        )
    ).hexdigest()
    if digest != expected:
        raise ValueError(f"Exp471 {dataset} TEST coordinates digest differs")
    return _TestCoordinatesAntigravity(count, indices, digest)


def _validate_row_coordinates_antigravity(
    row: Mapping[str, object],
    expected: _TestCoordinatesAntigravity,
    *,
    scope: str,
) -> tuple[float, ...]:
    indices = row.get("test_indices")
    if not isinstance(indices, (tuple, list)) or tuple(indices) != expected.indices:
        raise ValueError(f"{scope} TEST coordinates differ")
    scores = _float64_vector_antigravity(
        row.get("test_scores"), f"{scope} TEST scores"
    )
    if len(scores) != expected.count:
        raise ValueError(f"{scope} TEST count differs from common coordinates")
    return scores


def _validate_verified_input_metadata_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    value: object,
) -> _VerifiedScoreMetadataAntigravity:
    """Validate authoritative identities and common TEST rows without calibration."""

    if not isinstance(authority, Exp471SourceAuthorityAntigravity):
        raise ValueError("Exp471 score input authority has the wrong type")
    bundle = _mapping_antigravity(value, "Exp471 verified score input")
    if set(bundle) != {
        "schema",
        "database_sha256",
        "dataset_order",
        "test_coordinates",
        "baseline_authority_sha256",
        "conditions",
        "baselines",
    }:
        raise ValueError("Exp471 verified score input schema differs")
    if bundle["schema"] != _INPUT_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp471 verified score input schema differs")
    database_sha256 = _digest_antigravity(
        bundle["database_sha256"], "Exp471 score input database_sha256"
    )
    if database_sha256 != authority.database_sha256:
        raise ValueError("Exp471 score input database identity differs")
    dataset_value = bundle["dataset_order"]
    if not isinstance(dataset_value, (tuple, list)) or not dataset_value:
        raise ValueError("Exp471 dataset order must be a non-empty sequence")
    datasets = tuple(dataset_value)
    if (
        any(type(dataset) is not str or not dataset for dataset in datasets)
        or len(datasets) != len(set(datasets))
    ):
        raise ValueError("Exp471 dataset order is invalid")

    coordinate_rows = _mapping_antigravity(
        bundle["test_coordinates"], "Exp471 TEST coordinates"
    )
    if tuple(coordinate_rows) != datasets:
        raise ValueError("Exp471 TEST coordinate dataset order differs")
    coordinates = {
        dataset: _test_coordinates_antigravity(coordinate_rows[dataset], dataset)
        for dataset in datasets
    }
    normalized_coordinates = {
        dataset: {
            "count": coordinates[dataset].count,
            "indices": list(coordinates[dataset].indices),
            "coordinates_sha256": coordinates[dataset].coordinates_sha256,
        }
        for dataset in datasets
    }
    coordinates_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(normalized_coordinates)
    ).hexdigest()

    conditions = _mapping_antigravity(bundle["conditions"], "Exp471 conditions")
    baselines = _mapping_antigravity(bundle["baselines"], "Exp471 baselines")
    baseline_authority_inventory = _mapping_antigravity(
        bundle["baseline_authority_sha256"],
        "Exp471 baseline authority inventory",
    )
    if tuple(conditions) != METHOD_ORDER_ANTIGRAVITY[:4]:
        raise ValueError("Exp471 condition method order differs")
    if tuple(baselines) != METHOD_ORDER_ANTIGRAVITY[4:]:
        raise ValueError("Exp471 baseline method order differs")
    if tuple(baseline_authority_inventory) != METHOD_ORDER_ANTIGRAVITY[4:]:
        raise ValueError("Exp471 baseline authority method order differs")

    source_digests: dict[str, str] = {}
    condition_scores: dict[str, dict[str, tuple[float, ...]]] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[:4]:
        row = _mapping_antigravity(
            conditions[method], f"Exp471 {method} condition source"
        )
        if set(row) != {"source_manifest_sha256", "scores"}:
            raise ValueError(f"Exp471 {method} source schema differs")
        source_digest = _digest_antigravity(
            row["source_manifest_sha256"],
            f"Exp471 {method} source_manifest_sha256",
        )
        if source_digest != authority.condition_manifest_sha256[method]:
            raise ValueError(
                f"Exp471 {method} condition manifest authority differs"
            )
        scores = _mapping_antigravity(row["scores"], f"Exp471 {method} scores")
        if tuple(scores) != datasets:
            raise ValueError(f"Exp471 {method} dataset order differs")
        source_digests[method] = source_digest
        condition_scores[method] = {}
        for dataset in datasets:
            vectors = _mapping_antigravity(
                scores[dataset], f"Exp471 {method}/{dataset} score vectors"
            )
            if set(vectors) != {"test_indices", "test_scores"}:
                raise ValueError(
                    f"Exp471 {method}/{dataset} score-vector schema differs"
                )
            condition_scores[method][dataset] = (
                _validate_row_coordinates_antigravity(
                    vectors,
                    coordinates[dataset],
                    scope=f"Exp471 {method}/{dataset}",
                )
            )

    baseline_authority_digests: dict[str, str] = {}
    baseline_scores: dict[
        str, dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
    ] = {}
    for method in METHOD_ORDER_ANTIGRAVITY[4:]:
        row = _mapping_antigravity(
            baselines[method], f"Exp471 {method} baseline source"
        )
        if set(row) != {"authority", "scores"}:
            raise ValueError(f"Exp471 {method} baseline source schema differs")
        baseline_authority = _mapping_antigravity(
            row["authority"], f"Exp471 {method} baseline authority"
        )
        required_authority = {
            "schema",
            "method",
            "database_sha256",
            "dataset_order",
            "test_coordinates_sha256",
            "source_manifest_sha256",
            "authority_sha256",
        }
        if set(baseline_authority) != required_authority:
            raise ValueError(f"Exp471 {method} baseline authority schema differs")
        if (
            baseline_authority["schema"] != _BASELINE_AUTHORITY_SCHEMA_ANTIGRAVITY
            or baseline_authority["method"] != method
        ):
            raise ValueError(f"Exp471 {method} baseline authority differs")
        baseline_database = _digest_antigravity(
            baseline_authority["database_sha256"],
            f"Exp471 {method} baseline database_sha256",
        )
        if baseline_database != database_sha256:
            raise ValueError(f"Exp471 {method} baseline database identity differs")
        if tuple(baseline_authority["dataset_order"]) != datasets:
            raise ValueError(f"Exp471 {method} baseline dataset order differs")
        if baseline_authority["test_coordinates_sha256"] != coordinates_sha256:
            raise ValueError(f"Exp471 {method} baseline TEST coordinates differ")
        source_digest = _digest_antigravity(
            baseline_authority["source_manifest_sha256"],
            f"Exp471 {method} baseline source_manifest_sha256",
        )
        authority_digest = _digest_antigravity(
            baseline_authority["authority_sha256"],
            f"Exp471 {method} baseline authority_sha256",
        )
        inventory_digest = _digest_antigravity(
            baseline_authority_inventory[method],
            f"Exp471 {method} baseline authority inventory digest",
        )
        bare = {
            key: baseline_authority[key]
            for key in baseline_authority
            if key != "authority_sha256"
        }
        expected_authority_digest = sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest()
        if authority_digest != expected_authority_digest or inventory_digest != authority_digest:
            raise ValueError(f"Exp471 {method} baseline authority differs")
        source_digests[method] = source_digest
        baseline_authority_digests[method] = authority_digest
        scores = _mapping_antigravity(
            row["scores"], f"Exp471 {method} baseline scores"
        )
        if tuple(scores) != datasets:
            raise ValueError(f"Exp471 {method} baseline dataset order differs")
        baseline_scores[method] = {}
        for dataset in datasets:
            vectors = _mapping_antigravity(
                scores[dataset], f"Exp471 {method}/{dataset} score vectors"
            )
            if set(vectors) != {
                "train_oof_scores",
                "test_indices",
                "test_scores",
            }:
                raise ValueError(
                    f"Exp471 {method}/{dataset} score-vector schema differs"
                )
            test_scores = _validate_row_coordinates_antigravity(
                vectors,
                coordinates[dataset],
                scope=f"Exp471 {method}/{dataset}",
            )
            train_scores = _float64_vector_antigravity(
                vectors["train_oof_scores"],
                f"Exp471 {method}/{dataset} TRAIN OOF scores",
            )
            baseline_scores[method][dataset] = (train_scores, test_scores)

    cache_context = _cache_context_antigravity(
        authority,
        datasets,
        coordinates_sha256,
        source_digests,
        baseline_authority_digests,
    )
    return _VerifiedScoreMetadataAntigravity(
        authority.authority_sha256,
        database_sha256,
        cache_context,
        datasets,
        coordinates,
        coordinates_sha256,
        source_digests,
        baseline_authority_digests,
        condition_scores,
        baseline_scores,
    )


def _prepare_score_identity_antigravity(
    metadata: _VerifiedScoreMetadataAntigravity,
    method: str,
    dataset: str,
) -> _ScoreTaskAntigravity:
    """Construct one missing task; this is the only calibration seam."""

    if method in METHOD_ORDER_ANTIGRAVITY[:4]:
        raw_scores = metadata.condition_scores[method][dataset]
        micro_scores = raw_scores
    else:
        train_scores, raw_scores = metadata.baseline_scores[method][dataset]
        train = np.ascontiguousarray(train_scores, dtype=np.float64)
        test = np.ascontiguousarray(raw_scores, dtype=np.float64)
        calibrated = 1.0 - upper_tail_pvalues_numba_antigravity(train, test)
        if calibrated.dtype != np.dtype(np.float64) or not np.isfinite(
            calibrated
        ).all():
            raise ValueError(
                f"Exp471 {method}/{dataset} micro scores must be finite float64"
            )
        micro_scores = tuple(float(item) for item in calibrated)
    return _ScoreTaskAntigravity(
        source_authority_sha256=metadata.source_authority_sha256,
        cache_context_sha256=metadata.cache_context_sha256,
        method=method,
        dataset=dataset,
        test_count=metadata.coordinates[dataset].count,
        raw_scores=raw_scores,
        micro_scores=micro_scores,
        source_manifest_sha256=metadata.source_manifest_sha256[method],
    )


def _build_tasks_from_verified_input_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    value: object,
) -> tuple[_ScoreTaskAntigravity, ...]:
    """Build all public synthetic tasks after authoritative metadata validation."""

    metadata = _validate_verified_input_metadata_antigravity(authority, value)
    return tuple(
        _prepare_score_identity_antigravity(metadata, method, dataset)
        for method in METHOD_ORDER_ANTIGRAVITY
        for dataset in metadata.dataset_order
    )


def _shard_from_task_antigravity(
    task: _ScoreTaskAntigravity,
) -> Exp471ScoreShardAntigravity:
    payload: dict[str, object] = {
        "source_authority_sha256": task.source_authority_sha256,
        "cache_context_sha256": task.cache_context_sha256,
        "method": task.method,
        "dataset": task.dataset,
        "test_count": task.test_count,
        "raw_scores": list(task.raw_scores),
        "micro_scores": list(task.micro_scores),
        "source_manifest_sha256": task.source_manifest_sha256,
    }
    payload_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(payload)
    ).hexdigest()
    envelope = {**payload, "payload_sha256": payload_sha256}
    envelope_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(envelope)
    ).hexdigest()
    return Exp471ScoreShardAntigravity(
        **payload,
        payload_sha256=payload_sha256,
        envelope_sha256=envelope_sha256,
    )


def _prepare_task_antigravity(
    task: _ScoreTaskAntigravity,
) -> Exp471ScoreShardAntigravity:
    return _shard_from_task_antigravity(task)


def _capture_task_antigravity(task: _ScoreTaskAntigravity) -> _TaskOutcomeAntigravity:
    try:
        return _TaskOutcomeAntigravity(
            task, _prepare_task_antigravity(task), None, None
        )
    except Exception as error:
        return _TaskOutcomeAntigravity(
            task, None, type(error).__name__, str(error)
        )


def _run_tasks_antigravity(
    tasks: Sequence[_ScoreTaskAntigravity], workers: int
) -> tuple[_TaskOutcomeAntigravity, ...]:
    if workers == 1 or len(tasks) < 2:
        return tuple(_capture_task_antigravity(task) for task in tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(_capture_task_antigravity, tasks, chunksize=1))


def _shard_bytes_antigravity(
    tasks: Sequence[_ScoreTaskAntigravity], workers: int
) -> tuple[bytes, ...]:
    outcomes = _run_tasks_antigravity(tasks, workers)
    failures = [outcome for outcome in outcomes if outcome.shard is None]
    if failures:
        failure = failures[0]
        raise RuntimeError(
            failure.error_message or failure.error_type or "score task failed"
        )
    return tuple(
        _artifacts_antigravity._canonical_shard_bytes_antigravity(outcome.shard)
        for outcome in outcomes
        if outcome.shard is not None
    )


def _finalize_benchmark_decision_antigravity(
    record: Mapping[str, object],
) -> dict[str, object]:
    bare = dict(record)
    bare.pop("decision_sha256", None)
    decision_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(bare)
    ).hexdigest()
    return {**bare, "decision_sha256": decision_sha256}


def _single_worker_decision_antigravity(
    cache_context_sha256: str,
    *,
    requested_workers: int,
    fallback_reason: str,
) -> dict[str, object]:
    return _finalize_benchmark_decision_antigravity(
        {
            "schema": _BENCHMARK_SCHEMA_ANTIGRAVITY,
            "cache_context_sha256": cache_context_sha256,
            "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
            "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
            "requested_parallel_workers": requested_workers,
            "selected_workers": 1,
            "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
            "single_worker_seconds": 0.0,
            "parallel_seconds": 0.0,
            "speedup_threshold": _SPEEDUP_THRESHOLD_ANTIGRAVITY,
            "speedup": 1.0,
            "byte_parity": True,
            "parallel_accepted": False,
            "parallel_error_type": None,
            "parallel_error_message": None,
            "fallback_reason": fallback_reason,
            "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
        }
    )


def benchmark_score_preparation_antigravity(
    tasks: Sequence[object], *, parallel_workers: int
) -> Mapping[str, object]:
    """Measure deterministic worker parity and select parallel only past 1.1x."""

    if type(parallel_workers) is not int or parallel_workers < 1:
        raise ValueError("Exp471 parallel_workers must be a positive integer")
    if parallel_workers > CPU_WORKERS_ANTIGRAVITY:
        raise ValueError("Exp471 parallel_workers exceed the CPU worker policy")
    typed_tasks = tuple(tasks)
    if not typed_tasks or any(
        not isinstance(task, _ScoreTaskAntigravity) for task in typed_tasks
    ):
        raise ValueError("Exp471 benchmark tasks must be non-empty score tasks")
    cache_context_sha256 = typed_tasks[0].cache_context_sha256
    if any(
        task.cache_context_sha256 != cache_context_sha256
        for task in typed_tasks
    ):
        raise ValueError("Exp471 benchmark task cache contexts differ")

    start = _monotonic_antigravity()
    single_bytes = _shard_bytes_antigravity(typed_tasks, 1)
    single_seconds = round(_monotonic_antigravity() - start, 12)
    start = _monotonic_antigravity()
    parallel_error_type: str | None = None
    parallel_error_message: str | None = None
    try:
        parallel_bytes = _shard_bytes_antigravity(
            typed_tasks, parallel_workers
        )
    except Exception as error:
        parallel_bytes = ()
        parallel_error_type = type(error).__name__
        parallel_error_message = str(error)
    parallel_seconds = round(_monotonic_antigravity() - start, 12)
    byte_parity = bool(
        parallel_error_type is None and single_bytes == parallel_bytes
    )
    speedup = (
        1.0
        if parallel_seconds <= 0.0
        else single_seconds / parallel_seconds
    )
    parallel_accepted = bool(
        parallel_workers > 1
        and byte_parity
        and speedup >= _SPEEDUP_THRESHOLD_ANTIGRAVITY
    )
    if parallel_error_type is not None:
        fallback_reason = (
            f"parallel smoke failed with {parallel_error_type}: "
            f"{parallel_error_message}"
        )
    elif parallel_workers == 1:
        fallback_reason = "parallel worker request is one; canonical CPU single-worker retained"
    elif not byte_parity:
        fallback_reason = "parallel shard bytes differ from canonical CPU single-worker bytes"
    elif parallel_seconds <= 0.0:
        fallback_reason = "parallel elapsed time is below measurable clock resolution"
    elif speedup < _SPEEDUP_THRESHOLD_ANTIGRAVITY:
        fallback_reason = "parallel speedup is below the required 1.1x gate"
    else:
        fallback_reason = ""
    return _finalize_benchmark_decision_antigravity(
        {
            "schema": _BENCHMARK_SCHEMA_ANTIGRAVITY,
            "cache_context_sha256": cache_context_sha256,
            "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
            "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
            "requested_parallel_workers": parallel_workers,
            "selected_workers": parallel_workers if parallel_accepted else 1,
            "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
            "single_worker_seconds": single_seconds,
            "parallel_seconds": parallel_seconds,
            "speedup_threshold": _SPEEDUP_THRESHOLD_ANTIGRAVITY,
            "speedup": speedup,
            "byte_parity": byte_parity,
            "parallel_accepted": parallel_accepted,
            "parallel_error_type": parallel_error_type,
            "parallel_error_message": parallel_error_message,
            "fallback_reason": fallback_reason,
            "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
        }
    )


def _validated_benchmark_decision_antigravity(
    value: object,
    *,
    expected_context_sha256: str | None = None,
) -> dict[str, object]:
    decision = _mapping_antigravity(value, "Exp471 benchmark decision")
    required = {
        "schema",
        "cache_context_sha256",
        "detected_logical_cpus",
        "default_cpu_workers",
        "requested_parallel_workers",
        "selected_workers",
        "inner_numerical_threads",
        "single_worker_seconds",
        "parallel_seconds",
        "speedup_threshold",
        "speedup",
        "byte_parity",
        "parallel_accepted",
        "parallel_error_type",
        "parallel_error_message",
        "fallback_reason",
        "acceleration",
        "decision_sha256",
    }
    if set(decision) != required or decision["schema"] != _BENCHMARK_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp471 benchmark decision schema differs")
    context = _digest_antigravity(
        decision["cache_context_sha256"],
        "Exp471 benchmark cache_context_sha256",
    )
    if expected_context_sha256 is not None and context != expected_context_sha256:
        raise ValueError("Exp471 benchmark decision cache context differs")
    integer_fields = (
        "detected_logical_cpus",
        "default_cpu_workers",
        "requested_parallel_workers",
        "selected_workers",
        "inner_numerical_threads",
    )
    if any(type(decision[field]) is not int or decision[field] < 1 for field in integer_fields):
        raise ValueError("Exp471 benchmark worker provenance is invalid")
    if (
        decision["detected_logical_cpus"] != LOGICAL_CPU_COUNT_ANTIGRAVITY
        or decision["default_cpu_workers"] != CPU_WORKERS_ANTIGRAVITY
        or decision["inner_numerical_threads"]
        != INNER_NUMERICAL_THREADS_ANTIGRAVITY
        or decision["requested_parallel_workers"] > CPU_WORKERS_ANTIGRAVITY
        or decision["selected_workers"]
        not in (1, decision["requested_parallel_workers"])
    ):
        raise ValueError("Exp471 benchmark worker provenance differs")
    for field in (
        "single_worker_seconds",
        "parallel_seconds",
        "speedup_threshold",
        "speedup",
    ):
        number = decision[field]
        if type(number) not in (int, float) or not math.isfinite(float(number)) or float(number) < 0.0:
            raise ValueError("Exp471 benchmark numeric provenance is invalid")
    if float(decision["speedup_threshold"]) != _SPEEDUP_THRESHOLD_ANTIGRAVITY:
        raise ValueError("Exp471 benchmark speedup threshold differs")
    if type(decision["byte_parity"]) is not bool or type(decision["parallel_accepted"]) is not bool:
        raise ValueError("Exp471 benchmark decision flags are invalid")
    error_type = decision["parallel_error_type"]
    error_message = decision["parallel_error_message"]
    if (error_type is None) != (error_message is None) or (
        error_type is not None
        and (type(error_type) is not str or type(error_message) is not str)
    ):
        raise ValueError("Exp471 benchmark error provenance is invalid")
    if type(decision["fallback_reason"]) is not str:
        raise ValueError("Exp471 benchmark fallback reason is invalid")
    if decision["acceleration"] != dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY):
        raise ValueError("Exp471 benchmark acceleration assessment differs")
    accepted = decision["parallel_accepted"]
    if accepted != bool(
        decision["requested_parallel_workers"] > 1
        and decision["selected_workers"] == decision["requested_parallel_workers"]
        and decision["byte_parity"]
        and float(decision["speedup"])
        >= _SPEEDUP_THRESHOLD_ANTIGRAVITY
        and error_type is None
    ):
        raise ValueError("Exp471 benchmark acceptance decision differs")
    digest = _digest_antigravity(
        decision["decision_sha256"], "Exp471 benchmark decision_sha256"
    )
    bare = {key: decision[key] for key in decision if key != "decision_sha256"}
    if digest != sha256(canonical_json_bytes_exp471_antigravity(bare)).hexdigest():
        raise ValueError("Exp471 benchmark decision digest differs")
    return dict(decision)


def _canonical_benchmark_bytes_antigravity(
    decision: Mapping[str, object],
) -> bytes:
    checked = _validated_benchmark_decision_antigravity(decision)
    return canonical_json_bytes_exp471_antigravity(checked)


def _publish_benchmark_decision_antigravity(
    root: Path, decision: Mapping[str, object]
) -> Path:
    payload = _canonical_benchmark_bytes_antigravity(decision)
    anchor = _artifacts_antigravity._open_absolute_directory_antigravity(
        Path(root), create=True
    )
    try:
        _artifacts_antigravity._publish_bytes_at_antigravity(
            anchor.descriptor,
            _BENCHMARK_NAME_ANTIGRAVITY,
            payload,
            temporary_scope="score_preparation_benchmark",
        )
        anchor.validate()
        return Path(root) / _BENCHMARK_NAME_ANTIGRAVITY
    finally:
        anchor.close()


def _read_benchmark_decision_antigravity(
    root: Path,
    *,
    expected_context_sha256: str | None = None,
) -> dict[str, object]:
    anchor = _artifacts_antigravity._open_absolute_directory_antigravity(
        Path(root), create=False
    )
    try:
        raw = _artifacts_antigravity._read_leaf_at_antigravity(
            anchor.descriptor, _BENCHMARK_NAME_ANTIGRAVITY
        )
        if raw is None:
            raise ValueError("Exp471 benchmark decision is missing")
        try:
            value = json.loads(
                raw.decode("utf-8"),
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {constant}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("Exp471 benchmark decision JSON is malformed") from error
        decision = _validated_benchmark_decision_antigravity(
            value, expected_context_sha256=expected_context_sha256
        )
        if raw != _canonical_benchmark_bytes_antigravity(decision):
            raise ValueError("Exp471 benchmark decision bytes are noncanonical")
        anchor.validate()
        return decision
    finally:
        anchor.close()


def _actual_wave_fallback_decision_antigravity(
    decision: Mapping[str, object],
    error: Exception,
    *,
    wave: str,
) -> dict[str, object]:
    record = dict(decision)
    record.pop("decision_sha256", None)
    record["selected_workers"] = 1
    record["parallel_accepted"] = False
    record["parallel_error_type"] = type(error).__name__
    record["parallel_error_message"] = str(error)
    record["fallback_reason"] = (
        f"actual {wave} wave failed with {type(error).__name__}: {error}; "
        "canonical CPU single-worker selected"
    )
    return _finalize_benchmark_decision_antigravity(record)


def _require_issued_session_antigravity(
    session: object,
) -> Exp471SourceSessionAntigravity:
    if (
        not isinstance(session, Exp471SourceSessionAntigravity)
        or not _source_session_is_issued_antigravity(session)
    ):
        raise ValueError("Exp471 score cache requires an issued source session")
    if not isinstance(session.authority, Exp471SourceAuthorityAntigravity):
        raise ValueError("Exp471 issued source authority has the wrong type")
    return session


def _metadata_matches_shard_antigravity(
    metadata: _VerifiedScoreMetadataAntigravity,
    method: str,
    dataset: str,
    shard: Exp471ScoreShardAntigravity,
) -> bool:
    return (
        shard.source_authority_sha256 == metadata.source_authority_sha256
        and shard.cache_context_sha256 == metadata.cache_context_sha256
        and shard.method == method
        and shard.dataset == dataset
        and shard.test_count == metadata.coordinates[dataset].count
        and shard.source_manifest_sha256
        == metadata.source_manifest_sha256[method]
    )


def _existing_shard_antigravity(
    root: Path,
    metadata: _VerifiedScoreMetadataAntigravity,
    method: str,
    dataset: str,
) -> Exp471ScoreShardAntigravity | None:
    path = root / score_shard_relative_path_antigravity(method, dataset)
    if not os.path.lexists(path):
        return None
    shard = read_score_shard_antigravity(root, method, dataset)
    if not _metadata_matches_shard_antigravity(
        metadata, method, dataset, shard
    ):
        raise ValueError("Exp471 existing score shard differs from verified input")
    return shard


def _manifest_from_shards_antigravity(
    metadata: _VerifiedScoreMetadataAntigravity,
    shards: Mapping[tuple[str, str], Exp471ScoreShardAntigravity],
) -> Exp471ScoreCacheManifestAntigravity:
    datasets = metadata.dataset_order
    required = {
        (method, dataset)
        for method in METHOD_ORDER_ANTIGRAVITY
        for dataset in datasets
    }
    if set(shards) != required:
        raise ValueError("Exp471 cannot publish an incomplete score-cache manifest")
    record: dict[str, object] = {
        "source_authority_sha256": metadata.source_authority_sha256,
        "cache_context_sha256": metadata.cache_context_sha256,
        "method_order": list(METHOD_ORDER_ANTIGRAVITY),
        "dataset_order": list(datasets),
        "shard_sha256": {
            f"{method}:{dataset}": shards[(method, dataset)].envelope_sha256
            for method in METHOD_ORDER_ANTIGRAVITY
            for dataset in datasets
        },
        "total_shards": len(required),
    }
    manifest_sha256 = sha256(
        canonical_json_bytes_exp471_antigravity(record)
    ).hexdigest()
    return Exp471ScoreCacheManifestAntigravity(
        **record, manifest_sha256=manifest_sha256
    )


def _validate_manifest_metadata_antigravity(
    manifest: Exp471ScoreCacheManifestAntigravity,
    metadata: _VerifiedScoreMetadataAntigravity,
) -> None:
    if manifest.source_authority_sha256 != metadata.source_authority_sha256:
        raise ValueError("Exp471 complete score cache authority differs")
    if manifest.cache_context_sha256 != metadata.cache_context_sha256:
        raise ValueError("Exp471 complete score cache context differs")
    if (
        manifest.method_order != METHOD_ORDER_ANTIGRAVITY
        or manifest.dataset_order != metadata.dataset_order
    ):
        raise ValueError("Exp471 complete score cache order differs")


def _run_actual_wave_antigravity(
    tasks: Sequence[_ScoreTaskAntigravity],
    workers: int,
    decision: Mapping[str, object],
    *,
    wave: str,
) -> tuple[tuple[_TaskOutcomeAntigravity, ...], dict[str, object]]:
    try:
        return _run_tasks_antigravity(tasks, workers), dict(decision)
    except Exception as error:
        if workers == 1:
            raise
        fallback = _actual_wave_fallback_decision_antigravity(
            decision, error, wave=wave
        )
        return _run_tasks_antigravity(tasks, 1), fallback


def _progress_antigravity(
    callback: Callable[[Mapping[str, object]], None] | None,
    *,
    outcome: _TaskOutcomeAntigravity,
    wave: str,
    attempt: int,
    status: str,
    durable_completed: int,
) -> None:
    if callback is None:
        return
    callback(
        {
            "schema": "exp471-score-cache-progress-antigravity.v1",
            "method": outcome.task.method,
            "dataset": outcome.task.dataset,
            "wave": wave,
            "attempt": attempt,
            "status": status,
            "durable_completed": durable_completed,
            "error_type": outcome.error_type,
            "error_message": outcome.error_message,
        }
    )


def prepare_score_cache_antigravity(
    session: Exp471SourceSessionAntigravity,
    *,
    workers: int | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> Exp471ScoreCacheManifestAntigravity:
    """Build, resume, retry once, and atomically publish complete score coverage."""

    session = _require_issued_session_antigravity(session)
    if progress is not None and not callable(progress):
        raise ValueError("Exp471 progress callback must be callable")
    selected_request = CPU_WORKERS_ANTIGRAVITY if workers is None else workers
    if type(selected_request) is not int or selected_request < 1:
        raise ValueError("Exp471 workers must be a positive integer")
    if selected_request > CPU_WORKERS_ANTIGRAVITY:
        raise ValueError("Exp471 workers exceed the CPU worker policy")
    if not callable(session.score_reader_antigravity):
        raise ValueError("Exp471 issued session has no score-reader capability")

    verified_input = session.score_reader_antigravity()
    metadata = _validate_verified_input_metadata_antigravity(
        session.authority, verified_input
    )
    root = Path(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
    manifest_path = root / "score_cache_manifest_antigravity.json"
    benchmark_path = root / _BENCHMARK_NAME_ANTIGRAVITY
    if os.path.lexists(manifest_path):
        manifest = read_cache_manifest_antigravity(root)
        _validate_manifest_metadata_antigravity(manifest, metadata)
        _read_benchmark_decision_antigravity(
            root, expected_context_sha256=metadata.cache_context_sha256
        )
        return manifest

    shards: dict[tuple[str, str], Exp471ScoreShardAntigravity] = {}
    missing: list[tuple[str, str]] = []
    for method in METHOD_ORDER_ANTIGRAVITY:
        for dataset in metadata.dataset_order:
            existing = _existing_shard_antigravity(
                root, metadata, method, dataset
            )
            if existing is None:
                missing.append((method, dataset))
            else:
                shards[(method, dataset)] = existing

    if os.path.lexists(benchmark_path) and missing:
        raise ValueError(
            "Exp471 benchmark decision exists before complete shard coverage"
        )
    if not missing:
        if os.path.lexists(benchmark_path):
            decision = _read_benchmark_decision_antigravity(
                root, expected_context_sha256=metadata.cache_context_sha256
            )
        else:
            decision = _single_worker_decision_antigravity(
                metadata.cache_context_sha256,
                requested_workers=selected_request,
                fallback_reason=(
                    "resumed complete shards have no pending benchmark tasks; "
                    "canonical CPU single-worker selected"
                ),
            )
            _publish_benchmark_decision_antigravity(root, decision)
        manifest = _manifest_from_shards_antigravity(metadata, shards)
        publish_cache_manifest_antigravity(root, manifest)
        return read_cache_manifest_antigravity(root)

    tasks = tuple(
        _prepare_score_identity_antigravity(metadata, method, dataset)
        for method, dataset in missing
    )
    if selected_request > 1 and len(tasks) > 1:
        decision = dict(
            benchmark_score_preparation_antigravity(
                tasks, parallel_workers=selected_request
            )
        )
    else:
        decision = _single_worker_decision_antigravity(
            metadata.cache_context_sha256,
            requested_workers=selected_request,
            fallback_reason=(
                "parallel worker request is one; canonical CPU single-worker retained"
                if selected_request == 1
                else "fewer than two pending tasks; canonical CPU single-worker retained"
            ),
        )
    decision = _validated_benchmark_decision_antigravity(
        decision, expected_context_sha256=metadata.cache_context_sha256
    )
    actual_workers = int(decision["selected_workers"])
    first_outcomes, decision = _run_actual_wave_antigravity(
        tasks, actual_workers, decision, wave="first"
    )
    actual_workers = int(decision["selected_workers"])
    failures = tuple(
        outcome for outcome in first_outcomes if outcome.shard is None
    )
    retry_outcomes: tuple[_TaskOutcomeAntigravity, ...] = ()
    if failures:
        retry_tasks = tuple(outcome.task for outcome in failures)
        retry_outcomes, decision = _run_actual_wave_antigravity(
            retry_tasks, actual_workers, decision, wave="retry"
        )

    for outcome in first_outcomes:
        identity = (outcome.task.method, outcome.task.dataset)
        if outcome.shard is None:
            _progress_antigravity(
                progress,
                outcome=outcome,
                wave="first",
                attempt=1,
                status="failed",
                durable_completed=len(shards),
            )
            continue
        publish_score_shard_antigravity(root, outcome.shard)
        reopened = read_score_shard_antigravity(root, *identity)
        if reopened != outcome.shard:
            raise ValueError("Exp471 durable score shard differs after publication")
        shards[identity] = reopened
        _progress_antigravity(
            progress,
            outcome=outcome,
            wave="first",
            attempt=1,
            status="durable",
            durable_completed=len(shards),
        )

    terminal_failure: _TaskOutcomeAntigravity | None = None
    if retry_outcomes:
        for outcome in retry_outcomes:
            identity = (outcome.task.method, outcome.task.dataset)
            if outcome.shard is None:
                if terminal_failure is None:
                    terminal_failure = outcome
                _progress_antigravity(
                    progress,
                    outcome=outcome,
                    wave="retry",
                    attempt=2,
                    status="terminal_failed",
                    durable_completed=len(shards),
                )
                continue
            publish_score_shard_antigravity(root, outcome.shard)
            reopened = read_score_shard_antigravity(root, *identity)
            if reopened != outcome.shard:
                raise ValueError("Exp471 durable retry shard differs after publication")
            shards[identity] = reopened
            _progress_antigravity(
                progress,
                outcome=outcome,
                wave="retry",
                attempt=2,
                status="durable",
                durable_completed=len(shards),
            )
    if terminal_failure is not None:
        raise RuntimeError(
            terminal_failure.error_message
            or terminal_failure.error_type
            or "Exp471 score task failed twice"
        )

    decision = _validated_benchmark_decision_antigravity(
        decision, expected_context_sha256=metadata.cache_context_sha256
    )
    _publish_benchmark_decision_antigravity(root, decision)
    reopened_decision = _read_benchmark_decision_antigravity(
        root, expected_context_sha256=metadata.cache_context_sha256
    )
    if reopened_decision != decision:
        raise ValueError("Exp471 benchmark decision differs after publication")
    manifest = _manifest_from_shards_antigravity(metadata, shards)
    publish_cache_manifest_antigravity(root, manifest)
    reopened_manifest = read_cache_manifest_antigravity(root)
    if reopened_manifest != manifest:
        raise ValueError("Exp471 terminal cache manifest differs after publication")
    return reopened_manifest


def _scan_shards_antigravity(root: Path) -> tuple[Exp471ScoreShardAntigravity, ...]:
    if not os.path.lexists(root):
        return ()
    anchor = _artifacts_antigravity._open_absolute_directory_antigravity(
        Path(root), create=False
    )
    directory: int | None = None
    try:
        try:
            directory, expected = _artifacts_antigravity._open_child_directory_antigravity(
                anchor.descriptor, "score_shards_antigravity", create=False
            )
        except ValueError:
            if not os.path.lexists(Path(root) / "score_shards_antigravity"):
                return ()
            raise
        shards: dict[tuple[str, str], Exp471ScoreShardAntigravity] = {}
        for name in sorted(os.listdir(directory)):
            raw = _artifacts_antigravity._read_leaf_at_antigravity(directory, name)
            if raw is None:
                raise ValueError("Exp471 status score shard disappeared")
            shard = _artifacts_antigravity._score_shard_from_bytes_antigravity(raw)
            expected_name = score_shard_relative_path_antigravity(
                shard.method, shard.dataset
            ).name
            if name != expected_name:
                raise ValueError("Exp471 status score shard identity differs from path")
            identity = (shard.method, shard.dataset)
            if identity in shards:
                raise ValueError("Exp471 status contains duplicate score shard identity")
            shards[identity] = shard
        _artifacts_antigravity._validate_child_directory_antigravity(
            anchor.descriptor, "score_shards_antigravity", expected
        )
        anchor.validate()
        return tuple(
            sorted(
                shards.values(),
                key=lambda shard: (
                    METHOD_ORDER_ANTIGRAVITY.index(shard.method), shard.dataset
                ),
            )
        )
    finally:
        if directory is not None:
            os.close(directory)
        anchor.close()


def status_score_cache_antigravity(root: Path) -> Mapping[str, object]:
    """Return verified durable coverage and the actual cache worker decision."""

    root = Path(root)
    manifest_path = root / "score_cache_manifest_antigravity.json"
    if os.path.lexists(manifest_path):
        manifest = read_cache_manifest_antigravity(root)
        complete = True
        durable_completed = manifest.total_shards
        total_shards: int | None = manifest.total_shards
        method_order: tuple[str, ...] = manifest.method_order
        dataset_order: tuple[str, ...] = manifest.dataset_order
        manifest_sha256: str | None = manifest.manifest_sha256
        expected_benchmark_context: str | None = manifest.cache_context_sha256
    else:
        shards = _scan_shards_antigravity(root)
        complete = False
        durable_completed = len(shards)
        total_shards = None
        method_order = METHOD_ORDER_ANTIGRAVITY
        dataset_order = tuple(dict.fromkeys(shard.dataset for shard in shards))
        manifest_sha256 = None
        shard_contexts = {shard.cache_context_sha256 for shard in shards}
        if len(shard_contexts) > 1:
            raise ValueError("Exp471 incomplete shard cache contexts differ")
        expected_benchmark_context = (
            next(iter(shard_contexts)) if shard_contexts else None
        )
    benchmark_path = root / _BENCHMARK_NAME_ANTIGRAVITY
    if complete and not os.path.lexists(benchmark_path):
        raise ValueError("Exp471 benchmark decision is missing")
    benchmark_decision = None
    if os.path.lexists(benchmark_path):
        benchmark_decision = _read_benchmark_decision_antigravity(
            root, expected_context_sha256=expected_benchmark_context
        )
    return {
        "schema": _STATUS_SCHEMA_ANTIGRAVITY,
        "root": str(root),
        "complete": complete,
        "durable_completed": durable_completed,
        "total_shards": total_shards,
        "method_order": method_order,
        "dataset_order": dataset_order,
        "manifest_sha256": manifest_sha256,
        "benchmark_decision": benchmark_decision,
        "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
        "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
        "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
    }


def open_complete_score_cache_antigravity(
    session: Exp471SourceSessionAntigravity,
) -> Exp471ScoreCacheManifestAntigravity:
    """Open a complete cache only for the issued source authority that built it."""

    session = _require_issued_session_antigravity(session)
    if not callable(session.score_reader_antigravity):
        raise ValueError("Exp471 issued session has no score-reader capability")
    metadata = _validate_verified_input_metadata_antigravity(
        session.authority, session.score_reader_antigravity()
    )
    root = Path(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
    manifest = read_cache_manifest_antigravity(root)
    _validate_manifest_metadata_antigravity(manifest, metadata)
    _read_benchmark_decision_antigravity(
        root, expected_context_sha256=metadata.cache_context_sha256
    )
    return manifest


__all__ = [
    "benchmark_score_preparation_antigravity",
    "open_complete_score_cache_antigravity",
    "prepare_score_cache_antigravity",
    "status_score_cache_antigravity",
]
