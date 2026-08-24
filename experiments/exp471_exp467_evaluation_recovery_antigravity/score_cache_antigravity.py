"""Label-free, resumable preparation of deterministic Exp471 score shards."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
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


_INPUT_SCHEMA_ANTIGRAVITY = "exp471-verified-score-input-antigravity.v1"
_STATUS_SCHEMA_ANTIGRAVITY = "exp471-score-cache-status-antigravity.v1"
_BENCHMARK_SCHEMA_ANTIGRAVITY = "exp471-score-preparation-benchmark-antigravity.v1"
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
    source_digests: Mapping[str, str],
) -> str:
    record = {
        "schema": "exp471-score-cache-context-antigravity.v1",
        "source_authority_sha256": authority.authority_sha256,
        "method_order": list(METHOD_ORDER_ANTIGRAVITY),
        "dataset_order": list(datasets),
        "source_manifest_sha256": dict(source_digests),
        "calibration": {
            "function": "upper_tail_pvalues_numba_antigravity",
            "dtype": "float64",
            "fastmath": False,
            "parallel": False,
            "nogil": True,
        },
    }
    return sha256(canonical_json_bytes_exp471_antigravity(record)).hexdigest()


def _build_tasks_from_verified_input_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    value: object,
) -> tuple[_ScoreTaskAntigravity, ...]:
    """Validate all inputs in the parent and return capability-free tasks."""

    if not isinstance(authority, Exp471SourceAuthorityAntigravity):
        raise ValueError("Exp471 score input authority has the wrong type")
    bundle = _mapping_antigravity(value, "Exp471 verified score input")
    if set(bundle) != {"schema", "dataset_order", "conditions", "baselines"}:
        raise ValueError("Exp471 verified score input schema differs")
    if bundle["schema"] != _INPUT_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp471 verified score input schema differs")
    dataset_value = bundle["dataset_order"]
    if not isinstance(dataset_value, (tuple, list)) or not dataset_value:
        raise ValueError("Exp471 dataset order must be a non-empty sequence")
    datasets = tuple(dataset_value)
    if (
        any(type(dataset) is not str or not dataset for dataset in datasets)
        or len(datasets) != len(set(datasets))
    ):
        raise ValueError("Exp471 dataset order is invalid")

    conditions = _mapping_antigravity(bundle["conditions"], "Exp471 conditions")
    baselines = _mapping_antigravity(bundle["baselines"], "Exp471 baselines")
    if tuple(conditions) != METHOD_ORDER_ANTIGRAVITY[:4]:
        raise ValueError("Exp471 condition method order differs")
    if tuple(baselines) != METHOD_ORDER_ANTIGRAVITY[4:]:
        raise ValueError("Exp471 baseline method order differs")

    source_digests: dict[str, str] = {}
    checked: dict[str, tuple[str, Mapping[str, object]]] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        row = _mapping_antigravity(
            conditions[method] if method in conditions else baselines[method],
            f"Exp471 {method} source",
        )
        if set(row) != {"source_manifest_sha256", "scores"}:
            raise ValueError(f"Exp471 {method} source schema differs")
        source_digest = _digest_antigravity(
            row["source_manifest_sha256"],
            f"Exp471 {method} source_manifest_sha256",
        )
        scores = _mapping_antigravity(row["scores"], f"Exp471 {method} scores")
        if tuple(scores) != datasets:
            raise ValueError(f"Exp471 {method} dataset order differs")
        source_digests[method] = source_digest
        checked[method] = (source_digest, scores)

    cache_context = _cache_context_antigravity(
        authority, datasets, source_digests
    )
    tasks: list[_ScoreTaskAntigravity] = []
    for method in METHOD_ORDER_ANTIGRAVITY:
        source_digest, scores = checked[method]
        for dataset in datasets:
            if method in METHOD_ORDER_ANTIGRAVITY[:4]:
                raw_scores = _float64_vector_antigravity(
                    scores[dataset], f"Exp471 {method}/{dataset} TEST scores"
                )
                micro_scores = raw_scores
            else:
                vectors = _mapping_antigravity(
                    scores[dataset], f"Exp471 {method}/{dataset} score vectors"
                )
                if set(vectors) != {"train_oof_scores", "test_scores"}:
                    raise ValueError(
                        f"Exp471 {method}/{dataset} score-vector schema differs"
                    )
                train_scores = _float64_vector_antigravity(
                    vectors["train_oof_scores"],
                    f"Exp471 {method}/{dataset} TRAIN OOF scores",
                )
                raw_scores = _float64_vector_antigravity(
                    vectors["test_scores"],
                    f"Exp471 {method}/{dataset} TEST scores",
                )
                train = np.ascontiguousarray(train_scores, dtype=np.float64)
                test = np.ascontiguousarray(raw_scores, dtype=np.float64)
                calibrated = 1.0 - upper_tail_pvalues_numba_antigravity(
                    train, test
                )
                if calibrated.dtype != np.dtype(np.float64) or not np.isfinite(
                    calibrated
                ).all():
                    raise ValueError(
                        f"Exp471 {method}/{dataset} micro scores must be finite float64"
                    )
                micro_scores = tuple(float(item) for item in calibrated)
            tasks.append(
                _ScoreTaskAntigravity(
                    source_authority_sha256=authority.authority_sha256,
                    cache_context_sha256=cache_context,
                    method=method,
                    dataset=dataset,
                    test_count=len(raw_scores),
                    raw_scores=raw_scores,
                    micro_scores=micro_scores,
                    source_manifest_sha256=source_digest,
                )
            )
    return tuple(tasks)


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
        parallel_workers > 1 and byte_parity and speedup >= 1.1
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
    elif speedup < 1.1:
        fallback_reason = "parallel speedup is below the required 1.1x gate"
    else:
        fallback_reason = ""
    return {
        "schema": _BENCHMARK_SCHEMA_ANTIGRAVITY,
        "detected_logical_cpus": LOGICAL_CPU_COUNT_ANTIGRAVITY,
        "default_cpu_workers": CPU_WORKERS_ANTIGRAVITY,
        "requested_parallel_workers": parallel_workers,
        "selected_workers": parallel_workers if parallel_accepted else 1,
        "inner_numerical_threads": INNER_NUMERICAL_THREADS_ANTIGRAVITY,
        "single_worker_seconds": single_seconds,
        "parallel_seconds": parallel_seconds,
        "speedup": speedup,
        "byte_parity": byte_parity,
        "parallel_accepted": parallel_accepted,
        "parallel_error_type": parallel_error_type,
        "parallel_error_message": parallel_error_message,
        "fallback_reason": fallback_reason,
        "acceleration": dict(ACCELERATION_ASSESSMENT_ANTIGRAVITY),
    }


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


def _task_matches_shard_antigravity(
    task: _ScoreTaskAntigravity, shard: Exp471ScoreShardAntigravity
) -> bool:
    return (
        shard.source_authority_sha256 == task.source_authority_sha256
        and shard.cache_context_sha256 == task.cache_context_sha256
        and shard.method == task.method
        and shard.dataset == task.dataset
        and shard.test_count == task.test_count
        and shard.raw_scores == task.raw_scores
        and shard.micro_scores == task.micro_scores
        and shard.source_manifest_sha256 == task.source_manifest_sha256
    )


def _existing_shard_antigravity(
    root: Path, task: _ScoreTaskAntigravity
) -> Exp471ScoreShardAntigravity | None:
    path = root / score_shard_relative_path_antigravity(task.method, task.dataset)
    if not os.path.lexists(path):
        return None
    shard = read_score_shard_antigravity(root, task.method, task.dataset)
    if not _task_matches_shard_antigravity(task, shard):
        raise ValueError("Exp471 existing score shard differs from verified input")
    return shard


def _manifest_from_shards_antigravity(
    tasks: Sequence[_ScoreTaskAntigravity],
    shards: Mapping[tuple[str, str], Exp471ScoreShardAntigravity],
) -> Exp471ScoreCacheManifestAntigravity:
    datasets = tuple(
        task.dataset
        for task in tasks
        if task.method == METHOD_ORDER_ANTIGRAVITY[0]
    )
    required = {
        (method, dataset)
        for method in METHOD_ORDER_ANTIGRAVITY
        for dataset in datasets
    }
    if set(shards) != required:
        raise ValueError("Exp471 cannot publish an incomplete score-cache manifest")
    first = tasks[0]
    record: dict[str, object] = {
        "source_authority_sha256": first.source_authority_sha256,
        "cache_context_sha256": first.cache_context_sha256,
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
    tasks = _build_tasks_from_verified_input_antigravity(
        session.authority, verified_input
    )
    root = Path(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
    shards: dict[tuple[str, str], Exp471ScoreShardAntigravity] = {}
    pending: list[_ScoreTaskAntigravity] = []
    for task in tasks:
        existing = _existing_shard_antigravity(root, task)
        if existing is None:
            pending.append(task)
        else:
            shards[(task.method, task.dataset)] = existing

    manifest_path = root / "score_cache_manifest_antigravity.json"
    if os.path.lexists(manifest_path):
        manifest = read_cache_manifest_antigravity(root)
        expected = _manifest_from_shards_antigravity(tasks, shards)
        if manifest != expected or pending:
            raise ValueError("Exp471 existing terminal cache manifest differs")
        return manifest

    actual_workers = 1
    if selected_request > 1 and len(pending) > 1:
        benchmark = benchmark_score_preparation_antigravity(
            pending, parallel_workers=selected_request
        )
        actual_workers = int(benchmark["selected_workers"])

    failures: list[_TaskOutcomeAntigravity] = []
    for outcome in _run_tasks_antigravity(pending, actual_workers):
        identity = (outcome.task.method, outcome.task.dataset)
        if outcome.shard is None:
            failures.append(outcome)
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
    if failures:
        retry_tasks = tuple(outcome.task for outcome in failures)
        for outcome in _run_tasks_antigravity(retry_tasks, actual_workers):
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

    manifest = _manifest_from_shards_antigravity(tasks, shards)
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
    """Return verified durable coverage and frozen CPU fallback provenance."""

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
    else:
        shards = _scan_shards_antigravity(root)
        complete = False
        durable_completed = len(shards)
        total_shards = None
        method_order = METHOD_ORDER_ANTIGRAVITY
        dataset_order = tuple(dict.fromkeys(shard.dataset for shard in shards))
        manifest_sha256 = None
    return {
        "schema": _STATUS_SCHEMA_ANTIGRAVITY,
        "root": str(root),
        "complete": complete,
        "durable_completed": durable_completed,
        "total_shards": total_shards,
        "method_order": method_order,
        "dataset_order": dataset_order,
        "manifest_sha256": manifest_sha256,
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
    manifest = read_cache_manifest_antigravity(
        Path(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
    )
    if manifest.source_authority_sha256 != session.authority.authority_sha256:
        raise ValueError("Exp471 complete score cache authority differs")
    return manifest


__all__ = [
    "benchmark_score_preparation_antigravity",
    "open_complete_score_cache_antigravity",
    "prepare_score_cache_antigravity",
    "status_score_cache_antigravity",
]
