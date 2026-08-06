"""Ordered, parent-owned Exp436 runtime with no label or DB access in workers."""
from __future__ import annotations

import os

# This is deliberately the first non-stdlib action.  Imports below reach NumPy
# through contracts/scoring, so every worker and the parent have one numerical
# library thread before that transitively happens.
_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
)
for _thread_name in _THREAD_ENV:
    os.environ[_thread_name] = "1"

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import platform
import sys
from typing import Callable, Mapping

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from .backend_antigravity import score_with_backend, validate_backend_engine
from .checkpoint_antigravity import Exp436CheckpointStore, Exp436RunContext, context_payload
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendDecision, Exp436DatasetResult, Exp436RunSummary
from .progress_antigravity import ProgressWriter, inspect_progress


_READINESS_KEYS = frozenset({
    "schema", "run_identity", "reviewed_commit", "database_path", "database_sha256",
    "exp291_prediction_path", "exp291_prediction_sha256", "exp291_manifest_path",
    "exp291_manifest_sha256", "exp292_prediction_path", "exp292_prediction_sha256",
    "exp292_manifest_path", "exp292_manifest_sha256", "datasets", "dataset_digest",
    "frozen_config", "frozen_config_sha256", "backend_decision",
    "backend_decision_sha256", "run_root", "checkpoint_root",
})


def _frozen_config_payload() -> dict[str, object]:
    return json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))


def _digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return str(Path(value).resolve())


def _canonical_readiness(readiness_path: Path, reviewed_commit: str, run_root: Path, checkpoint_root: Path) -> dict[str, object]:
    """Validate every public external binding before creating an input session."""

    try:
        raw = Path(readiness_path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("readiness artifact is invalid") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw or set(payload) != _READINESS_KEYS:
        raise ValueError("readiness artifact schema is invalid")
    if payload["schema"] != "exp436.readiness-binding.v1" or payload["run_identity"] != FROZEN_CONFIG.run_identity:
        raise ValueError("readiness artifact identity is invalid")
    if type(reviewed_commit) is not str or len(reviewed_commit) != 40 or payload["reviewed_commit"] != reviewed_commit:
        raise ValueError("readiness reviewed commit is invalid")
    if _absolute(payload["run_root"], "readiness run_root") != str(Path(run_root).resolve()):
        raise ValueError("readiness run root is invalid")
    if _absolute(payload["checkpoint_root"], "readiness checkpoint_root") != str(Path(checkpoint_root).resolve()):
        raise ValueError("readiness checkpoint root is invalid")
    paths = ("database", "exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest")
    for name in paths:
        _absolute(payload[f"{name}_path"], f"readiness {name}")
        value = payload[f"{name}_sha256"]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("readiness digest is invalid")
    datasets = payload["datasets"]
    if not isinstance(datasets, list) or len(datasets) != FROZEN_CONFIG.expected_datasets or datasets != sorted(datasets) or len(datasets) != len(set(datasets)) or any(type(item) is not str or not item for item in datasets):
        raise ValueError("readiness population is invalid")
    if payload["dataset_digest"] != _digest(datasets):
        raise ValueError("readiness dataset digest is invalid")
    config = _frozen_config_payload()
    if payload["frozen_config"] != config or payload["frozen_config_sha256"] != _digest(config):
        raise ValueError("readiness frozen config is invalid")
    try:
        decision = BackendDecision.from_payload(payload["backend_decision"])
    except (TypeError, ValueError) as error:
        raise ValueError("readiness backend decision is invalid") from error
    if decision.to_payload() != payload["backend_decision"] or payload["backend_decision_sha256"] != _digest(payload["backend_decision"]):
        raise ValueError("readiness backend decision is invalid")
    return payload


@contextmanager
def _session_from_factory(factory: Callable[[], object]):
    if not callable(factory):
        raise ValueError("input_session_factory is required")
    owned = factory()
    if hasattr(owned, "__enter__") and hasattr(owned, "__exit__"):
        with owned as session:
            yield session
        return
    try:
        yield owned
    finally:
        close = getattr(owned, "_close_antigravity", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class LaneMembershipSnapshot:
    """Picklable copy of all frozen lanes; workers receive no live session."""

    hard: frozenset[tuple[str, int]]
    standard: frozenset[tuple[str, int]]
    priority: frozenset[tuple[str, int]]
    automatic_normal: frozenset[tuple[str, int]]

    def contains(self, dataset: str, instance_index: int) -> bool:
        coordinate = (dataset, instance_index)
        return any(coordinate in lane for lane in (self.hard, self.standard, self.priority, self.automatic_normal))


def _lane_mapping(lanes: object, name: str) -> Mapping[str, object]:
    value = getattr(lanes, name, None)
    if value is None and isinstance(lanes, Mapping):
        value = lanes.get(name)
    if not isinstance(value, Mapping):
        raise ValueError("frozen lane mapping is invalid")
    return value


def _lanes_snapshot(lanes: object, datasets: tuple[str, ...]) -> LaneMembershipSnapshot:
    copied: dict[str, frozenset[tuple[str, int]]] = {}
    seen: set[tuple[str, int]] = set()
    for name in ("hard", "standard", "priority", "automatic_normal"):
        items: set[tuple[str, int]] = set()
        for dataset, indices in _lane_mapping(lanes, name).items():
            if dataset not in datasets or not isinstance(indices, (tuple, list, frozenset)):
                raise ValueError("frozen lane mapping is invalid")
            for index in indices:
                if type(index) is not int or index < 0 or (dataset, index) in seen:
                    raise ValueError("frozen lane mappings overlap or are invalid")
                items.add((dataset, index)); seen.add((dataset, index))
        copied[name] = frozenset(items)
    return LaneMembershipSnapshot(**copied)


def _context_lanes_payload(lanes: object) -> dict[str, dict[str, list[int]]]:
    return {
        name: {dataset: list(indices) for dataset, indices in sorted(_lane_mapping(lanes, name).items())}
        for name in ("hard", "standard", "priority", "automatic_normal")
    }


def _build_run_context(*, session: object, readiness: Mapping[str, object], readiness_path: Path | str, run_root: Path | str, checkpoint_root: Path | str, decision: BackendDecision) -> Exp436RunContext:
    """Make the exact v1 checkpoint context from already-validated authorities."""

    source = getattr(session, "context", None)
    datasets = tuple(getattr(source, "datasets", ()))
    lanes = getattr(source, "lanes", None)
    manifest = getattr(source, "test_identity_manifest", None)
    if not datasets or datasets != tuple(readiness["datasets"]) or lanes is None or not isinstance(manifest, Mapping):
        raise ValueError("input session context does not match readiness")
    if getattr(source, "code_commit", None) != readiness["reviewed_commit"] or getattr(source, "database_sha256", None) != readiness["database_sha256"]:
        raise ValueError("input session identity does not match readiness")
    source_paths = getattr(session, "source_paths", None)
    source_payload = getattr(source_paths, "to_payload", None)
    if not callable(source_payload):
        raise ValueError("input session source paths are unavailable")
    paths = source_payload()
    if not isinstance(paths, Mapping):
        raise ValueError("input session source paths are invalid")
    lane_artifacts: dict[str, dict[str, str]] = {}
    for name in ("exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest"):
        if paths.get(name) != readiness[f"{name}_path"] or getattr(lanes, f"{name}_sha256", None) != readiness[f"{name}_sha256"]:
            raise ValueError("input session lane authority does not match readiness")
        lane_artifacts[name] = {"path": _absolute(paths[name], name), "sha256": str(readiness[f"{name}_sha256"])}
    if paths.get("database") != readiness["database_path"]:
        raise ValueError("input session database path does not match readiness")
    test_manifest = {dataset: [[index, length] for index, length in manifest[dataset]] for dataset in datasets}
    lane_mappings = _context_lanes_payload(lanes)
    config = _frozen_config_payload()
    payload: dict[str, object] = {
        "schema": "exp436.run-context.v1", "protocol_versions": {"context": "v1", "checkpoint": "v1", "readiness": "v1"},
        "experiment_id": FROZEN_CONFIG.experiment_id, "run_identity": FROZEN_CONFIG.run_identity,
        "reviewed_commit": readiness["reviewed_commit"], "readiness_path": _absolute(str(readiness_path), "readiness_path"),
        "readiness_artifact_sha256": _digest(dict(readiness)), "backend_decision": decision.to_payload(),
        "backend_decision_sha256": _digest(decision.to_payload()), "database_path": _absolute(paths["database"], "database"),
        "database_sha256": readiness["database_sha256"], "lane_artifacts": lane_artifacts, "lane_mappings": lane_mappings,
        "lane_snapshot_sha256": _digest({"artifacts": lane_artifacts, "mappings": lane_mappings, "datasets": list(datasets)}),
        "datasets": list(datasets), "dataset_order_sha256": _digest(list(datasets)), "test_identity_manifest": test_manifest,
        "test_manifest_sha256": _digest(test_manifest), "frozen_config": config, "frozen_config_sha256": _digest(config),
        "python_implementation": platform.python_implementation(), "python_version": platform.python_version(),
        "python_executable": _absolute(sys.executable, "python executable"), "numpy_version": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(), "multiprocessing_start_method": multiprocessing.get_start_method(),
        "logical_cpus": os.cpu_count() or 1, "cpu_workers": FROZEN_CONFIG.cpu_workers,
        "selected_framework_provenance": {"framework": decision.backend, "adopted": decision.adopted},
        "backend_protocol": FROZEN_CONFIG.backend_protocol, "selected_batch_size": decision.selected_batch_size,
        "output_path": _absolute(str(run_root), "run_root"), "checkpoint_path": _absolute(str(checkpoint_root), "checkpoint_root"),
    }
    payload["context_sha256"] = _digest(payload)
    return Exp436RunContext.from_payload(payload)


def _worker_initializer() -> None:
    for name in _THREAD_ENV:
        os.environ[name] = "1"


@dataclass(frozen=True)
class CpuDatasetTask:
    """Only immutable raw values, lanes, and the frozen CPU decision cross IPC."""

    dataset: str
    train: object
    test: object
    lanes: LaneMembershipSnapshot
    decision: BackendDecision


def _score_cpu_task(task: CpuDatasetTask) -> Exp436DatasetResult:
    from .contracts_antigravity import MissingSafeRawDataset
    from .features_antigravity import fit_calibrated_group, prepare_train_groups
    from .scoring_antigravity import score_dataset_cpu

    if (not isinstance(task, CpuDatasetTask) or not isinstance(task.train, MissingSafeRawDataset)
            or not isinstance(task.test, MissingSafeRawDataset) or task.train.split != "TRAIN"
            or task.test.split != "TEST" or task.train.dataset != task.dataset or task.test.dataset != task.dataset
            or not isinstance(task.lanes, LaneMembershipSnapshot) or not isinstance(task.decision, BackendDecision)
            or task.decision.backend != "cpu" or task.decision.adopted):
        raise ValueError("CPU task is invalid")
    groups = tuple(fit_calibrated_group(group) for group in prepare_train_groups(task.train).groups)
    result = score_dataset_cpu(groups, task.test, task.lanes)
    if result.dataset != task.dataset:
        raise ValueError("CPU task returned the wrong dataset")
    return result


def _score_accelerated(train: object, test: object, lanes: LaneMembershipSnapshot, decision: BackendDecision, engine: object) -> Exp436DatasetResult:
    from .features_antigravity import fit_calibrated_group, prepare_train_groups
    from .scoring_antigravity import score_dataset_backend

    groups = tuple(fit_calibrated_group(group) for group in prepare_train_groups(train).groups)
    return score_dataset_backend(groups, test, lanes, lambda query, references, k: score_with_backend(decision, query, references, k, engine=engine))


def _load(session: object, dataset: str) -> tuple[object, object, dict[str, object]]:
    load_train, load_test = getattr(session, "load_train", None), getattr(session, "load_test_values", None)
    if not callable(load_train) or not callable(load_test):
        raise ValueError("input session lacks label-free loaders")
    train = load_train(dataset); test = load_test(dataset, train)
    def histogram(values: object) -> dict[str, int]:
        lengths = sorted(len(row.values) for row in getattr(values, "series", ()))
        return {str(length): lengths.count(length) for length in sorted(set(lengths))}
    metadata = {"train_count": len(getattr(train, "series", ())), "test_count": len(getattr(test, "series", ())),
                "train_length_histogram": histogram(train), "test_length_histogram": histogram(test)}
    return train, test, metadata


def _counts(store: Exp436CheckpointStore) -> dict[str, int]:
    values = tuple(record for _, _, record, _ in store.completed_records())
    return {"completed": len(values), "success": sum(item.status == "scored" for item in values),
            "abstention": sum(item.status == "abstained" for item in values), "error": 0,
            "candidates": sum(item.candidate_count for item in values),
            "excluded": sum(len(item.train_exclusion_audits) for item in values),
            "nonfinite": sum(sum(row.exclusion_reason == "nonfinite_test" for row in item.evidence) for item in values)}


def _summary(store: Exp436CheckpointStore, backend: BackendDecision, status: str) -> Exp436RunSummary:
    counters = _counts(store)
    return Exp436RunSummary(store.run_identity, status, len(store.datasets), counters["completed"], counters["success"], counters["abstention"], 0, counters["candidates"], backend, counters["excluded"], counters["nonfinite"])


def _writer_update(writer: ProgressWriter, counters: Mapping[str, int], stage: str, dataset: str | None, metadata: Mapping[str, object] | None, inflight: tuple[str, ...] = ()) -> None:
    writer.update(stage=stage, current_dataset=dataset, dataset_metadata=metadata, in_flight_datasets=tuple(sorted(inflight)),
                  completed=counters["completed"], success=counters["success"], abstention=counters["abstention"], error=0,
                  candidates=counters["candidates"], excluded_train_rows=counters["excluded"], nonfinite_test_rows=counters["nonfinite"])


def _commit(store: Exp436CheckpointStore, writer: ProgressWriter, result: Exp436DatasetResult, metadata: Mapping[str, object], counters: dict[str, int]) -> None:
    if counters["candidates"] + result.candidate_count > FROZEN_CONFIG.global_candidate_budget:
        raise ValueError("Exp436 candidate budget exceeded before checkpoint intent")
    _writer_update(writer, counters, "committing", result.dataset, metadata)
    digest = store.commit(result, dataset_metadata=metadata)  # checkpoint record is authoritative before event/progress.
    counters["completed"] += 1; counters["success"] += result.status == "scored"; counters["abstention"] += result.status == "abstained"
    counters["candidates"] += result.candidate_count; counters["excluded"] += len(result.train_exclusion_audits)
    counters["nonfinite"] += sum(row.exclusion_reason == "nonfinite_test" for row in result.evidence)
    writer.record_dataset_complete(sequence_index=counters["completed"] - 1, dataset=result.dataset, record_sha256=digest,
                                  status=result.status, dataset_metadata=metadata, completed_datasets=counters["completed"],
                                  success=counters["success"], abstention=counters["abstention"], error=0,
                                  excluded_train_rows=counters["excluded"], nonfinite_test_rows=counters["nonfinite"],
                                  candidate_count=counters["candidates"])


def _engine(decision: BackendDecision) -> object:
    if decision.backend == "mlx":
        from .mlx_antigravity import MlxEngine
        return validate_backend_engine(decision, MlxEngine())
    if decision.backend == "mps":
        from .mps_antigravity import MpsEngine
        return validate_backend_engine(decision, MpsEngine())
    raise ValueError("CPU has no accelerator engine")


def _terminate(executor: ProcessPoolExecutor) -> None:
    for process in tuple(getattr(executor, "_processes", {}).values()):
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def _mark_checkpoint_complete(store: Exp436CheckpointStore, writer: ProgressWriter) -> None:
    """Terminal completion is legal only when the checkpoint authority is full."""
    if len(store.completed_digests()) != len(store.datasets):
        raise ValueError("checkpoint completion requires a complete checkpoint")
    writer.mark_checkpoint_complete()


def _run_cpu_schedule(*, executor: ProcessPoolExecutor, missing: tuple[str, ...], session: object,
                      lanes: LaneMembershipSnapshot, decision: BackendDecision, store: Exp436CheckpointStore,
                      writer: ProgressWriter, counters: dict[str, int], _observer: Callable[[int, int], None] | None = None) -> None:
    """Bounded ordered scheduler; injected only privately for deterministic tests."""

    pending: dict[Future[Exp436DatasetResult], tuple[str, Mapping[str, object]]] = {}
    buffered: dict[str, tuple[Exp436DatasetResult, Mapping[str, object]]] = {}
    submit_index = commit_index = 0
    try:
        while submit_index < len(missing) or pending:
            while submit_index < len(missing) and len(pending) + len(buffered) < FROZEN_CONFIG.cpu_workers:
                dataset = missing[submit_index]
                train, test, metadata = _load(session, dataset)
                _writer_update(writer, counters, "loading", dataset, metadata, tuple(name for name, _ in pending.values()))
                _writer_update(writer, counters, "dispatching", dataset, metadata, tuple(name for name, _ in pending.values()))
                task = CpuDatasetTask(dataset, train, test, lanes, decision)
                pending[executor.submit(_score_cpu_task, task)] = (dataset, metadata)
                submit_index += 1
            if _observer is not None: _observer(len(pending), len(buffered))
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                dataset, metadata = pending.pop(future)
                # Bind the exact future before retrieving its result so a
                # worker exception cannot be attributed to a later submission.
                _writer_update(writer, counters, "awaiting_worker_result", dataset, metadata, tuple(name for name, _ in pending.values()))
                buffered[dataset] = (future.result(), metadata)
            if _observer is not None: _observer(len(pending), len(buffered))
            while commit_index < len(missing) and missing[commit_index] in buffered:
                dataset = missing[commit_index]
                result, metadata = buffered.pop(dataset)
                _commit(store, writer, result, metadata, counters)
                commit_index += 1
            if _observer is not None: _observer(len(pending), len(buffered))
            if pending:
                dataset, metadata = next(iter(pending.values()))
                _writer_update(writer, counters, "awaiting_worker_result", dataset, metadata, tuple(name for name, _ in pending.values()))
    except KeyboardInterrupt:
        for future in pending: future.cancel()
        _terminate(executor)
        raise
    except BaseException:
        for future in pending: future.cancel()
        _terminate(executor)
        raise


def _reconcile_progress(store: Exp436CheckpointStore, writer: ProgressWriter) -> None:
    inspection = store.inspect()
    records = store.completed_records()
    completed_digests = {dataset: digest for dataset, digest, _, _ in records}
    checkpoint_inspection = {"completed": inspection.completed, "datasets": tuple(completed_digests), "record_digests": completed_digests}
    state = inspect_progress(writer.output, run_identity=store.run_identity, checkpoint=store.root,
                             context_fingerprint=store.context_fingerprint, checkpoint_inspection=checkpoint_inspection)
    cumulative = {"success": 0, "abstention": 0, "error": 0, "excluded_train_rows": 0, "nonfinite_test_rows": 0, "candidate_count": 0}
    completion_payloads = tuple(state["completion_payloads"])
    for index, (dataset, digest, result, metadata) in enumerate(records):
        cumulative["success"] += result.status == "scored"; cumulative["abstention"] += result.status == "abstained"
        cumulative["excluded_train_rows"] += len(result.train_exclusion_audits)
        cumulative["nonfinite_test_rows"] += sum(row.exclusion_reason == "nonfinite_test" for row in result.evidence)
        cumulative["candidate_count"] += result.candidate_count
        if index < len(completion_payloads):
            payload = completion_payloads[index]
            expected = {"sequence_index": index, "dataset": dataset, "record_sha256": digest, "status": result.status, "dataset_metadata": metadata, "completed_datasets": index + 1, "recomputed_datasets": index + 1, **cumulative}
            if dict(payload) != expected:
                raise ValueError("runtime event does not match cumulative checkpoint contents")
    writer.synchronize_event_prefix(checkpoint_inspection=checkpoint_inspection)
    missing = tuple(state["reconciliation_required"])
    if not missing:
        return
    bindings = []
    cumulative = {"success": 0, "abstention": 0, "error": 0, "excluded_train_rows": 0, "nonfinite_test_rows": 0, "candidate_count": 0}
    for index, (dataset, digest, result, metadata) in enumerate(records):
        cumulative["success"] += result.status == "scored"; cumulative["abstention"] += result.status == "abstained"
        cumulative["excluded_train_rows"] += len(result.train_exclusion_audits)
        cumulative["nonfinite_test_rows"] += sum(row.exclusion_reason == "nonfinite_test" for row in result.evidence)
        cumulative["candidate_count"] += result.candidate_count
        if dataset not in missing:
            continue
        bindings.append((index, dataset, completed_digests[dataset], result.status, metadata,
                         dict(cumulative)))
    writer.reconcile_checkpoint_tail(tuple(bindings), checkpoint_inspection=checkpoint_inspection)


def _run(*, run_root: Path, checkpoint_root: Path, readiness_path: Path, readiness: Mapping[str, object], input_session: object, decision: BackendDecision, resume: bool) -> Exp436RunSummary:
    context = _build_run_context(session=input_session, readiness=readiness, readiness_path=readiness_path, run_root=run_root, checkpoint_root=checkpoint_root, decision=decision)
    lanes = _lanes_snapshot(getattr(getattr(input_session, "context", None), "lanes", None), tuple(context.to_payload()["datasets"]))
    if resume:
        # Read-only classification accepts only recognized crash leftovers;
        # output ownership is claimed before the mutable checkpoint repair.
        readonly = Exp436CheckpointStore.open_classified_read_only(checkpoint_root, expected_context=context)
        inspection = readonly.inspect()
        writer = ProgressWriter.open_existing(run_root, run_identity=readonly.run_identity, checkpoint=checkpoint_root, backend=decision.backend, total_datasets=len(readonly.datasets), context_fingerprint=readonly.context_fingerprint, checkpoint_inspection={"completed": inspection.completed, "datasets": tuple(readonly.completed_digests()), "record_digests": readonly.completed_digests()})
        # The output writer now owns its exclusive lock before mutable checkpoint repair.
        store = Exp436CheckpointStore.open_existing(checkpoint_root, context)
    else:
        if run_root.exists() or checkpoint_root.exists():
            raise ValueError("new runtime roots must be absent")
        store = Exp436CheckpointStore.create(checkpoint_root, context)
        writer = ProgressWriter.create(run_root, run_identity=store.run_identity, checkpoint=checkpoint_root, total_datasets=len(store.datasets), backend=decision.backend, context_fingerprint=store.context_fingerprint)
    try:
        _reconcile_progress(store, writer)
        completed = set(store.completed_digests()); missing = tuple(dataset for dataset in store.datasets if dataset not in completed)
        if not missing:
            _mark_checkpoint_complete(store, writer)
            return _summary(store, decision, "checkpoint_complete")
        counters = _counts(store); writer.start_heartbeat()
        if decision.adopted:
            engine = _engine(decision)  # exactly one validated engine for this serial run
            for dataset in missing:
                train, test, metadata = _load(input_session, dataset)
                _writer_update(writer, counters, "loading", dataset, metadata)
                _writer_update(writer, counters, "dispatching", dataset, metadata)
                result = _score_accelerated(train, test, lanes, decision, engine)
                _commit(store, writer, result, metadata, counters)
        else:
            executor = ProcessPoolExecutor(max_workers=FROZEN_CONFIG.cpu_workers, initializer=_worker_initializer)
            try:
                _run_cpu_schedule(executor=executor, missing=missing, session=input_session, lanes=lanes,
                                  decision=decision, store=store, writer=writer, counters=counters)
            except KeyboardInterrupt as error:
                current = writer.progress.get("current_dataset")
                writer.mark_interrupted(current if isinstance(current, str) else None, str(writer.progress["stage"]), error); raise
            except BaseException as error:
                current = writer.progress.get("current_dataset")
                writer.mark_failed(current if isinstance(current, str) else None, str(writer.progress["stage"]), error); raise
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        _mark_checkpoint_complete(store, writer)
        return _summary(store, decision, "checkpoint_complete")
    except KeyboardInterrupt as error:
        if writer.progress.get("state") not in {"interrupted", "failed", "checkpoint_complete"}:
            writer.mark_interrupted(writer.progress.get("current_dataset") if isinstance(writer.progress.get("current_dataset"), str) else None, str(writer.progress.get("stage", "idle")), error)
        raise
    except BaseException as error:
        if writer.progress.get("state") not in {"failed", "checkpoint_complete", "interrupted"}:
            writer.mark_failed(writer.progress.get("current_dataset") if isinstance(writer.progress.get("current_dataset"), str) else None, str(writer.progress.get("stage", "idle")), error)
        raise
    finally:
        writer.stop_heartbeat(); writer.close()


def _decision(readiness: Mapping[str, object]) -> BackendDecision:
    return BackendDecision.from_payload(readiness["backend_decision"])


def _validate_stored_context(store: Exp436CheckpointStore, readiness: Mapping[str, object], readiness_path: Path, run_root: Path, checkpoint_root: Path, decision: BackendDecision) -> None:
    """Bind read-only complete/inspect routes to the exact reviewed readiness."""
    context = context_payload(store.context)
    if context.get("reviewed_commit") != readiness["reviewed_commit"] or context.get("readiness_path") != str(readiness_path.resolve()) or context.get("readiness_artifact_sha256") != _digest(dict(readiness)):
        raise ValueError("stored context does not match readiness identity")
    if context.get("output_path") != str(run_root.resolve()) or context.get("checkpoint_path") != str(checkpoint_root.resolve()):
        raise ValueError("stored context does not match runtime roots")
    if context.get("database_path") != _absolute(readiness["database_path"], "database") or context.get("database_sha256") != readiness["database_sha256"] or context.get("datasets") != readiness["datasets"] or context.get("dataset_order_sha256") != readiness["dataset_digest"]:
        raise ValueError("stored context does not match source readiness")
    config = _frozen_config_payload()
    if context.get("frozen_config") != config or context.get("frozen_config_sha256") != readiness["frozen_config_sha256"] or context.get("backend_decision") != decision.to_payload() or context.get("backend_decision_sha256") != readiness["backend_decision_sha256"]:
        raise ValueError("stored context does not match frozen readiness")
    lanes = context.get("lane_artifacts")
    if not isinstance(lanes, Mapping):
        raise ValueError("stored context lane artifacts are invalid")
    for name in ("exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest"):
        if lanes.get(name) != {"path": _absolute(readiness[f"{name}_path"], name), "sha256": readiness[f"{name}_sha256"]}:
            raise ValueError("stored context does not match lane readiness")


def run_exp436_new(*, readiness_path: Path, run_root: Path, checkpoint_root: Path, reviewed_commit: str, input_session_factory: Callable[[], object]) -> Exp436RunSummary:
    readiness = _canonical_readiness(readiness_path, reviewed_commit, run_root, checkpoint_root)
    if Path(run_root).exists() or Path(checkpoint_root).exists():
        raise ValueError("readiness-owned new roots must be absent")
    with _session_from_factory(input_session_factory) as session:
        return _run(run_root=Path(run_root), checkpoint_root=Path(checkpoint_root), readiness_path=Path(readiness_path), readiness=readiness, input_session=session, decision=_decision(readiness), resume=False)


def resume_exp436(*, readiness_path: Path, run_root: Path, checkpoint_root: Path, reviewed_commit: str, input_session_factory: Callable[[], object]) -> Exp436RunSummary:
    readiness = _canonical_readiness(readiness_path, reviewed_commit, run_root, checkpoint_root)
    store = Exp436CheckpointStore.open_classified_read_only(checkpoint_root)
    decision = _decision(readiness)
    _validate_stored_context(store, readiness, Path(readiness_path), Path(run_root), Path(checkpoint_root), decision)
    inspection = store.inspect()
    # A record-set-stable complete root needs no writer or session.  An
    # interrupted complete root (including the final intent-with-record crash
    # window) instead goes through the owner/replay/finalization path below.
    preflight = ProgressWriter.preflight_existing(run_root, run_identity=store.run_identity, checkpoint=checkpoint_root,
                                                  backend=decision.backend, total_datasets=len(store.datasets),
                                                  context_fingerprint=store.context_fingerprint,
                                                  checkpoint_inspection={"completed": inspection.completed, "datasets": tuple(store.completed_digests()), "record_digests": store.completed_digests()})
    if len(store.completed_digests()) == len(store.datasets) and preflight["progress"]["state"] == "checkpoint_complete":
        return _summary(store, decision, "checkpoint_complete")
    recoverable_final_record = "intent_with_record" in inspection.reconciliation_reasons and inspection.completed + 1 == inspection.total
    if len(store.completed_digests()) == len(store.datasets) or recoverable_final_record:
        writer = ProgressWriter.open_existing(run_root, run_identity=store.run_identity, checkpoint=checkpoint_root,
                                              backend=decision.backend, total_datasets=len(store.datasets),
                                              context_fingerprint=store.context_fingerprint,
                                              checkpoint_inspection={"completed": inspection.completed, "datasets": tuple(store.completed_digests()), "record_digests": store.completed_digests()})
        try:
            repaired = Exp436CheckpointStore.open_existing(checkpoint_root, Exp436RunContext.from_payload(context_payload(store.context)))
            _reconcile_progress(repaired, writer)
            _mark_checkpoint_complete(repaired, writer)
            return _summary(repaired, decision, "checkpoint_complete")
        finally:
            writer.stop_heartbeat(); writer.close()
    # The preflight above is intentionally before the only session factory call.
    with _session_from_factory(input_session_factory) as session:
        return _run(run_root=Path(run_root), checkpoint_root=Path(checkpoint_root), readiness_path=Path(readiness_path), readiness=readiness, input_session=session, decision=decision, resume=True)


def inspect_exp436(*, readiness_path: Path, run_root: Path, checkpoint_root: Path, reviewed_commit: str) -> Exp436RunSummary:
    readiness = _canonical_readiness(readiness_path, reviewed_commit, run_root, checkpoint_root)
    store = Exp436CheckpointStore.open_classified_read_only(checkpoint_root)
    inspection = store.inspect()
    decision = _decision(readiness)
    _validate_stored_context(store, readiness, Path(readiness_path), Path(run_root), Path(checkpoint_root), decision)
    progress = inspect_progress(run_root, run_identity=store.run_identity, checkpoint=checkpoint_root, context_fingerprint=store.context_fingerprint,
                                checkpoint_inspection={"completed": inspection.completed, "datasets": tuple(store.completed_digests()), "record_digests": store.completed_digests()})
    # The progress terminal state is the public lifecycle report.  A complete
    # checkpoint with a lagging interrupted progress stream must not be
    # misreported as terminally complete before replay/finalization.
    status = str(progress["progress"]["state"])
    return _summary(store, decision, status)


__all__ = ["run_exp436_new", "resume_exp436", "inspect_exp436", "LaneMembershipSnapshot"]
