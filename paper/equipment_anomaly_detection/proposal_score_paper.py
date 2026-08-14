"""Narrow Task4--7 score composition for the frozen proposal population."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
from time import monotonic
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from .artifacts_paper import Task4Task5SuccessAuthority, append_task4_task5_attempt_record, freeze_score_root, read_task4_task5_success_record, verify_task4_task5_attempt_namespace, verify_task4_task5_success_namespace, write_task4_task5_success_record, write_typed_proposal_shard
from .contracts_paper import FrozenTask8BackendDecision, LowFrequencyResult, RawDataset, Task8FrozenManifest
from .data_paper import PROGRESS_TICK_INTERVAL_S, ProgressCallback, emit_progress_paper
from .hard_route_paper import HardRouteResult, replay_hard_route
from .routing_paper import PolicyBundle, build_hard_evidence_surface, build_policies
from .sources_paper.canonical_paper import verify_pycatch22_environment_paper
from .sources_paper.hard_sources_paper import HardSourceBundle, score_hard_sources
from .sources_paper.low_frequency_paper import score_low_frequency
from .sources_paper.minirocket_paper import MiniRocketResult, score_minirocket_confirmations
from .sources_paper.multirocket_hydra_paper import verify_exp84_environment_paper
from .sources_paper.preprocessing_paper import DependencyUnavailablePaper
from .sources_paper.stability_paper import StabilityPartition, classify_stability


@dataclass(frozen=True, slots=True)
class ProposalScore:
    raw: Mapping[str, RawDataset]
    sources: Mapping[str, HardSourceBundle]
    routes: Mapping[str, HardRouteResult]
    low_frequency: LowFrequencyResult
    stability: StabilityPartition
    policies: PolicyBundle
    backend: FrozenTask8BackendDecision
    stage_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        raw, sources, routes, timing = dict(self.raw), dict(self.sources), dict(self.routes), dict(self.stage_seconds)
        if not raw or tuple(raw) != tuple(sorted(raw)) or set(raw) != set(sources) or set(raw) != set(routes) or set(raw) != set(self.low_frequency.datasets):
            raise ValueError("proposal score population identities differ")
        if any(not isinstance(value, RawDataset) or value.dataset != dataset for dataset, value in raw.items()) or any(not isinstance(value, HardSourceBundle) for value in sources.values()) or any(not isinstance(value, HardRouteResult) for value in routes.values()):
            raise ValueError("proposal score typed intermediates differ")
        if not isinstance(self.stability, StabilityPartition) or not isinstance(self.policies, PolicyBundle) or not isinstance(self.backend, FrozenTask8BackendDecision) or any(type(value) is not float or value < 0.0 for value in timing.values()):
            raise ValueError("proposal score result is invalid")
        object.__setattr__(self, "raw", MappingProxyType(raw))
        object.__setattr__(self, "sources", MappingProxyType(sources))
        object.__setattr__(self, "routes", MappingProxyType(routes))
        object.__setattr__(self, "stage_seconds", MappingProxyType(timing))


class ProposalScoreIncomplete(RuntimeError):
    """Typed operational result: Task4+5 lacks exact verified success coverage."""
    status = "proposal_score_incomplete"

    def __init__(self, *, completed: int, failed: int, pending: tuple[str, ...], attempts_path: str | None) -> None:
        self.completed = completed
        self.failed = failed
        self.pending = pending
        self.attempts_path = attempts_path
        super().__init__(f"proposal_score_incomplete completed={completed} failed={failed} pending={len(pending)} queue={','.join(pending)}")

    def artifact_record(self) -> dict[str, object]:
        return {"status": self.status, "completed": self.completed, "failed": self.failed, "pending": len(self.pending),
                "pending_datasets": list(self.pending), "attempts_path": self.attempts_path}


def _label_firewall() -> None:
    if "paper.equipment_anomaly_detection.label_access_paper" in sys.modules:
        raise RuntimeError("label_access_paper must remain unloaded during proposal scoring")


_NUMERICAL_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _pin_inner_numerical_threads():
    previous = {name: os.environ.get(name) for name in _NUMERICAL_THREAD_ENV}
    try:
        for name in _NUMERICAL_THREAD_ENV:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _score_dataset_task4_task5(raw: RawDataset) -> tuple[HardSourceBundle, HardRouteResult]:
    sources = score_hard_sources(raw)
    mini = score_minirocket_confirmations(raw)
    return sources, replay_hard_route(raw, sources, mini)


def _progress(stage: str, completed: int, total: int, started: float, dataset: str | None = None) -> None:
    elapsed = float(monotonic() - started)
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = total - completed
    eta = remaining / rate if rate > 0.0 else 0.0
    suffix = f" dataset={dataset}" if dataset is not None else ""
    print(f"{stage} {completed}/{total} {100.0 * completed / total:.1f}% elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s{suffix}", file=sys.stderr, flush=True)


def make_stderr_progress_sink_paper(
    *,
    min_tick_interval_s: float = PROGRESS_TICK_INTERVAL_S,
    clock=monotonic,
    stream=None,
) -> ProgressCallback:
    """Format one flushed stderr line. Rate-gate ticks only."""
    output = sys.stderr if stream is None else stream
    last_tick: dict[str, float] = {}

    def sink(event: Mapping[str, object]) -> None:
        stage = str(event["stage"])
        phase = str(event["phase"])
        now = float(clock())
        if phase == "start":
            last_tick.pop(stage, None)
        elif phase == "tick":
            previous = last_tick.get(stage)
            if previous is not None and (now - previous) < min_tick_interval_s:
                return
            last_tick[stage] = now
        completed = int(event["completed"])
        total = int(event["total"])
        elapsed = float(event.get("elapsed_s", 0.0))
        pct = (100.0 * completed / total) if total > 0 else 0.0
        rate = (completed / elapsed) if elapsed > 0.0 else 0.0
        eta = ((total - completed) / rate) if rate > 0.0 else 0.0
        suffix = ""
        dataset = event.get("dataset")
        if type(dataset) is str:
            suffix += f" dataset={dataset}"
        detail = event.get("detail")
        if type(detail) is str:
            suffix += f" {detail}"
        print(
            f"{stage} {completed}/{total} {pct:.1f}% elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s{suffix}",
            file=output,
            flush=True,
        )

    return sink


Task4Task5SuccessPublisher = Callable[[RawDataset, HardSourceBundle, HardRouteResult], None]


def task4_task5_success_publisher(root, catalog: dict[str, object], authority: Task4Task5SuccessAuthority) -> Task4Task5SuccessPublisher:
    """Return a parent-only publisher; workers receive neither root nor authority."""
    if not isinstance(authority, Task4Task5SuccessAuthority):
        raise TypeError("Task4+5 success authority is required")
    def publish(raw: RawDataset, sources: HardSourceBundle, route: HardRouteResult) -> None:
        write_task4_task5_success_record(root, catalog, authority, raw, sources, route)
    return publish


def _risk_first_queue(raw: Mapping[str, RawDataset]) -> tuple[str, ...]:
    """Label-free operational ordering; completed records remain catalog-sorted on reconstruction."""
    def risk(values: RawDataset) -> tuple[int, str]:
        rows = tuple(values.train_rows) + tuple(values.test_rows)
        finite = tuple(bool(np.all(np.isfinite(row))) for row in rows)
        if not all(finite):
            return (0 if any(finite) else 1, values.dataset)
        if len(values.train_rows) < 4:
            return (2, values.dataset)
        if len(values.train_rows) >= 100:
            return (3, values.dataset)
        if values.metadata_length >= 1024:
            return (4, values.dataset)
        return (5, values.dataset)
    return tuple(sorted(raw, key=lambda dataset: risk(raw[dataset])))


def _collect_task4_task5(raw: Mapping[str, RawDataset], backend: FrozenTask8BackendDecision, *, root=None, context: dict[str, object] | None = None, catalog: dict[str, object] | None = None, acceleration_manifest: object | None = None, publish_success: Task4Task5SuccessPublisher | None = None, progress: ProgressCallback | None = None) -> tuple[dict[str, HardSourceBundle], dict[str, HardRouteResult]]:
    verify_exp84_environment_paper()
    verify_pycatch22_environment_paper()
    recovery_inputs = (root is not None, context is not None, catalog is not None)
    if any(recovery_inputs) and (not all(recovery_inputs) or acceleration_manifest is None):
        raise ValueError("Task4+5 recovery inputs are incomplete")
    authority = None
    sources: dict[str, HardSourceBundle] = {}
    routes: dict[str, HardRouteResult] = {}
    failed: list[str] = []
    pending_raw = dict(raw)
    wave = 1
    if all(recovery_inputs):
        if tuple(entry.get("dataset") for entry in catalog.get("datasets", ())) != tuple(raw) or any(not isinstance(values, RawDataset) or values.dataset != dataset for dataset, values in raw.items()):
            raise ValueError("Task4+5 catalog raw identity differs")
        authority = Task4Task5SuccessAuthority.from_verified_context(context, catalog, acceleration_manifest)
        verified = verify_task4_task5_success_namespace(root, catalog, authority, dict(raw))
        attempts = verify_task4_task5_attempt_namespace(root, catalog, authority, dict(raw))
        wave = max((record.wave for record in attempts), default=0) + 1
        for dataset in verified:
            record = read_task4_task5_success_record(root, catalog, authority, raw[dataset])
            sources[dataset], routes[dataset] = record.sources, record.route
            pending_raw.pop(dataset)
        if publish_success is None:
            publish_success = task4_task5_success_publisher(root, catalog, authority)
    queue = _risk_first_queue(pending_raw)
    if not queue:
        return dict(sorted(sources.items())), dict(sorted(routes.items()))
    started, total = monotonic(), len(queue)
    _progress("Task4+Task5 start", 0, total, started)
    emit_progress_paper(progress, {"stage": "task4_task5", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    def persist_result(dataset: str, result: tuple[HardSourceBundle, HardRouteResult]) -> None:
        sources[dataset], routes[dataset] = result
        if publish_success is not None:
            publish_success(raw[dataset], sources[dataset], routes[dataset])

    def persist_failure(dataset: str, error: Exception) -> None:
        failed.append(dataset)
        if authority is not None:
            append_task4_task5_attempt_record(root, catalog, authority, raw[dataset], wave, error)

    if backend.dataset_workers == 1:
        for completed, dataset in enumerate(queue, 1):
            try:
                persist_result(dataset, _score_dataset_task4_task5(raw[dataset]))
            except DependencyUnavailablePaper:
                raise
            except Exception as error:
                persist_failure(dataset, error)
            _progress("Task4+Task5", completed, total, started, dataset)
            elapsed = float(monotonic() - started)
            if dataset in failed:
                emit_progress_paper(progress, {"stage": "task4_task5", "phase": "error", "completed": completed, "total": total, "dataset": dataset, "elapsed_s": elapsed})
            elif completed == total and not failed:
                emit_progress_paper(progress, {"stage": "task4_task5", "phase": "end", "completed": completed, "total": total, "elapsed_s": elapsed})
    elif queue:
      with _pin_inner_numerical_threads():
        with ProcessPoolExecutor(max_workers=backend.dataset_workers) as executor:
            futures = {}
            iterator = iter(queue)
            window = min(len(queue), 2 * backend.dataset_workers)
            for _ in range(window):
                dataset = next(iterator)
                futures[executor.submit(_score_dataset_task4_task5, raw[dataset])] = dataset
            completed = 0
            while futures:
                future = next(iter(as_completed(tuple(futures))))
                dataset = futures.pop(future)
                completed += 1
                try:
                    persist_result(dataset, future.result())
                except DependencyUnavailablePaper:
                    raise
                except Exception as error:
                    persist_failure(dataset, error)
                _progress("Task4+Task5", completed, total, started, dataset)
                elapsed = float(monotonic() - started)
                if dataset in failed:
                    emit_progress_paper(progress, {"stage": "task4_task5", "phase": "error", "completed": completed, "total": total, "dataset": dataset, "elapsed_s": elapsed})
                elif completed == total and not failed:
                    emit_progress_paper(progress, {"stage": "task4_task5", "phase": "end", "completed": completed, "total": total, "elapsed_s": elapsed})
                try:
                    next_dataset = next(iterator)
                except StopIteration:
                    continue
                futures[executor.submit(_score_dataset_task4_task5, raw[next_dataset])] = next_dataset
    if failed:
        pending = tuple(sorted(failed))
        attempts_path = str(root / "stage_artifacts_paper" / "proposal_work" / "task4_task5_attempts") if root is not None else None
        raise ProposalScoreIncomplete(completed=len(sources), failed=len(failed), pending=pending, attempts_path=attempts_path)
    return dict(sorted(sources.items())), dict(sorted(routes.items()))


def compose_proposal_score(raw_datasets: Mapping[str, RawDataset], acceleration_manifest: object, *, root=None, context: dict[str, object] | None = None, catalog: dict[str, object] | None = None, progress: ProgressCallback | None = None) -> ProposalScore:
    """Execute Task4+5 independently, then exactly one population-global Task6 and selected-only Task7."""
    _label_firewall()
    raw = dict(raw_datasets)
    if not raw or tuple(raw) != tuple(sorted(raw)) or any(not isinstance(values, RawDataset) or values.dataset != dataset for dataset, values in raw.items()):
        raise ValueError("proposal raw dataset mapping is invalid")
    backend = FrozenTask8BackendDecision.from_acceleration_manifest(acceleration_manifest)
    supplied = (root is not None, context is not None, catalog is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("proposal success publication inputs are incomplete")
    publisher = None
    if all(supplied):
        authority = Task4Task5SuccessAuthority.from_verified_context(context, catalog, acceleration_manifest)
        publisher = task4_task5_success_publisher(root, catalog, authority)
    started = monotonic()
    sources, routes = _collect_task4_task5(raw, backend, root=root, context=context, catalog=catalog, acceleration_manifest=acceleration_manifest, publish_success=publisher, progress=progress)
    hard_seconds = float(monotonic() - started)
    _label_firewall()
    print("Task6 start", file=sys.stderr, flush=True)
    low_started = monotonic()
    hard_lanes = {dataset: routes[dataset].exp253.hard for dataset in raw}
    low = score_low_frequency(raw, hard_lanes) if progress is None else score_low_frequency(raw, hard_lanes, progress=progress)
    low_seconds = float(monotonic() - low_started)
    print("Task6 finish", file=sys.stderr, flush=True)
    _label_firewall()
    print("Task7 start", file=sys.stderr, flush=True)
    stability_started = monotonic()
    stability = classify_stability(low, raw) if progress is None else classify_stability(low, raw, progress=progress)
    stability_seconds = float(monotonic() - stability_started)
    print("Task7 finish", file=sys.stderr, flush=True)
    surface = build_hard_evidence_surface(sources, routes, raw) if progress is None else build_hard_evidence_surface(sources, routes, raw, progress=progress)
    policies = build_policies(surface, low, stability, raw, routes) if progress is None else build_policies(surface, low, stability, raw, routes, progress=progress)
    _label_firewall()
    return ProposalScore(raw, sources, routes, low, stability, policies, backend, {"task4_task5": hard_seconds, "task6": low_seconds, "task7": stability_seconds, "total": float(monotonic() - started)})


def write_and_freeze_proposal_score(root, catalog: dict[str, object], proposal: ProposalScore, *, progress: ProgressCallback | None = None) -> Task8FrozenManifest:
    """Pass only Task4--7 typed values to Task8's create-only shard writer."""
    if not isinstance(proposal, ProposalScore) or tuple(entry["dataset"] for entry in catalog.get("datasets", ())) != tuple(proposal.raw):
        raise ValueError("proposal score catalog identity differs")
    names = tuple(proposal.raw)
    started = monotonic()
    total = len(names)
    completed = 0
    emit_progress_paper(progress, {"stage": "shard_write", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    try:
        for completed, dataset in enumerate(names, 1):
            write_typed_proposal_shard(
                root, catalog, proposal.raw[dataset], proposal.sources[dataset],
                proposal.routes[dataset], proposal.low_frequency, proposal.stability,
                proposal.policies, proposal.backend,
            )
            emit_progress_paper(progress, {
                "stage": "shard_write", "phase": "tick",
                "completed": completed, "total": total,
                "dataset": dataset, "detail": f"{dataset}_paper.json",
                "elapsed_s": float(monotonic() - started),
            })
        emit_progress_paper(progress, {"stage": "shard_write", "phase": "end", "completed": total, "total": total, "elapsed_s": float(monotonic() - started)})
    except Exception:
        emit_progress_paper(progress, {"stage": "shard_write", "phase": "error", "completed": completed if names else 0, "total": total, "elapsed_s": float(monotonic() - started)})
        raise
    return freeze_score_root(root, catalog, progress=progress)


__all__ = ["ProposalScore", "ProposalScoreIncomplete", "Task4Task5SuccessPublisher", "compose_proposal_score", "make_stderr_progress_sink_paper", "task4_task5_success_publisher", "write_and_freeze_proposal_score"]
