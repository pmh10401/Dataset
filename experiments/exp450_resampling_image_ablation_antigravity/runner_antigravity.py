"""Bounded local Exp450 execution plus label-free global score composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.routing_paper import (
    POLICY_ORDER,
    build_hard_evidence_surface,
    build_policies,
)
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import score_low_frequency
from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability

from .artifacts_antigravity import (
    append_dataset_condition_failure_antigravity,
    read_dataset_condition_success_antigravity,
    verified_checkpoint_hashes_antigravity,
    verify_dataset_condition_inventory_antigravity,
    write_dataset_condition_success_antigravity,
    write_score_root_file_antigravity,
)
from .contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
    Exp450ConditionScore,
    Exp450DatasetConditionResult,
    Exp450RunContext,
    canonical_json_bytes,
)
from .scoring_antigravity import score_dataset_condition_exp450_antigravity
from .verifier_antigravity import verify_exp450_score_root_antigravity


_CONDITIONS = (
    Exp450Condition("A", "linear", "linear"),
    Exp450Condition("B", "polyphase_pchip", "linear"),
    Exp450Condition("C", "linear", "area"),
    Exp450Condition("D", "polyphase_pchip", "area"),
)
_SCORE_ROOT_DIRECTORY = "score_roots_antigravity"
_SHARD_NAME = "condition_score_shard_antigravity.json"
_MANIFEST_NAME = "condition_score_manifest_antigravity.json"


class Exp450Incomplete(RuntimeError):
    def __init__(self, pending: tuple[tuple[str, str], ...]) -> None:
        self.pending = pending
        super().__init__(f"exp450_incomplete pending={len(pending)}")


@dataclass(frozen=True, slots=True)
class Exp450ExecutionScheduleAntigravity:
    cpu_conditions: tuple[str, ...]
    mps_conditions: tuple[str, ...]
    cpu_workers: int
    maximum_concurrent_mps_jobs: int


def build_execution_schedule_antigravity(
    catalog: dict[str, object], backend: Exp450BackendDecision
) -> Exp450ExecutionScheduleAntigravity:
    """Freeze the CPU/MPS waves before any local coordinate begins."""
    if not isinstance(catalog, dict) or not isinstance(catalog.get("datasets"), list) or not isinstance(backend, Exp450BackendDecision):
        raise ValueError("Exp450 execution schedule inputs are invalid")
    if backend.tensor_backend == "mps":
        return Exp450ExecutionScheduleAntigravity(("A", "B"), ("C", "D"), backend.cpu_workers, 1)
    return Exp450ExecutionScheduleAntigravity(("A", "B", "C", "D"), (), backend.cpu_workers, 0)


def run_dataset_local_exp450_antigravity(
    values: RawDataset, condition: Exp450Condition, backend: Exp450BackendDecision
) -> Exp450DatasetConditionResult:
    """Run only one values-only dataset-condition scorer boundary."""
    return score_dataset_condition_exp450_antigravity(values, condition, backend)


def _conditions_by_id(conditions: tuple[Exp450Condition, ...]) -> dict[str, Exp450Condition]:
    if tuple(condition.condition_id for condition in conditions) != ("A", "B", "C", "D"):
        raise ValueError("Exp450 condition inventory differs")
    return {condition.condition_id: condition for condition in conditions}


def _pending(
    catalog: dict[str, object], completed: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    done = set(completed)
    return tuple((entry["dataset"], condition_id) for entry in catalog["datasets"] for condition_id in ("A", "B", "C", "D") if (entry["dataset"], condition_id) not in done)


def _verified_result_preflight_antigravity(
    raw: Mapping[str, RawDataset], results: Mapping[tuple[str, str], object], conditions: tuple[Exp450Condition, ...]
) -> None:
    """Reject every missing, extra, or falsely keyed local result before globals."""
    by_condition = _conditions_by_id(conditions)
    expected = tuple((dataset, condition_id) for dataset in raw for condition_id in by_condition)
    actual = dict(results)
    invalid = set(expected) ^ set(actual)
    for coordinate in set(expected) & set(actual):
        value = actual[coordinate]
        if not isinstance(value, Exp450DatasetConditionResult) or value.dataset != coordinate[0] or value.condition_id != coordinate[1]:
            invalid.add(coordinate)
    if invalid:
        ordered = tuple(coordinate for coordinate in expected if coordinate in invalid) + tuple(sorted(invalid - set(expected)))
        raise Exp450Incomplete(ordered)


def _persist_result(
    coordinate: tuple[str, str],
    result: object,
    *,
    root: Path,
    context: Exp450RunContext,
    catalog: dict[str, object],
    raw: Mapping[str, RawDataset],
    by_condition: Mapping[str, Exp450Condition],
    wave: int,
) -> None:
    dataset, condition_id = coordinate
    try:
        if isinstance(result, Exception):
            raise result
        if not isinstance(result, Exp450DatasetConditionResult) or result.dataset != dataset or result.condition_id != condition_id:
            raise ValueError("dataset-condition scorer result differs")
        write_dataset_condition_success_antigravity(root, context, catalog, raw[dataset], result)
    except Exception as error:
        append_dataset_condition_failure_antigravity(root, context, catalog, raw[dataset], by_condition[condition_id], wave, error)


def _run_cpu_wave(
    coordinates: tuple[tuple[str, str], ...],
    *,
    workers: int,
    root: Path,
    context: Exp450RunContext,
    catalog: dict[str, object],
    raw: Mapping[str, RawDataset],
    backend: Exp450BackendDecision,
    by_condition: Mapping[str, Exp450Condition],
    wave: int,
) -> None:
    queue = iter(coordinates)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active: dict[object, tuple[str, str]] = {}
        for _ in range(workers):
            try:
                dataset, condition_id = next(queue)
            except StopIteration:
                break
            active[pool.submit(run_dataset_local_exp450_antigravity, raw[dataset], by_condition[condition_id], backend)] = (dataset, condition_id)
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                coordinate = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    _persist_result(coordinate, error, root=root, context=context, catalog=catalog, raw=raw, by_condition=by_condition, wave=wave)
                else:
                    _persist_result(coordinate, result, root=root, context=context, catalog=catalog, raw=raw, by_condition=by_condition, wave=wave)
                try:
                    dataset, condition_id = next(queue)
                except StopIteration:
                    continue
                active[pool.submit(run_dataset_local_exp450_antigravity, raw[dataset], by_condition[condition_id], backend)] = (dataset, condition_id)


def _run_mps_wave(
    coordinates: tuple[tuple[str, str], ...],
    *,
    root: Path,
    context: Exp450RunContext,
    catalog: dict[str, object],
    raw: Mapping[str, RawDataset],
    backend: Exp450BackendDecision,
    by_condition: Mapping[str, Exp450Condition],
    wave: int,
) -> None:
    for dataset, condition_id in coordinates:
        try:
            result = run_dataset_local_exp450_antigravity(raw[dataset], by_condition[condition_id], backend)
        except Exception as error:
            _persist_result((dataset, condition_id), error, root=root, context=context, catalog=catalog, raw=raw, by_condition=by_condition, wave=wave)
        else:
            _persist_result((dataset, condition_id), result, root=root, context=context, catalog=catalog, raw=raw, by_condition=by_condition, wave=wave)


def execute_local_schedule_antigravity(
    root: Path,
    context: Exp450RunContext,
    catalog: dict[str, object],
    raw_datasets: Mapping[str, RawDataset],
    backend: Exp450BackendDecision,
    conditions: tuple[Exp450Condition, ...] = _CONDITIONS,
) -> dict[tuple[str, str], Exp450DatasetConditionResult]:
    """Run exactly two bounded waves, persisting every result before refill."""
    raw = dict(raw_datasets)
    if not isinstance(context, Exp450RunContext) or backend != context.backend:
        raise ValueError("Exp450 execution backend differs from context")
    by_condition = _conditions_by_id(tuple(conditions))
    schedule = build_execution_schedule_antigravity(catalog, backend)
    inventory = verify_dataset_condition_inventory_antigravity(root, context, catalog, raw)
    initial = _pending(catalog, inventory.completed)
    failed_wave_one = {(record.dataset, record.condition_id) for record in inventory.failures if record.wave == 1}
    if initial and not any(record.wave == 2 for record in inventory.failures):
        never_attempted = tuple(coordinate for coordinate in initial if coordinate not in failed_wave_one)
        if never_attempted:
            _execute_wave_antigravity(never_attempted, schedule=schedule, root=root, context=context, catalog=catalog, raw=raw, backend=backend, by_condition=by_condition, wave=1)
    inventory = verify_dataset_condition_inventory_antigravity(root, context, catalog, raw)
    if inventory.wave_two:
        _execute_wave_antigravity(inventory.wave_two, schedule=schedule, root=root, context=context, catalog=catalog, raw=raw, backend=backend, by_condition=by_condition, wave=2)
    inventory = verify_dataset_condition_inventory_antigravity(root, context, catalog, raw)
    pending = _pending(catalog, inventory.completed)
    if pending:
        raise Exp450Incomplete(pending)
    return {
        (dataset, condition_id): read_dataset_condition_success_antigravity(root, context, catalog, raw[dataset], by_condition[condition_id])
        for dataset, condition_id in _pending(catalog, ())
    }


def _execute_wave_antigravity(
    coordinates: tuple[tuple[str, str], ...], *, schedule: Exp450ExecutionScheduleAntigravity,
    root: Path, context: Exp450RunContext, catalog: dict[str, object], raw: Mapping[str, RawDataset],
    backend: Exp450BackendDecision, by_condition: Mapping[str, Exp450Condition], wave: int,
) -> None:
    cpu = tuple(coordinate for coordinate in coordinates if coordinate[1] in schedule.cpu_conditions)
    if cpu:
        _run_cpu_wave(cpu, workers=schedule.cpu_workers, root=root, context=context, catalog=catalog, raw=raw, backend=backend, by_condition=by_condition, wave=wave)
    mps = tuple(coordinate for coordinate in coordinates if coordinate[1] in schedule.mps_conditions)
    if mps:
        _run_mps_wave(mps, root=root, context=context, catalog=catalog, raw=raw, backend=backend, by_condition=by_condition, wave=wave)


def compose_condition_score_antigravity(
    condition: Exp450Condition,
    raw: Mapping[str, RawDataset],
    results: Mapping[tuple[str, str], Exp450DatasetConditionResult],
    backend: Exp450BackendDecision,
    timings: Mapping[str, float],
) -> Exp450ConditionScore:
    """Compose unchanged Task6/7 global science for one fully complete condition."""
    sources = {dataset: results[(dataset, condition.condition_id)].sources for dataset in raw}
    routes = {dataset: results[(dataset, condition.condition_id)].route for dataset in raw}
    hard_lanes = {dataset: route.exp253.hard for dataset, route in routes.items()}
    low = score_low_frequency(raw, hard_lanes)
    stability = classify_stability(low, raw)
    surface = build_hard_evidence_surface(sources, routes, raw)
    policies = build_policies(surface, low, stability, raw, routes)
    return Exp450ConditionScore(condition, raw, sources, routes, low, stability, policies, backend, dict(timings))


def compose_all_conditions_antigravity(
    results: Mapping[tuple[str, str], Exp450DatasetConditionResult], raw: Mapping[str, RawDataset],
    backend: Exp450BackendDecision, conditions: tuple[Exp450Condition, ...] = _CONDITIONS,
) -> dict[str, Exp450ConditionScore]:
    """Refuse all global stages until the complete dataset-condition cartesian product exists."""
    by_condition = _conditions_by_id(tuple(conditions))
    _verified_result_preflight_antigravity(raw, results, tuple(conditions))
    started = monotonic()
    scores = {}
    for condition_id, condition in by_condition.items():
        scores[condition_id] = compose_condition_score_antigravity(condition, raw, results, backend, {"total": float(monotonic() - started)})
    return scores


def _condition_record(condition: Exp450Condition) -> dict[str, object]:
    return {"condition_id": condition.condition_id, "length_mode": condition.length_mode, "image_mode": condition.image_mode}


def _score_shard(score: Exp450ConditionScore, dataset: str, entry: dict[str, object], context: Exp450RunContext) -> dict[str, object]:
    values = score.raw[dataset]
    if values.dataset != dataset or entry.get("test", {}).get("count") != len(values.test_rows) or tuple(values.test_indices) != tuple(range(len(values.test_rows))):
        raise ValueError("score shard TEST identity differs")
    policies = []
    for policy in score.policies.policies:
        scores = np.asarray(policy.scores[dataset])
        alerts = policy.alerts[dataset]
        if scores.dtype != np.dtype("float64") or scores.ndim != 1 or scores.size != len(values.test_rows) or not np.all(np.isfinite(scores)) or any(type(index) is not int for index in alerts):
            raise ValueError("score shard policy differs")
        mask = np.asarray(scores >= 0.5, dtype=bool)
        if frozenset(np.flatnonzero(mask).tolist()) != alerts:
            raise ValueError("score shard alert threshold differs")
        policies.append({"policy_id": policy.name, "alerts": sorted(alerts), "score_dtype": "float64", "scores": [float(value) for value in scores], "score_mask": [bool(value) for value in mask]})
    bare = {"schema": "exp450-condition-score-shard-antigravity.v1", "context_sha256": context.sha256, "catalog_sha256": context.catalog_sha256, "condition": _condition_record(score.condition), "backend": context.backend.artifact_record(), "dataset": dataset, "test_identity": entry["test"], "policy_order": list(POLICY_ORDER), "policies": policies}
    return {**bare, "shard_sha256": sha256(canonical_json_bytes(bare)).hexdigest()}


def freeze_condition_score_antigravity(
    score: Exp450ConditionScore, root: Path, context: Exp450RunContext, catalog: dict[str, object]
) -> dict[str, object]:
    """Freeze all four score vectors for one condition, bound to local checkpoints."""
    if not isinstance(score, Exp450ConditionScore) or score.backend != context.backend or Path(root) != context.run_root:
        raise ValueError("score root context differs")
    raw = dict(score.raw)
    entries = catalog.get("datasets")
    if not isinstance(entries, list) or tuple(raw) != tuple(entry.get("dataset") for entry in entries) or tuple(score.policies.order) != POLICY_ORDER:
        raise ValueError("score root inventory differs")
    checkpoint_hashes = verified_checkpoint_hashes_antigravity(root, context, catalog, raw, score.condition)
    descriptors = []
    condition = _condition_record(score.condition)
    for entry in entries:
        dataset = entry["dataset"]
        shard = _score_shard(score, dataset, entry, context)
        token = sha256(dataset.encode("utf-8")).hexdigest()
        parts = (_SCORE_ROOT_DIRECTORY, f"{score.condition.condition_id}_antigravity", f"{token}_antigravity", _SHARD_NAME)
        path = write_score_root_file_antigravity(root, parts, canonical_json_bytes(shard))
        descriptors.append({"dataset": dataset, "file": str(path.relative_to(root)), "sha256": sha256(path.read_bytes()).hexdigest()})
    bare = {"schema": "exp450-condition-score-manifest-antigravity.v1", "context_sha256": context.sha256, "catalog_sha256": context.catalog_sha256, "condition": condition, "backend": context.backend.artifact_record(), "datasets": entries, "checkpoint_hashes": [{"dataset": dataset, "sha256": digest} for dataset, digest in checkpoint_hashes], "shards": descriptors}
    manifest = {**bare, "manifest_sha256": sha256(canonical_json_bytes(bare)).hexdigest()}
    write_score_root_file_antigravity(root, (_SCORE_ROOT_DIRECTORY, f"{score.condition.condition_id}_antigravity", _MANIFEST_NAME), canonical_json_bytes(manifest))
    return verify_exp450_score_root_antigravity(root, context=context, catalog=catalog, condition_id=score.condition.condition_id)


def freeze_all_conditions_antigravity(
    scores: Mapping[str, Exp450ConditionScore], root: Path, context: Exp450RunContext, catalog: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Freeze and independently verify the A--D roots in approved order."""
    if tuple(scores) != ("A", "B", "C", "D"):
        raise Exp450Incomplete(tuple((dataset, condition_id) for dataset in (entry["dataset"] for entry in catalog.get("datasets", ())) for condition_id in ("A", "B", "C", "D")))
    entries = catalog.get("datasets")
    invalid: list[tuple[str, str]] = []
    if not isinstance(entries, list):
        raise Exp450Incomplete(())
    expected_raw = tuple(entry.get("dataset") for entry in entries)
    for condition_id, score in scores.items():
        if not isinstance(score, Exp450ConditionScore) or score.condition.condition_id != condition_id or tuple(score.raw) != expected_raw or tuple(score.sources) != expected_raw or tuple(score.routes) != expected_raw or score.backend != context.backend:
            invalid.extend((dataset, condition_id) for dataset in expected_raw if isinstance(dataset, str))
    if invalid:
        raise Exp450Incomplete(tuple(invalid))
    # Reread every local authority before the first create-only score-root write.
    for condition_id, score in scores.items():
        verified_checkpoint_hashes_antigravity(root, context, catalog, dict(score.raw), score.condition)
    return {condition_id: freeze_condition_score_antigravity(scores[condition_id], root, context, catalog) for condition_id in ("A", "B", "C", "D")}


__all__ = [
    "Exp450ExecutionScheduleAntigravity", "Exp450Incomplete", "build_execution_schedule_antigravity",
    "compose_all_conditions_antigravity", "compose_condition_score_antigravity", "execute_local_schedule_antigravity",
    "freeze_all_conditions_antigravity", "freeze_condition_score_antigravity", "run_dataset_local_exp450_antigravity",
]
