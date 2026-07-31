"""Validated immutable records for Exp419's deterministic CPU runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config import Exp419FrozenConfig, FROZEN_CONFIG, seed_workers_for_outer_workers


STRATA = frozenset({"preprocessing_only", "diagnostic_oof", "candidate_eligible"})
WORKER_COMPLETION_STATUSES = frozenset({"completed", "failed"})
PARALLEL_RUN_STATUSES = frozenset({"completed", "failed"})
RUNTIME_BENCHMARK_STATUSES = frozenset(
    {"qualified_parallel_runtime", "no_qualified_parallel_runtime"}
)
BENCHMARK_SELECTION_WORKERS = (2, 4, 6)

_T = TypeVar("_T")


def _name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _tuple(values: object, name: str) -> tuple[_T, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return tuple(values)


def _datasets(values: object, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    datasets = _tuple(values, name)
    if not allow_empty and not datasets:
        raise ValueError(f"{name} must not be empty")
    for dataset in datasets:
        _name(dataset, name)
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"{name} contains a duplicate dataset")
    return datasets


def _sha256(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _string_mapping(values: object, name: str) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied: dict[str, str] = {}
    for key, value in values.items():
        copied[_name(key, f"{name} key")] = _name(value, f"{name} value")
    return MappingProxyType(copied)


def _digest_mapping(values: object, name: str) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied: dict[str, str] = {}
    for key, value in values.items():
        copied[_name(key, f"{name} key")] = _sha256(value, f"{name}[{key!r}]")
    return MappingProxyType(copied)


def _mapping(values: object, name: str) -> Mapping[object, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType(dict(values))


def _path(value: object, name: str) -> Path:
    if not isinstance(value, Path):
        raise ValueError(f"{name} must be a Path")
    return value


def _lane_payload(lanes: object) -> dict[str, object]:
    """Return the non-TEST lane snapshot portion of a train context."""
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        FrozenLaneSnapshot,
    )

    if not isinstance(lanes, FrozenLaneSnapshot):
        raise ValueError("lanes must be a FrozenLaneSnapshot")
    return {
        name: {
            dataset: list(indices)
            for dataset, indices in getattr(lanes, name).items()
        }
        for name in ("hard", "standard", "priority", "automatic_normal")
    }


def _lanes_from_payload(payload: object) -> object:
    from experiments.exp417_causal_multihorizon_shadow.contracts import (
        FrozenLaneSnapshot,
    )

    if not isinstance(payload, Mapping) or set(payload) != {
        "hard",
        "standard",
        "priority",
        "automatic_normal",
    }:
        raise ValueError("benchmark train context lanes are invalid")
    restored: dict[str, dict[str, tuple[int, ...]]] = {}
    for name in ("hard", "standard", "priority", "automatic_normal"):
        lane = payload[name]
        if not isinstance(lane, Mapping):
            raise ValueError("benchmark train context lane is invalid")
        restored[name] = {}
        for dataset, indices in lane.items():
            _name(dataset, "benchmark train context lane dataset")
            values = _tuple(indices, "benchmark train context lane indices")
            if any(type(index) is not int or index < 0 for index in values):
                raise ValueError("benchmark train context lane index is invalid")
            restored[name][dataset] = tuple(values)
    return FrozenLaneSnapshot(**restored)


@dataclass(frozen=True)
class TrainLengthGroupMeta:
    dataset: str
    length: int
    series_count: int

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.length, "length", minimum=1)
        _integer(self.series_count, "series_count", minimum=1)


@dataclass(frozen=True)
class DatasetCost:
    dataset: str
    cost: int
    stratum: str

    def __post_init__(self) -> None:
        _name(self.dataset, "dataset")
        _integer(self.cost, "cost", minimum=0)
        if self.stratum not in STRATA:
            raise ValueError("stratum is not an Exp419 benchmark stratum")


@dataclass(frozen=True)
class ShardAssignment:
    shard_index: int
    datasets: tuple[str, ...]
    total_cost: int

    def __post_init__(self) -> None:
        _integer(self.shard_index, "shard_index")
        object.__setattr__(self, "datasets", _datasets(self.datasets, "datasets"))
        _integer(self.total_cost, "total_cost", minimum=0)


@dataclass(frozen=True)
class ShardPlan:
    worker_count: int
    assignments: tuple[ShardAssignment, ...]
    population: tuple[str, ...]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _integer(self.worker_count, "worker_count", minimum=1)
        if self.worker_count not in FROZEN_CONFIG.worker_candidates:
            raise ValueError("worker_count is not frozen")
        assignments = _tuple(self.assignments, "assignments")
        if len(assignments) != self.worker_count or any(
            not isinstance(assignment, ShardAssignment) for assignment in assignments
        ):
            raise ValueError("assignments must contain one ShardAssignment per worker")
        indices = tuple(assignment.shard_index for assignment in assignments)
        if indices != tuple(range(self.worker_count)):
            raise ValueError("shard indices must be ordered and contiguous")
        population = _datasets(self.population, "population")
        assigned = tuple(dataset for assignment in assignments for dataset in assignment.datasets)
        if len(assigned) != len(set(assigned)):
            raise ValueError("assignments contain a duplicate dataset")
        if set(assigned) != set(population):
            raise ValueError("assignments do not cover the exact population")
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "population", population)
        payload = {
            "assignments": [
                {
                    "datasets": assignment.datasets,
                    "shard_index": assignment.shard_index,
                    "total_cost": assignment.total_cost,
                }
                for assignment in assignments
            ],
            "population": population,
            "worker_count": self.worker_count,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        object.__setattr__(self, "sha256", sha256(canonical.encode("utf-8")).hexdigest())


def _lexical_datasets(values: object, name: str) -> tuple[str, ...]:
    datasets = _datasets(values, name)
    if datasets != tuple(sorted(datasets)):
        raise ValueError(f"{name} must be lexical")
    return datasets


def _benchmark_selection_loads(values: object) -> Mapping[int, tuple[int, ...]]:
    if not isinstance(values, Mapping) or any(type(key) is not int for key in values):
        raise ValueError("shard_loads worker key must be an integer")
    if tuple(values) != BENCHMARK_SELECTION_WORKERS:
        raise ValueError("shard_loads must have exact ordered benchmark worker keys")
    copied: dict[int, tuple[int, ...]] = {}
    for worker_count in BENCHMARK_SELECTION_WORKERS:
        loads = _tuple(values[worker_count], f"shard_loads[{worker_count}]")
        if len(loads) != worker_count:
            raise ValueError("shard_loads length must match its worker count")
        copied[worker_count] = tuple(
            _integer(load, f"shard_loads[{worker_count}]", minimum=0)
            for load in loads
        )
    return MappingProxyType(copied)


def _benchmark_selection_headrooms(values: object) -> Mapping[int, float]:
    if not isinstance(values, Mapping) or any(type(key) is not int for key in values):
        raise ValueError("proxy_headrooms worker key must be an integer")
    if tuple(values) != BENCHMARK_SELECTION_WORKERS:
        raise ValueError("proxy_headrooms must have exact ordered benchmark worker keys")
    copied: dict[int, float] = {}
    for worker_count in BENCHMARK_SELECTION_WORKERS:
        copied[worker_count] = _finite(
            values[worker_count], f"proxy_headrooms[{worker_count}]"
        )
    return MappingProxyType(copied)


@dataclass(frozen=True)
class BenchmarkSelectionProof:
    anchor_datasets: tuple[str, ...]
    added_datasets: tuple[str, ...]
    selected_datasets: tuple[str, ...]
    proxy_headroom_gate: float
    total_cost: int
    shard_loads: Mapping[int, tuple[int, ...]]
    proxy_headrooms: Mapping[int, float]
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        anchors = _lexical_datasets(self.anchor_datasets, "anchor_datasets")
        additions = _lexical_datasets(self.added_datasets, "added_datasets")
        selected = _lexical_datasets(self.selected_datasets, "selected_datasets")
        if set(anchors) & set(additions):
            raise ValueError("anchor_datasets and added_datasets must be disjoint")
        if selected != tuple(sorted((*anchors, *additions))):
            raise ValueError("selected_datasets must equal the anchor and addition union")
        gate = _finite(self.proxy_headroom_gate, "proxy_headroom_gate")
        if gate != FROZEN_CONFIG.benchmark_proxy_headroom_gate:
            raise ValueError("proxy_headroom_gate is not frozen")
        total_cost = _integer(self.total_cost, "total_cost", minimum=1)
        loads = _benchmark_selection_loads(self.shard_loads)
        headrooms = _benchmark_selection_headrooms(self.proxy_headrooms)
        for worker_count in BENCHMARK_SELECTION_WORKERS:
            worker_loads = loads[worker_count]
            if sum(worker_loads) != total_cost:
                raise ValueError("shard_loads must sum to total_cost")
            maximum_load = max(worker_loads)
            if maximum_load == 0:
                raise ValueError("shard_loads must contain a positive load")
            expected_headroom = total_cost / maximum_load
            if abs(headrooms[worker_count] - expected_headroom) > 1e-12:
                raise ValueError("proxy_headrooms must match shard_loads")
        for worker_count in FROZEN_CONFIG.benchmark_proxy_headroom_workers:
            if headrooms[worker_count] < gate:
                raise ValueError("proxy_headrooms must meet the frozen gate")
        object.__setattr__(self, "anchor_datasets", anchors)
        object.__setattr__(self, "added_datasets", additions)
        object.__setattr__(self, "selected_datasets", selected)
        object.__setattr__(self, "proxy_headroom_gate", gate)
        object.__setattr__(self, "shard_loads", loads)
        object.__setattr__(self, "proxy_headrooms", headrooms)
        object.__setattr__(
            self,
            "sha256",
            sha256(canonical_json_bytes(_benchmark_selection_proof_body(self))).hexdigest(),
        )


def _benchmark_selection_proof_body(proof: BenchmarkSelectionProof) -> dict[str, object]:
    return {
        "anchor_datasets": list(proof.anchor_datasets),
        "added_datasets": list(proof.added_datasets),
        "selected_datasets": list(proof.selected_datasets),
        "proxy_headroom_gate": proof.proxy_headroom_gate,
        "total_cost": proof.total_cost,
        "shard_loads": {
            str(worker_count): list(proof.shard_loads[worker_count])
            for worker_count in BENCHMARK_SELECTION_WORKERS
        },
        "proxy_headrooms": {
            str(worker_count): proof.proxy_headrooms[worker_count]
            for worker_count in BENCHMARK_SELECTION_WORKERS
        },
    }


def benchmark_selection_proof_payload(proof: BenchmarkSelectionProof) -> dict[str, object]:
    if type(proof) is not BenchmarkSelectionProof:
        raise ValueError("proof must be a BenchmarkSelectionProof")
    return {**_benchmark_selection_proof_body(proof), "sha256": proof.sha256}


def _benchmark_selection_mapping_from_payload(
    values: object, name: str
) -> dict[int, object]:
    expected_keys = tuple(str(worker_count) for worker_count in BENCHMARK_SELECTION_WORKERS)
    if type(values) is not dict or tuple(values) != expected_keys:
        raise ValueError(f"{name} payload keys are invalid")
    return {worker_count: values[str(worker_count)] for worker_count in BENCHMARK_SELECTION_WORKERS}


def benchmark_selection_proof_from_payload(payload: object) -> BenchmarkSelectionProof:
    if type(payload) is not dict:
        raise ValueError("benchmark selection proof payload must be a dictionary")
    expected_fields = {
        "anchor_datasets",
        "added_datasets",
        "selected_datasets",
        "proxy_headroom_gate",
        "total_cost",
        "shard_loads",
        "proxy_headrooms",
        "sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("benchmark selection proof payload fields are invalid")
    supplied_sha256 = _sha256(payload["sha256"], "sha256")
    proof = BenchmarkSelectionProof(
        anchor_datasets=payload["anchor_datasets"],
        added_datasets=payload["added_datasets"],
        selected_datasets=payload["selected_datasets"],
        proxy_headroom_gate=payload["proxy_headroom_gate"],
        total_cost=payload["total_cost"],
        shard_loads=_benchmark_selection_mapping_from_payload(
            payload["shard_loads"], "shard_loads"
        ),
        proxy_headrooms=_benchmark_selection_mapping_from_payload(
            payload["proxy_headrooms"], "proxy_headrooms"
        ),
    )
    if proof.sha256 != supplied_sha256:
        raise ValueError("benchmark selection proof payload sha256 is invalid")
    return proof


@dataclass(frozen=True)
class WorkerSpec:
    shard: ShardAssignment
    checkpoint_root: Path
    outer_worker_count: int = 1
    status_root: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shard, ShardAssignment):
            raise ValueError("shard must be a ShardAssignment")
        checkpoint_root = _path(self.checkpoint_root, "checkpoint_root")
        outer_workers = _integer(
            self.outer_worker_count, "outer_worker_count", minimum=1
        )
        seed_workers_for_outer_workers(outer_workers)
        status_root = (
            checkpoint_root.parent / f"{checkpoint_root.name}.status"
            if self.status_root is None
            else _path(self.status_root, "status_root")
        )
        if _parts_are_nested_path(checkpoint_root, status_root):
            raise ValueError("status_root must be disjoint from checkpoint_root")
        object.__setattr__(self, "checkpoint_root", checkpoint_root)
        object.__setattr__(self, "outer_worker_count", outer_workers)
        object.__setattr__(self, "status_root", status_root)


def _parts_are_nested_path(first: Path, second: Path) -> bool:
    try:
        left = first.resolve(strict=False)
        right = second.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("worker root path cannot be resolved") from error
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


@dataclass(frozen=True)
class BenchmarkTrainContext:
    """Pinned Exp419 benchmark inputs with no TEST-derived field or payload."""

    database_sha256: str
    datasets: tuple[str, ...]
    lanes: object
    code_commit: str
    runtime_versions: Mapping[str, str]
    config: Exp419FrozenConfig = FROZEN_CONFIG
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.contracts import (
            FrozenLaneSnapshot,
        )

        database_sha256 = _sha256(self.database_sha256, "database_sha256")
        datasets = _lexical_datasets(self.datasets, "datasets")
        if not isinstance(self.lanes, FrozenLaneSnapshot):
            raise ValueError("lanes must be a FrozenLaneSnapshot")
        if (
            not isinstance(self.code_commit, str)
            or len(self.code_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.code_commit)
        ):
            raise ValueError("code_commit must be a lowercase Git SHA-1")
        versions = _string_mapping(self.runtime_versions, "runtime_versions")
        if not versions:
            raise ValueError("runtime_versions must not be empty")
        if not isinstance(self.config, Exp419FrozenConfig):
            raise ValueError("config must be an Exp419FrozenConfig")
        self.config.validate()
        if database_sha256 != self.config.database_sha256:
            raise ValueError("database SHA256 is not frozen")
        if len(datasets) != self.config.expected_datasets:
            raise ValueError("benchmark train population count is not frozen")
        expected_lane_hashes = (
            self.config.exp291_prediction_sha256,
            self.config.exp291_manifest_sha256,
            self.config.exp292_prediction_sha256,
            self.config.exp292_manifest_sha256,
        )
        observed_lane_hashes = (
            self.lanes.exp291_prediction_sha256,
            self.lanes.exp291_manifest_sha256,
            self.lanes.exp292_prediction_sha256,
            self.lanes.exp292_manifest_sha256,
        )
        if observed_lane_hashes != expected_lane_hashes:
            raise ValueError("benchmark train lane snapshot is not frozen")
        object.__setattr__(self, "database_sha256", database_sha256)
        object.__setattr__(self, "datasets", datasets)
        object.__setattr__(self, "runtime_versions", versions)
        object.__setattr__(
            self,
            "sha256",
            sha256(canonical_json_bytes(_benchmark_train_context_body(self))).hexdigest(),
        )


def _benchmark_train_context_body(context: BenchmarkTrainContext) -> dict[str, object]:
    if type(context) is not BenchmarkTrainContext:
        raise ValueError("context must be a BenchmarkTrainContext")
    return {
        "code_commit": context.code_commit,
        "config_sha256": sha256(canonical_json_bytes(asdict(context.config))).hexdigest(),
        "database_sha256": context.database_sha256,
        "datasets": list(context.datasets),
        "lanes": _lane_payload(context.lanes),
        "runtime_versions": dict(context.runtime_versions),
    }


def benchmark_train_context_payload(
    context: BenchmarkTrainContext,
) -> dict[str, object]:
    """Serialize a benchmark context without any TEST-derived material."""
    return {**_benchmark_train_context_body(context), "sha256": context.sha256}


def benchmark_train_context_from_payload(payload: object) -> BenchmarkTrainContext:
    if not isinstance(payload, Mapping) or set(payload) != {
        "code_commit",
        "config_sha256",
        "database_sha256",
        "datasets",
        "lanes",
        "runtime_versions",
        "sha256",
    }:
        raise ValueError("benchmark train context payload is invalid")
    expected_config_sha256 = sha256(
        canonical_json_bytes(asdict(FROZEN_CONFIG))
    ).hexdigest()
    if payload["config_sha256"] != expected_config_sha256:
        raise ValueError("benchmark train context config is invalid")
    context = BenchmarkTrainContext(
        database_sha256=payload["database_sha256"],
        datasets=_tuple(payload["datasets"], "datasets"),
        lanes=_lanes_from_payload(payload["lanes"]),
        code_commit=payload["code_commit"],
        runtime_versions=payload["runtime_versions"],
    )
    supplied_sha256 = _sha256(payload["sha256"], "sha256")
    if context.sha256 != supplied_sha256:
        raise ValueError("benchmark train context payload SHA-256 is invalid")
    return context


@dataclass(frozen=True)
class WorkerBootstrap:
    resources: object
    provenance: object
    expected_context_sha256: str
    context: object | None = None

    def __post_init__(self) -> None:
        if self.resources is None or self.provenance is None:
            raise ValueError("resources and provenance are required")
        _sha256(self.expected_context_sha256, "expected_context_sha256")


@dataclass(frozen=True)
class BenchmarkTrainBootstrap:
    resources: object
    context: BenchmarkTrainContext
    expected_context_sha256: str

    def __post_init__(self) -> None:
        if self.resources is None:
            raise ValueError("resources are required")
        if type(self.context) is not BenchmarkTrainContext:
            raise ValueError("context must be a BenchmarkTrainContext")
        expected = _sha256(
            self.expected_context_sha256, "expected_context_sha256"
        )
        if expected != self.context.sha256:
            raise ValueError("benchmark train context SHA-256 does not match bootstrap")


@dataclass(frozen=True)
class WorkerCompletion:
    shard_index: int
    pid: int
    completed_datasets: tuple[str, ...]
    record_digests: Mapping[str, str]
    elapsed_seconds: float
    peak_rss_bytes: int
    thread_environment: Mapping[str, str]
    status: str
    error: str

    def __post_init__(self) -> None:
        _integer(self.shard_index, "shard_index")
        _integer(self.pid, "pid", minimum=1)
        completed = _datasets(self.completed_datasets, "completed_datasets", allow_empty=True)
        digests = _digest_mapping(self.record_digests, "record_digests")
        if set(digests) != set(completed):
            raise ValueError("record_digests must cover the exact completed_datasets")
        _finite(self.elapsed_seconds, "elapsed_seconds")
        _integer(self.peak_rss_bytes, "peak_rss_bytes")
        environment = _string_mapping(self.thread_environment, "thread_environment")
        if self.status not in WORKER_COMPLETION_STATUSES:
            raise ValueError("status is not a worker completion status")
        if not isinstance(self.error, str):
            raise ValueError("error must be a string")
        if self.status == "completed" and self.error:
            raise ValueError("completed worker cannot have an error")
        if self.status == "failed" and not self.error:
            raise ValueError("failed worker must include an error")
        object.__setattr__(self, "completed_datasets", completed)
        object.__setattr__(self, "record_digests", digests)
        object.__setattr__(self, "thread_environment", environment)


@dataclass(frozen=True)
class ParallelRoots:
    root: Path
    worker_roots: Mapping[int, Path]
    canonical_root: Path

    def __post_init__(self) -> None:
        _path(self.root, "root")
        if not isinstance(self.worker_roots, Mapping) or not self.worker_roots:
            raise ValueError("worker_roots must be a non-empty mapping")
        worker_roots: dict[int, Path] = {}
        for index, root in self.worker_roots.items():
            worker_roots[_integer(index, "worker_roots key")] = _path(root, "worker_roots value")
        _path(self.canonical_root, "canonical_root")
        object.__setattr__(self, "worker_roots", MappingProxyType(worker_roots))


@dataclass(frozen=True)
class ParallelRun:
    plan: ShardPlan
    completions: tuple[WorkerCompletion, ...]
    record_digests: Mapping[str, str]
    summary: Mapping[object, object]
    publication_allowed: bool
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ShardPlan):
            raise ValueError("plan must be a ShardPlan")
        completions = _tuple(self.completions, "completions")
        if any(not isinstance(completion, WorkerCompletion) for completion in completions):
            raise ValueError("completions must contain WorkerCompletion records")
        indices = tuple(completion.shard_index for completion in completions)
        if indices != tuple(range(self.plan.worker_count)):
            raise ValueError("completions must cover exact plan shard indices")
        digests = _digest_mapping(self.record_digests, "record_digests")
        if type(self.publication_allowed) is not bool:
            raise ValueError("publication_allowed must be a bool")
        if self.status not in PARALLEL_RUN_STATUSES:
            raise ValueError("status is not a parallel run status")
        if self.publication_allowed != (self.status == "completed"):
            raise ValueError("publication_allowed must match terminal status")
        expected_digests: dict[str, str] = {}
        for completion, assignment in zip(completions, self.plan.assignments, strict=True):
            if completion.completed_datasets != assignment.datasets[:len(completion.completed_datasets)]:
                raise ValueError("completion datasets do not match the assigned shard")
            expected_digests.update(completion.record_digests)
        if dict(digests) != expected_digests:
            raise ValueError("record_digests must match completed worker records")
        if self.status == "completed":
            if any(completion.status != "completed" for completion in completions):
                raise ValueError("completed run requires completed worker status")
            if any(completion.completed_datasets != assignment.datasets for completion, assignment in zip(completions, self.plan.assignments, strict=True)):
                raise ValueError("completed run requires exact shard datasets")
            if set(digests) != set(self.plan.population):
                raise ValueError("record_digests must cover the exact plan population")
        elif not any(completion.status == "failed" for completion in completions):
            raise ValueError("failed run requires failed worker status")
        object.__setattr__(self, "completions", completions)
        object.__setattr__(self, "record_digests", digests)
        object.__setattr__(self, "summary", _mapping(self.summary, "summary"))


@dataclass(frozen=True)
class RuntimeTrial:
    worker_count: int
    elapsed_seconds: float
    peak_rss_bytes: int
    parity: bool
    finite: bool
    fallback: str
    completed_datasets: tuple[str, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        _integer(self.worker_count, "worker_count", minimum=1)
        if self.worker_count not in FROZEN_CONFIG.worker_candidates:
            raise ValueError("worker_count is not frozen")
        _finite(self.elapsed_seconds, "elapsed_seconds")
        _integer(self.peak_rss_bytes, "peak_rss_bytes")
        if type(self.parity) is not bool or type(self.finite) is not bool:
            raise ValueError("parity and finite must be bool")
        if not isinstance(self.fallback, str):
            raise ValueError("fallback must be a string")
        object.__setattr__(self, "completed_datasets", _datasets(self.completed_datasets, "completed_datasets"))
        _sha256(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True)
class RuntimeBenchmark:
    trials: tuple[RuntimeTrial, ...]
    selected_worker_count: int | None
    status: str
    benchmark_datasets: tuple[str, ...]
    selection_proof: BenchmarkSelectionProof
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        trials = _tuple(self.trials, "trials")
        if not trials or any(not isinstance(trial, RuntimeTrial) for trial in trials):
            raise ValueError("trials must contain RuntimeTrial records")
        counts = tuple(trial.worker_count for trial in trials)
        if counts != FROZEN_CONFIG.worker_candidates:
            raise ValueError("trials must cover exact frozen worker_candidates")
        if self.status not in RUNTIME_BENCHMARK_STATUSES:
            raise ValueError("status is not a runtime benchmark status")
        datasets = _datasets(self.benchmark_datasets, "benchmark_datasets")
        if type(self.selection_proof) is not BenchmarkSelectionProof:
            raise ValueError(
                "selection proof must be a BenchmarkSelectionProof"
            )
        try:
            proof = benchmark_selection_proof_from_payload(
                benchmark_selection_proof_payload(self.selection_proof)
            )
        except ValueError as error:
            raise ValueError("selection proof is not canonical") from error
        if datasets != proof.selected_datasets:
            raise ValueError(
                "benchmark datasets must equal selection proof datasets"
            )
        if any(trial.completed_datasets != datasets for trial in trials):
            raise ValueError("trials must cover exact benchmark completed_datasets")
        if self.status == "qualified_parallel_runtime":
            _integer(self.selected_worker_count, "selected_worker_count", minimum=2)
            if self.selected_worker_count not in counts:
                raise ValueError("selected_worker_count is not a trial")
        elif self.selected_worker_count is not None:
            raise ValueError("unqualified benchmark cannot select a worker count")
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "benchmark_datasets", datasets)
        object.__setattr__(self, "selection_proof", proof)
        payload = {
            "benchmark_datasets": datasets,
            "selection_proof": benchmark_selection_proof_payload(proof),
            "selected_worker_count": self.selected_worker_count,
            "status": self.status,
            "trials": [
                {
                    "completed_datasets": trial.completed_datasets,
                    "elapsed_seconds": trial.elapsed_seconds,
                    "evidence_sha256": trial.evidence_sha256,
                    "fallback": trial.fallback,
                    "finite": trial.finite,
                    "parity": trial.parity,
                    "peak_rss_bytes": trial.peak_rss_bytes,
                    "worker_count": trial.worker_count,
                }
                for trial in trials
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        object.__setattr__(self, "sha256", sha256(canonical.encode("utf-8")).hexdigest())
