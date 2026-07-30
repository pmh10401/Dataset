#!/usr/bin/env python3
"""Local-only frozen-subset runtime benchmark for Exp419."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import resource
import subprocess
import sys
from time import monotonic
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    assert_label_free_sql_contract,
    _verified_database_connection,
    freeze_lane_inputs,
    freeze_run_context,
    verify_input_snapshots,
)
from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources
from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    RuntimeBenchmark,
    RuntimeTrial,
    WorkerBootstrap,
    WorkerSpec,
)
from experiments.exp419_deterministic_cpu_parallel.coordinator import (
    _context_sha256,
    select_runtime,
)
from experiments.exp419_deterministic_cpu_parallel.planner import (
    build_shard_plan,
    estimate_dataset_cost,
    select_benchmark_datasets,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    TrainLengthGroupMeta,
)
from experiments.exp419_deterministic_cpu_parallel.workers import spawn_workers
from scratch.run_exp417_causal_multihorizon_shadow import (
    _backend_audit_path,
    _provenance,
)


BENCHMARK_FILE = "EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json"


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def _new_root(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if path.exists():
        raise argparse.ArgumentTypeError("benchmark root must be new")
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError("benchmark parent must exist")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local-only Exp419 label-free benchmark for exactly "
            "1, 2, 4, and 6 CPU workers."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--database", type=_existing_file, required=True)
    parser.add_argument("--exp291-prediction", type=_existing_file, required=True)
    parser.add_argument("--exp291-manifest", type=_existing_file, required=True)
    parser.add_argument("--exp292-prediction", type=_existing_file, required=True)
    parser.add_argument("--exp292-manifest", type=_existing_file, required=True)
    parser.add_argument("--benchmark-root", type=_new_root, required=True)
    return parser


def require_clean_git(repository: Path = REPO_ROOT) -> str:
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("Exp419 benchmark requires a clean Git worktree")
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40:
        raise ValueError("Exp419 benchmark could not verify Git HEAD")
    return head


@dataclass(frozen=True)
class FrozenBenchmarkInputs:
    context: Any
    resources: RunnerResources
    costs: tuple[DatasetCost, ...]
    bootstrap: WorkerBootstrap
    paths: tuple[Path, ...]


def costs_from_metadata_rows(
    rows: Sequence[tuple[object, object, object]],
    *,
    expected_datasets: Sequence[str],
) -> tuple[DatasetCost, ...]:
    grouped: dict[str, list[TrainLengthGroupMeta]] = {}
    for row in rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], str)
            or type(row[1]) is not int
            or row[1] <= 0
            or row[1] % 4
            or type(row[2]) is not int
            or row[2] <= 0
        ):
            raise ValueError("TRAIN metadata row is invalid")
        grouped.setdefault(row[0], []).append(
            TrainLengthGroupMeta(row[0], row[1] // 4, row[2])
        )
    expected = tuple(expected_datasets)
    if tuple(sorted(grouped)) != expected:
        raise ValueError("TRAIN metadata does not cover the exact frozen population")
    return tuple(
        estimate_dataset_cost(dataset, tuple(grouped[dataset]))
        for dataset in expected
    )


def _load_train_metadata_costs(database: Path, context: Any) -> tuple[DatasetCost, ...]:
    query = """
    SELECT d.name, length(i.values_blob), count(*)
    FROM instances AS i
    JOIN datasets AS d ON d.id = i.dataset_id
    WHERE i.split = 'TRAIN' AND i.label = '0'
    GROUP BY d.name, length(i.values_blob)
    ORDER BY d.name, length(i.values_blob)
    """
    with _verified_database_connection(
        database, context.config.database_sha256
    ) as connection:
        rows = connection.execute(query).fetchall()
    return costs_from_metadata_rows(rows, expected_datasets=context.datasets)


def freeze_benchmark_inputs(args: argparse.Namespace, verified_head: str) -> FrozenBenchmarkInputs:
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    lanes = freeze_lane_inputs(*paths[1:])
    context = freeze_run_context(paths[0], lanes, verified_code_commit=verified_head)
    resources = RunnerResources(
        *paths, backend_audit_path=_backend_audit_path(paths[0])
    )
    provenance = _provenance(context)
    costs = _load_train_metadata_costs(paths[0], context)
    return FrozenBenchmarkInputs(
        context,
        resources,
        costs,
        WorkerBootstrap(resources, provenance, _context_sha256(context)),
        paths,
    )


def prepare_trial_roots(root: Path) -> Mapping[int, Path]:
    root = Path(root).resolve(strict=False)
    if root.exists():
        raise ValueError("benchmark root must be new")
    if not root.parent.is_dir():
        raise ValueError("benchmark parent must exist")
    root.mkdir()
    return {
        count: root / f"workers-{count}"
        for count in FROZEN_CONFIG.worker_candidates
    }


def _run_trials(
    frozen: FrozenBenchmarkInputs,
    trial_roots: Mapping[int, Path],
) -> RuntimeBenchmark:
    selected = select_benchmark_datasets(frozen.costs)
    if not selected:
        raise ValueError("frozen benchmark subset is empty")
    selected_set = set(selected)
    subset_costs = tuple(cost for cost in frozen.costs if cost.dataset in selected_set)
    if tuple(sorted(cost.dataset for cost in subset_costs)) != selected:
        raise ValueError("benchmark subset differs from frozen TRAIN-only selection")

    trials: list[RuntimeTrial] = []
    reference_digests: dict[str, str] | None = None
    for worker_count in FROZEN_CONFIG.worker_candidates:
        root = trial_roots[worker_count]
        if root.exists():
            raise ValueError("every benchmark candidate root must be fresh")
        root.mkdir()
        plan = build_shard_plan(subset_costs, worker_count)
        specs = tuple(
            WorkerSpec(
                assignment,
                root / f"worker-{assignment.shard_index}",
            )
            for assignment in plan.assignments
        )
        started = monotonic()
        completions = spawn_workers(specs, frozen.bootstrap)
        elapsed = monotonic() - started
        record_digests = {
            dataset: digest
            for completion in completions
            for dataset, digest in completion.record_digests.items()
        }
        completed = (
            tuple(sorted(record_digests))
            if set(record_digests) == set(selected)
            else selected
        )
        finite = (
            elapsed > 0.0
            and all(completion.status == "completed" for completion in completions)
            and set(record_digests) == set(selected)
        )
        if worker_count == 1 and finite:
            reference_digests = dict(record_digests)
        parity = finite and reference_digests is not None and record_digests == reference_digests
        fallback = "" if finite else "; ".join(
            completion.error
            for completion in completions
            if completion.error
        ) or "incomplete benchmark trial"
        evidence = sha256(
            canonical_json_bytes(
                {
                    "completed_datasets": completed,
                    "record_digests": record_digests,
                }
            )
        ).hexdigest()
        parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            parent_peak_bytes = int(parent_peak)
        else:
            parent_peak_bytes = int(parent_peak) * 1024
        peak_rss = parent_peak_bytes + sum(
            completion.peak_rss_bytes for completion in completions
        )
        trials.append(
            RuntimeTrial(
                worker_count,
                elapsed,
                peak_rss,
                parity,
                finite,
                fallback,
                selected,
                evidence,
            )
        )
    return select_runtime(tuple(trials))


def _trial_payload(trial: RuntimeTrial) -> dict[str, object]:
    return {
        "worker_count": trial.worker_count,
        "elapsed_seconds": trial.elapsed_seconds,
        "peak_rss_bytes": trial.peak_rss_bytes,
        "parity": trial.parity,
        "finite": trial.finite,
        "fallback": trial.fallback,
        "completed_datasets": list(trial.completed_datasets),
        "evidence_sha256": trial.evidence_sha256,
    }


def benchmark_payload(benchmark: RuntimeBenchmark) -> dict[str, object]:
    return {
        "trials": [_trial_payload(trial) for trial in benchmark.trials],
        "selected_worker_count": benchmark.selected_worker_count,
        "status": benchmark.status,
        "benchmark_datasets": list(benchmark.benchmark_datasets),
        "sha256": benchmark.sha256,
    }


def benchmark_from_payload(payload: object) -> RuntimeBenchmark:
    if not isinstance(payload, dict) or set(payload) != {
        "trials",
        "selected_worker_count",
        "status",
        "benchmark_datasets",
        "sha256",
    }:
        raise ValueError("benchmark evidence fields are invalid")
    raw_trials = payload["trials"]
    if not isinstance(raw_trials, list):
        raise ValueError("benchmark trials are invalid")
    trials = []
    exact_fields = {
        "worker_count",
        "elapsed_seconds",
        "peak_rss_bytes",
        "parity",
        "finite",
        "fallback",
        "completed_datasets",
        "evidence_sha256",
    }
    for raw in raw_trials:
        if not isinstance(raw, dict) or set(raw) != exact_fields:
            raise ValueError("benchmark trial fields are invalid")
        trials.append(
            RuntimeTrial(
                raw["worker_count"],
                raw["elapsed_seconds"],
                raw["peak_rss_bytes"],
                raw["parity"],
                raw["finite"],
                raw["fallback"],
                raw["completed_datasets"],
                raw["evidence_sha256"],
            )
        )
    benchmark = RuntimeBenchmark(
        tuple(trials),
        payload["selected_worker_count"],
        payload["status"],
        payload["benchmark_datasets"],
    )
    if benchmark.sha256 != payload["sha256"]:
        raise ValueError("benchmark digest mismatch")
    if select_runtime(benchmark.trials) != benchmark:
        raise ValueError("benchmark selection is not the frozen passing selection")
    return benchmark


def write_benchmark_evidence(
    path: Path,
    *,
    benchmark: RuntimeBenchmark,
    git_commit: str,
    context_sha256: str,
    trial_roots: Sequence[Path],
) -> None:
    roots = tuple(Path(root).resolve(strict=False) for root in trial_roots)
    if len(roots) != len(FROZEN_CONFIG.worker_candidates) or len(set(roots)) != len(roots):
        raise ValueError("benchmark evidence requires exact disjoint trial roots")
    payload = {
        "artifact_kind": "exp419_non_official_runtime_benchmark",
        "benchmark": benchmark_payload(benchmark),
        "context_sha256": context_sha256,
        "git_commit": git_commit,
        "schema_version": 1,
        "trial_roots": [str(root) for root in roots],
        "worker_candidates": list(FROZEN_CONFIG.worker_candidates),
    }
    canonical = canonical_json_bytes(payload) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical)


@dataclass(frozen=True)
class BenchmarkStages:
    clean_git: Callable[[], str]
    label_guard: Callable[[], Any]
    freeze_inputs: Callable[[str], FrozenBenchmarkInputs]
    prepare_roots: Callable[[], Mapping[int, Path]]
    run_trials: Callable[[FrozenBenchmarkInputs, Mapping[int, Path]], RuntimeBenchmark]
    revalidate: Callable[[], Any]
    write: Callable[[RuntimeBenchmark, str, FrozenBenchmarkInputs, Mapping[int, Path]], Any]


def execute_benchmark(stages: BenchmarkStages) -> int:
    head = stages.clean_git()
    stages.label_guard()
    frozen = stages.freeze_inputs(head)
    roots = stages.prepare_roots()
    benchmark = stages.run_trials(frozen, roots)
    stages.revalidate()
    stages.write(benchmark, head, frozen, roots)
    return 0


def make_test_stages(*, clean_git: Callable[[], str]) -> BenchmarkStages:
    unreachable = lambda *_args: (_ for _ in ()).throw(
        AssertionError("later stage reached")
    )
    return BenchmarkStages(
        clean_git,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
    )


def execute_for_test(stages: BenchmarkStages) -> int:
    return execute_benchmark(stages)


def _build_stages(args: argparse.Namespace) -> BenchmarkStages:
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    state: dict[str, Any] = {}

    def freeze(head: str) -> FrozenBenchmarkInputs:
        frozen = freeze_benchmark_inputs(args, head)
        state["head"] = head
        return frozen

    def revalidate() -> None:
        if require_clean_git() != state["head"]:
            raise ValueError("Git HEAD changed during Exp419 benchmark")
        verify_input_snapshots(*paths)

    def write(
        benchmark: RuntimeBenchmark,
        head: str,
        frozen: FrozenBenchmarkInputs,
        roots: Mapping[int, Path],
    ) -> None:
        write_benchmark_evidence(
            args.benchmark_root / BENCHMARK_FILE,
            benchmark=benchmark,
            git_commit=head,
            context_sha256=_context_sha256(frozen.context),
            trial_roots=tuple(roots[count] for count in FROZEN_CONFIG.worker_candidates),
        )

    return BenchmarkStages(
        require_clean_git,
        assert_label_free_sql_contract,
        freeze,
        lambda: prepare_trial_roots(args.benchmark_root),
        _run_trials,
        revalidate,
        write,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return execute_benchmark(_build_stages(args))


if __name__ == "__main__":
    raise SystemExit(main())
