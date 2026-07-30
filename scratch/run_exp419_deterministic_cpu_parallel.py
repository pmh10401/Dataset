#!/usr/bin/env python3
"""Separately approved all-population Exp419 deterministic CPU CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointStore,
)
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    verify_input_snapshots,
)
from experiments.exp417_causal_multihorizon_shadow.reporting import (
    TrustedCheckpointProvenance,
)
from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    ParallelRoots,
    RuntimeBenchmark,
)
from experiments.exp419_deterministic_cpu_parallel.coordinator import (
    _context_sha256,
    resume_or_run_parallel,
)
from experiments.exp419_deterministic_cpu_parallel.planner import (
    build_shard_plan,
    select_balanced_benchmark_datasets,
)
from experiments.exp419_deterministic_cpu_parallel.publication import publish_exp419
from experiments.exp419_deterministic_cpu_parallel.reporting import (
    render_exp419_payloads,
    verify_exp419_payloads,
)
from scratch.benchmark_exp419_deterministic_cpu_parallel import (
    FrozenBenchmarkInputs,
    assert_exp419_label_isolation,
    benchmark_from_payload,
    freeze_benchmark_inputs,
    require_clean_git,
    runtime_measurements_from_payload,
    validate_benchmark_source_manifest_binding,
)


APPROVAL_PHRASE = "Exp419 1,117개 전체 label-free 병렬 실행 승인"
TERMINATED_DIAGNOSTIC_ROOT = Path(
    "/Users/mac/Documents/Dataset/"
    "exp419_deterministic_cpu_parallel_benchmark"
).resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Exp419 only after the user has explicitly said: "
            f"{APPROVAL_PHRASE}. This command never infers approval from "
            "environment variables or files."
        ),
        allow_abbrev=False,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=160),
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--exp291-prediction", type=Path, required=True)
    parser.add_argument("--exp291-manifest", type=Path, required=True)
    parser.add_argument("--exp292-prediction", type=Path, required=True)
    parser.add_argument("--exp292-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        metadata = os.stat(path, follow_symlinks=True)
    except FileNotFoundError:
        return None
    return metadata.st_dev, metadata.st_ino


def require_disjoint_roots(checkpoint_dir: Path, output_dir: Path) -> None:
    checkpoint = Path(checkpoint_dir).resolve(strict=False)
    output_source = Path(output_dir)
    output = output_source.resolve(strict=False)
    if (
        checkpoint == output
        or checkpoint in output.parents
        or output in checkpoint.parents
    ):
        raise ValueError("Exp419 checkpoint and output roots must be disjoint")
    checkpoint_identity = _identity(checkpoint)
    output_identity = _identity(output_source)
    if (
        checkpoint_identity is not None
        and output_identity is not None
        and checkpoint_identity == output_identity
    ):
        raise ValueError("Exp419 checkpoint and output roots contain an alias")
    if output_source.exists():
        raise ValueError("Exp419 official output root must be new")


def _read_json_regular(path: Path) -> object:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ValueError("benchmark evidence cannot be opened") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink < 1:
        raise ValueError("benchmark evidence must be a linked regular file")
    try:
        return json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("benchmark evidence is invalid JSON") from error


def verify_benchmark_evidence(
    path: Path,
    *,
    expected_git_commit: str,
    expected_context_sha256: str,
    expected_source_manifest_sha256: str,
    expected_costs: Sequence[DatasetCost],
) -> RuntimeBenchmark:
    payload = _read_json_regular(Path(path))
    exact_fields = {
        "artifact_kind",
        "benchmark",
        "context_sha256",
        "git_commit",
        "runtime_measurements",
        "schema_version",
        "source_manifest_sha256",
        "trial_roots",
        "worker_candidates",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != exact_fields
        or payload["artifact_kind"]
        != "exp419_non_official_runtime_benchmark"
        or payload["schema_version"] != 1
        or payload["git_commit"] != expected_git_commit
        or payload["context_sha256"] != expected_context_sha256
        or payload["source_manifest_sha256"]
        != expected_source_manifest_sha256
        or payload["worker_candidates"] != list(FROZEN_CONFIG.worker_candidates)
    ):
        raise ValueError("benchmark evidence provenance or schema mismatch")
    roots = payload["trial_roots"]
    if (
        not isinstance(roots, list)
        or len(roots) != len(FROZEN_CONFIG.worker_candidates)
        or any(not isinstance(root, str) or not root for root in roots)
        or len({str(Path(root).resolve(strict=False)) for root in roots})
        != len(roots)
    ):
        raise ValueError("benchmark evidence trial roots are invalid")
    resolved_roots = tuple(
        Path(root).resolve(strict=False) for root in roots
    )
    if any(
        root == TERMINATED_DIAGNOSTIC_ROOT
        or TERMINATED_DIAGNOSTIC_ROOT in root.parents
        for root in resolved_roots
    ):
        raise ValueError(
            "benchmark evidence uses the terminated diagnostic root"
        )
    benchmark = benchmark_from_payload(payload["benchmark"])
    measurements = runtime_measurements_from_payload(
        payload["runtime_measurements"]
    )
    validate_benchmark_source_manifest_binding(
        benchmark, measurements, payload["source_manifest_sha256"]
    )
    expected_proof = select_balanced_benchmark_datasets(
        tuple(expected_costs)
    )
    if benchmark.selection_proof != expected_proof:
        raise ValueError(
            "benchmark selection proof differs from frozen TRAIN-only balance"
        )
    expected_datasets = expected_proof.selected_datasets
    if (
        benchmark.benchmark_datasets != expected_datasets
        or any(
            trial.completed_datasets != expected_datasets
            for trial in benchmark.trials
        )
    ):
        raise ValueError(
            "benchmark datasets differ from the frozen TRAIN-only balance"
        )
    if (
        benchmark.status != "qualified_parallel_runtime"
        or benchmark.selected_worker_count not in FROZEN_CONFIG.worker_candidates[1:]
    ):
        raise ValueError("benchmark does not authorize a parallel runtime")
    return benchmark


@dataclass(frozen=True)
class OpenedRun:
    plan: Any
    roots: ParallelRoots
    frozen: FrozenBenchmarkInputs


@dataclass(frozen=True)
class OfficialStages:
    clean_git: Callable[[], str]
    label_guard: Callable[[], str]
    freeze_inputs: Callable[[str, str], Any]
    verify_benchmark: Callable[[Any, str, str], RuntimeBenchmark]
    open_roots: Callable[[Any, RuntimeBenchmark], Any]
    run_resume: Callable[[Any, RuntimeBenchmark, Any], Any]
    revalidate: Callable[[], Any]
    render: Callable[[Any, RuntimeBenchmark, Any], Mapping[str, bytes]]
    verify: Callable[[Mapping[str, bytes], Any], Any]
    publish: Callable[[Mapping[str, bytes], Any], Any]


def execute_official(stages: OfficialStages) -> int:
    if not isinstance(stages, OfficialStages):
        raise ValueError("official stages must be typed")
    head = stages.clean_git()
    if not isinstance(head, str) or len(head) != 40:
        raise ValueError("clean Git stage must return the verified HEAD")
    source_manifest_sha256 = stages.label_guard()
    frozen = stages.freeze_inputs(head, source_manifest_sha256)
    benchmark = stages.verify_benchmark(
        frozen, head, source_manifest_sha256
    )
    opened = stages.open_roots(frozen, benchmark)
    run = stages.run_resume(frozen, benchmark, opened)
    stages.revalidate()
    payloads = stages.render(run, benchmark, opened)
    stages.verify(payloads, opened)
    stages.publish(payloads, opened)
    return 0


def make_test_stages(*, clean_git: Callable[[], str]) -> OfficialStages:
    unreachable = lambda *_args: (_ for _ in ()).throw(
        AssertionError("later stage reached")
    )
    return OfficialStages(
        clean_git,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
        unreachable,
    )


def execute_for_test(stages: OfficialStages) -> int:
    return execute_official(stages)


def _build_stages(args: argparse.Namespace) -> OfficialStages:
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    state: dict[str, Any] = {}

    def freeze(
        head: str, source_manifest_sha256: str
    ) -> FrozenBenchmarkInputs:
        frozen = freeze_benchmark_inputs(
            args, head, source_manifest_sha256
        )
        state["head"] = head
        state["source_manifest_sha256"] = source_manifest_sha256
        state["context_sha256"] = _context_sha256(frozen.context)
        state["costs"] = frozen.costs
        return frozen

    def verify_benchmark(
        frozen: FrozenBenchmarkInputs,
        head: str,
        source_manifest_sha256: str,
    ) -> RuntimeBenchmark:
        benchmark = verify_benchmark_evidence(
            args.benchmark_evidence,
            expected_git_commit=head,
            expected_context_sha256=_context_sha256(frozen.context),
            expected_source_manifest_sha256=source_manifest_sha256,
            expected_costs=frozen.costs,
        )
        state["benchmark_sha256"] = benchmark.sha256
        return benchmark

    def open_roots(
        frozen: FrozenBenchmarkInputs,
        benchmark: RuntimeBenchmark,
    ) -> OpenedRun:
        require_disjoint_roots(args.checkpoint_dir, args.output_dir)
        worker_count = benchmark.selected_worker_count
        if type(worker_count) is not int:
            raise ValueError("verified benchmark did not select workers")
        plan = build_shard_plan(frozen.costs, worker_count)
        roots = ParallelRoots(
            args.checkpoint_dir,
            {
                index: args.checkpoint_dir / f"worker-{index}"
                for index in range(worker_count)
            },
            args.checkpoint_dir / "canonical",
        )
        return OpenedRun(plan, roots, frozen)

    def run_resume(
        frozen: FrozenBenchmarkInputs,
        benchmark: RuntimeBenchmark,
        opened: OpenedRun,
    ):
        if benchmark.selected_worker_count != opened.plan.worker_count:
            raise ValueError("worker count differs from verified benchmark")
        run = resume_or_run_parallel(
            frozen.context,
            opened.plan,
            opened.roots,
            frozen.bootstrap,
        )
        if run.status != "completed" or not run.publication_allowed:
            raise ValueError("Exp419 parallel run did not pass publication gates")
        return run

    def revalidate() -> None:
        if require_clean_git() != state["head"]:
            raise ValueError("Exp419 verified Git HEAD changed before publication")
        if assert_exp419_label_isolation() != state[
            "source_manifest_sha256"
        ]:
            raise ValueError(
                "Exp419 source manifest changed before publication"
            )
        verify_input_snapshots(*paths)
        observed = verify_benchmark_evidence(
            args.benchmark_evidence,
            expected_git_commit=state["head"],
            expected_context_sha256=state["context_sha256"],
            expected_source_manifest_sha256=
                state["source_manifest_sha256"],
            expected_costs=state["costs"],
        )
        if observed.sha256 != state["benchmark_sha256"]:
            raise ValueError("Exp419 benchmark evidence changed before publication")
        require_disjoint_roots(args.checkpoint_dir, args.output_dir)

    def render(run, benchmark: RuntimeBenchmark, opened: OpenedRun):
        trusted = TrustedCheckpointProvenance.from_store(state["checkpoints"])
        return render_exp419_payloads(
            run, benchmark, opened.plan, trusted
        )

    def verify(payloads: Mapping[str, bytes], opened: OpenedRun) -> None:
        verify_exp419_payloads(
            payloads,
            trusted_checkpoint=TrustedCheckpointProvenance.from_store(
                state["checkpoints"]
            ),
        )

    def publish(payloads: Mapping[str, bytes], opened: OpenedRun) -> None:
        publish_exp419(
            payloads,
            args.output_dir,
            checkpoints=state["checkpoints"],
            run=state["run"],
            benchmark=state["benchmark"],
            plan=opened.plan,
            pre_publish_verify=revalidate,
        )

    def record_run(frozen, benchmark, opened):
        run = run_resume(frozen, benchmark, opened)
        state["checkpoints"] = Exp417CheckpointStore(
            opened.roots.canonical_root,
            frozen.bootstrap.provenance,
            frozen.context.datasets,
        )
        state["run"] = run
        state["benchmark"] = benchmark
        return run

    return OfficialStages(
        require_clean_git,
        assert_exp419_label_isolation,
        freeze,
        verify_benchmark,
        open_roots,
        record_run,
        revalidate,
        render,
        verify,
        publish,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return execute_official(_build_stages(args))


if __name__ == "__main__":
    raise SystemExit(main())
