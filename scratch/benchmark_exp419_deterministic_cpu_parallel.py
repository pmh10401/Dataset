#!/usr/bin/env python3
"""Local-only frozen-subset runtime benchmark for Exp419."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
import re
import resource
import subprocess
import sys
from time import monotonic, process_time
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local-only Exp419 label-free benchmark for exactly "
            "1, 2, 4, and 6 CPU workers."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--exp291-prediction", type=Path, required=True)
    parser.add_argument("--exp291-manifest", type=Path, required=True)
    parser.add_argument("--exp292-prediction", type=Path, required=True)
    parser.add_argument("--exp292-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
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


def _static_string(node: ast.AST, constants: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left, constants)
        right = _static_string(node.right, constants)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            target = value.value if isinstance(value, ast.FormattedValue) else value
            part = _static_string(target, constants)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _assignment_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _contains_projected_label(query: str) -> bool:
    without_comments = re.sub(
        r"/\*.*?\*/|--[^\r\n]*", " ", query, flags=re.DOTALL
    )
    normalized = " ".join(without_comments.split())
    normalized = normalized.translate(
        str.maketrans({"\"": "", "`": "", "[": "", "]": ""})
    )
    for match in re.finditer(
        r"\bselect\b(.*?)\bfrom\b",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        if re.search(
            r"\b(?:[a-z_][a-z0-9_]*\.)?(?:label|labels_blob)\b",
            match.group(1),
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _looks_like_label_artifact(value: str) -> bool:
    lowered = value.casefold()
    artifact_shape = any(character in lowered for character in ("/", "\\", ".", "_", "-"))
    return artifact_shape and bool(
        re.search(
            r"(?:^|[/\\._-])(?:frozen[/\\._-]*)?(?:test[/\\._-]*)?labels?(?:$|[/\\._-])",
            lowered,
        )
        or "labels_blob" in lowered
    )


_APPROVED_DYNAMIC_PATH_ROLES = {
    ("coordinator.py", "_RunRootTransaction.__init__"): (
        "root",
        "self.path",
    ),
    ("coordinator.py", "_RunRootTransaction.verify"): ("self.path",),
    ("coordinator.py", "_RunRootTransaction.verify_checkpoint_path"): ("path",),
    ("coordinator.py", "_read_optional_root_file"): ("name",),
    ("coordinator.py", "_publish_binding_no_replace"): ("temporary",),
    ("coordinator.py", "_nonempty_checkpoint_root"): ("path",),
    ("coordinator.py", "_replace_runtime_evidence"): ("temporary",),
    ("coordinator.py", "_running_attempt_record"): (
        "spec.checkpoint_root",
        "roots.root",
    ),
    ("coordinator.py", "_interrupted_attempt_record"): (
        "worker['checkpoint_root']",
    ),
    ("coordinator.py", "_freeze_plan_binding"): (
        "roots.root",
        "roots.canonical_root",
        "roots.worker_roots[index]",
        "path",
    ),
    ("coordinator.py", "_validated_evidence_digests"): (
        "roots.root",
        "worker['checkpoint_root']",
        "worker['resolved_checkpoint_root']",
    ),
    ("coordinator.py", "_validate_and_recover_attempts"): (
        "worker['checkpoint_root']",
    ),
    ("publication.py", "publish_exp419"): ("output_dir",),
    ("reporting.py", "_source_digest"): (
        "__file__",
        "root",
        "relative_path",
    ),
    ("reporting.py", "_exp419_source_digests"): (
        "__file__",
        "root",
        "relative_path",
    ),
    ("workers.py", "_casefolded_path_parts"): ("path",),
    ("workers.py", "_filesystem_identity"): ("path",),
    ("workers.py", "_validated_specs"): ("spec.checkpoint_root",),
    ("workers.py", "_bootstrap_from_payload"): (
        "resources_payload['database']",
        "resources_payload['exp291_prediction']",
        "resources_payload['exp291_manifest']",
        "resources_payload['exp292_prediction']",
        "resources_payload['exp292_manifest']",
        "resources_payload['backend_audit_path']",
    ),
    ("benchmark_exp419_deterministic_cpu_parallel.py", "<module>"): (
        "__file__",
    ),
    (
        "benchmark_exp419_deterministic_cpu_parallel.py",
        "prepare_trial_roots",
    ): ("root",),
    (
        "benchmark_exp419_deterministic_cpu_parallel.py",
        "_run_trials",
    ): ("root",),
    (
        "benchmark_exp419_deterministic_cpu_parallel.py",
        "write_benchmark_evidence",
    ): ("path", "root"),
    (
        "benchmark_exp419_deterministic_cpu_parallel.py",
        "assert_exp419_label_isolation",
    ): ("path",),
    (
        "benchmark_exp419_deterministic_cpu_parallel.py",
        "_approved_dynamic_path",
    ): ("source_name",),
    ("run_exp419_deterministic_cpu_parallel.py", "<module>"): ("__file__",),
    (
        "run_exp419_deterministic_cpu_parallel.py",
        "require_disjoint_roots",
    ): ("checkpoint_dir", "output_dir", "output_source"),
    (
        "run_exp419_deterministic_cpu_parallel.py",
        "_identity",
    ): ("path",),
    (
        "run_exp419_deterministic_cpu_parallel.py",
        "_read_json_regular",
    ): ("path",),
    (
        "run_exp419_deterministic_cpu_parallel.py",
        "verify_benchmark_evidence",
    ): ("path", "root"),
}


def _node_scope(node: ast.AST, parents: Mapping[ast.AST, ast.AST]) -> str:
    functions = []
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(current.name)
        elif isinstance(current, ast.ClassDef):
            functions.append(current.name)
            break
    return ".".join(reversed(functions)) if functions else "<module>"


def _filesystem_target(call: ast.Call) -> ast.AST | None:
    called = _call_name(call.func).casefold()
    last = called.split(".")[-1]
    if called in {"open", "os.open", "os.stat", "os.lstat"}:
        return call.args[0] if call.args else None
    if last == "path":
        return call.args[0] if call.args else None
    if last in {
        "stat",
        "lstat",
        "exists",
        "is_file",
        "read_bytes",
        "read_text",
        "resolve",
        "open",
    } and isinstance(call.func, ast.Attribute):
        return call.func.value
    return None


def _role_occurs(expression: str, role: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(role)}(?![A-Za-z0-9_])",
            expression,
        )
    )


def _approved_dynamic_path(
    source_name: str,
    scope: str,
    target: ast.AST,
    constants: Mapping[str, str],
) -> bool:
    static = _static_string(target, constants)
    if static is not None:
        return not _looks_like_label_artifact(static)
    if (
        isinstance(target, ast.Call)
        and _call_name(target.func).casefold().split(".")[-1] == "path"
        and target.args
    ):
        return _approved_dynamic_path(
            source_name, scope, target.args[0], constants
        )
    expression = ast.unparse(target).replace('"', "'")
    roles = _APPROVED_DYNAMIC_PATH_ROLES.get(
        (Path(source_name).name, scope),
        (),
    )
    return any(_role_occurs(expression, role) for role in roles)


def assert_source_label_isolation(sources: Mapping[str, str]) -> None:
    """Reject statically reachable label loaders, projections, and artifacts."""
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("label isolation requires source mappings")
    for name, source in sources.items():
        if not isinstance(name, str) or not isinstance(source, str):
            raise ValueError("label isolation sources must be strings")
        try:
            tree = ast.parse(source, filename=name)
        except SyntaxError as error:
            raise ValueError(f"label isolation cannot parse {name}") from error

        constants: dict[str, str] = {}
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        denylist_nodes: set[ast.AST] = set()
        guard_definition_nodes: set[ast.AST] = set()
        for candidate in ast.walk(tree):
            if (
                isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
                and candidate.name
                in {
                    "_contains_projected_label",
                    "_looks_like_label_artifact",
                    "assert_source_label_isolation",
                }
            ):
                guard_definition_nodes.update(ast.walk(candidate))
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        ]
        for _ in range(len(assignments) + 1):
            changed = False
            for assignment in assignments:
                target = _assignment_name(assignment)
                value_node = _assignment_value(assignment)
                if target is None or value_node is None:
                    continue
                value = _static_string(value_node, constants)
                if value is not None and constants.get(target) != value:
                    constants[target] = value
                    changed = True
                if any(token in target.casefold() for token in ("forbidden", "denylist")):
                    denylist_nodes.update(ast.walk(value_node))
            if not changed:
                break

        dynamic_getattr_aliases: set[str] = set()
        filesystem_aliases: set[str] = set()
        for _ in range(len(assignments) + 1):
            changed = False
            for assignment in assignments:
                target = _assignment_name(assignment)
                value_node = _assignment_value(assignment)
                if target is None or value_node is None:
                    continue
                dynamic = False
                if (
                    isinstance(value_node, ast.Call)
                    and _call_name(value_node.func).casefold() == "getattr"
                    and len(value_node.args) >= 2
                    and _static_string(value_node.args[1], constants) is None
                ):
                    dynamic = True
                elif (
                    isinstance(value_node, ast.Name)
                    and value_node.id in dynamic_getattr_aliases
                ):
                    dynamic = True
                if dynamic and target not in dynamic_getattr_aliases:
                    dynamic_getattr_aliases.add(target)
                    changed = True
                source_call = _call_name(value_node).casefold()
                filesystem_alias = (
                    source_call
                    in {
                        "open",
                        "path",
                        "os.open",
                        "os.stat",
                        "os.lstat",
                    }
                    or (
                        isinstance(value_node, ast.Name)
                        and value_node.id in filesystem_aliases
                    )
                )
                if filesystem_alias and target not in filesystem_aliases:
                    filesystem_aliases.add(target)
                    changed = True
            if not changed:
                break

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = tuple(alias.name for alias in node.names)
                module = node.module or ""
                for imported_name in imported:
                    qualified = f"{module}.{imported_name}".casefold()
                    if (
                        imported_name != "assert_label_free_sql_contract"
                        and (
                            re.search(r"(?:load|read|fetch|get).*labels?", qualified)
                            or re.search(r"(?:^|\.)labels?(?:\.|$)", module.casefold())
                        )
                    ):
                        raise ValueError(f"label loader import in {name}")
            elif isinstance(node, ast.Import):
                if any(
                    re.search(r"(?:^|\.)labels?(?:\.|$)", alias.name.casefold())
                    for alias in node.names
                ):
                    raise ValueError(f"label module import in {name}")
            elif isinstance(node, ast.Attribute):
                attribute_name = _call_name(node).casefold()
                if re.search(
                    r"(?:load|read|fetch|get).*labels?", attribute_name
                ):
                    raise ValueError(f"label loader attribute in {name}")

            if isinstance(node, ast.Call):
                called = _call_name(node.func).casefold()
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in dynamic_getattr_aliases
                ) or (
                    isinstance(node.func, ast.Attribute)
                    and _call_name(node.func).split(".", 1)[0]
                    in dynamic_getattr_aliases
                ) or (
                    isinstance(node.func, ast.Call)
                    and _call_name(node.func.func).casefold() == "getattr"
                    and len(node.func.args) >= 2
                    and _static_string(node.func.args[1], constants) is None
                ):
                    raise ValueError(f"dynamic label loader callable in {name}")
                if called in {"__import__", "importlib.import_module"}:
                    target = (
                        _static_string(node.args[0], constants)
                        if node.args
                        else None
                    )
                    if target is None:
                        raise ValueError(f"dynamic label import target in {name}")
                    if re.search(
                        r"(?:^|\.)labels?(?:\.|$)", target.casefold()
                    ):
                        raise ValueError(f"label import target in {name}")
                if (
                    "assert_label_free_sql_contract" not in called
                    and re.search(r"(?:load|read|fetch|get).*labels?", called)
                ):
                    raise ValueError(f"label loader call in {name}")
                if called == "getattr" and len(node.args) >= 2:
                    attribute = _static_string(node.args[1], constants)
                    if attribute and re.search(
                        r"(?:load|read|fetch|get).*labels?", attribute.casefold()
                    ):
                        raise ValueError(f"aliased label loader in {name}")
                filesystem_target = (
                    node.args[0]
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in filesystem_aliases
                        and node.args
                    )
                    else _filesystem_target(node)
                )
                if filesystem_target is not None and not _approved_dynamic_path(
                    name,
                    _node_scope(node, parents),
                    filesystem_target,
                    constants,
                ):
                    raise ValueError(
                        f"dynamic or label filesystem path in {name}"
                    )

            value = _static_string(node, constants)
            if value is None:
                continue
            if (
                node in guard_definition_nodes
                or value
                in {
                    "assert_label_free_sql_contract",
                    "assert_exp419_label_isolation",
                }
            ):
                continue
            if _contains_projected_label(value):
                raise ValueError(f"label SQL projection in {name}")
            if node not in denylist_nodes and _looks_like_label_artifact(value):
                raise ValueError(f"frozen label artifact in {name}")


def assert_exp419_label_isolation() -> None:
    assert_label_free_sql_contract()
    package = REPO_ROOT / "experiments/exp419_deterministic_cpu_parallel"
    paths = tuple(sorted(package.glob("*.py"))) + (
        REPO_ROOT / "scratch/benchmark_exp419_deterministic_cpu_parallel.py",
        REPO_ROOT / "scratch/run_exp419_deterministic_cpu_parallel.py",
    )
    assert_source_label_isolation(
        {str(path): path.read_text(encoding="utf-8") for path in paths}
    )


@dataclass(frozen=True)
class FrozenBenchmarkInputs:
    context: Any
    resources: RunnerResources
    costs: tuple[DatasetCost, ...]
    bootstrap: WorkerBootstrap
    paths: tuple[Path, ...]


def _sha256_text(value: object, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class WorkerUtilization:
    shard_index: int
    pid: int
    elapsed_seconds: float
    wall_occupancy_fraction: float

    def __post_init__(self) -> None:
        if type(self.shard_index) is not int or self.shard_index < 0:
            raise ValueError("worker utilization shard index is invalid")
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("worker utilization PID is invalid")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, Real)
            or not isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("worker elapsed time must be finite and nonnegative")
        if (
            isinstance(self.wall_occupancy_fraction, bool)
            or not isinstance(self.wall_occupancy_fraction, Real)
            or not isfinite(self.wall_occupancy_fraction)
            or not 0.0 <= self.wall_occupancy_fraction <= 1.0
        ):
            raise ValueError("worker utilization must be finite and within [0, 1]")


def _worker_utilization_payload(worker: WorkerUtilization) -> dict[str, object]:
    return {
        "elapsed_seconds": worker.elapsed_seconds,
        "pid": worker.pid,
        "shard_index": worker.shard_index,
        "wall_occupancy_fraction": worker.wall_occupancy_fraction,
    }


@dataclass(frozen=True)
class TrialRuntimeMeasurement:
    worker_count: int
    wall_seconds: float
    process_cpu_seconds: float
    process_cpu_utilization: float
    worker_utilization: tuple[WorkerUtilization, ...]
    scientific_evidence_sha256: str
    telemetry_sha256: str = ""
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.worker_count) is not int
            or self.worker_count not in FROZEN_CONFIG.worker_candidates
        ):
            raise ValueError("runtime measurement worker count is not frozen")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, Real)
            or not isfinite(self.wall_seconds)
            or self.wall_seconds <= 0.0
        ):
            raise ValueError("runtime wall time must be finite and positive")
        if (
            isinstance(self.process_cpu_seconds, bool)
            or not isinstance(self.process_cpu_seconds, Real)
            or not isfinite(self.process_cpu_seconds)
            or self.process_cpu_seconds < 0.0
        ):
            raise ValueError("process CPU time must be finite and nonnegative")
        expected_cpu_utilization = self.process_cpu_seconds / (
            self.wall_seconds * (self.worker_count + 1)
        )
        if (
            isinstance(self.process_cpu_utilization, bool)
            or not isinstance(self.process_cpu_utilization, Real)
            or not isfinite(self.process_cpu_utilization)
            or not 0.0 <= self.process_cpu_utilization <= 1.0
            or not isclose(
                self.process_cpu_utilization,
                expected_cpu_utilization,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("process CPU utilization is semantically inconsistent")
        workers = tuple(self.worker_utilization)
        if (
            len(workers) != self.worker_count
            or any(not isinstance(worker, WorkerUtilization) for worker in workers)
            or tuple(worker.shard_index for worker in workers)
            != tuple(range(self.worker_count))
        ):
            raise ValueError("worker utilization must cover exact frozen shards")
        for worker in workers:
            expected_occupancy = worker.elapsed_seconds / self.wall_seconds
            if (
                worker.elapsed_seconds > self.wall_seconds
                or not isclose(
                    worker.wall_occupancy_fraction,
                    expected_occupancy,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("worker utilization is semantically inconsistent")
        scientific = _sha256_text(
            self.scientific_evidence_sha256,
            "scientific evidence",
        )
        telemetry_payload = {
            "process_cpu_seconds": self.process_cpu_seconds,
            "process_cpu_utilization": self.process_cpu_utilization,
            "wall_seconds": self.wall_seconds,
            "worker_count": self.worker_count,
            "worker_utilization": [
                _worker_utilization_payload(worker) for worker in workers
            ],
        }
        telemetry = sha256(canonical_json_bytes(telemetry_payload)).hexdigest()
        evidence = sha256(
            canonical_json_bytes(
                {
                    "scientific_evidence_sha256": scientific,
                    "telemetry_sha256": telemetry,
                }
            )
        ).hexdigest()
        if self.telemetry_sha256 and self.telemetry_sha256 != telemetry:
            raise ValueError("runtime telemetry digest mismatch")
        if self.evidence_sha256 and self.evidence_sha256 != evidence:
            raise ValueError("runtime evidence digest mismatch")
        object.__setattr__(self, "worker_utilization", workers)
        object.__setattr__(self, "telemetry_sha256", telemetry)
        object.__setattr__(self, "evidence_sha256", evidence)


@dataclass(frozen=True)
class MeasuredBenchmark:
    benchmark: RuntimeBenchmark
    runtime_measurements: tuple[TrialRuntimeMeasurement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark, RuntimeBenchmark):
            raise ValueError("measured benchmark requires RuntimeBenchmark")
        measurements = tuple(self.runtime_measurements)
        if (
            tuple(item.worker_count for item in measurements)
            != FROZEN_CONFIG.worker_candidates
        ):
            raise ValueError("measurements must cover exact frozen worker candidates")
        for trial, measurement in zip(
            self.benchmark.trials, measurements, strict=True
        ):
            if (
                trial.worker_count != measurement.worker_count
                or not isclose(
                    trial.elapsed_seconds,
                    measurement.wall_seconds,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or trial.evidence_sha256 != measurement.evidence_sha256
            ):
                raise ValueError("runtime measurement differs from benchmark trial")
        object.__setattr__(self, "runtime_measurements", measurements)


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


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _run_trials(
    frozen: FrozenBenchmarkInputs,
    trial_roots: Mapping[int, Path],
) -> MeasuredBenchmark:
    selected = select_benchmark_datasets(frozen.costs)
    if not selected:
        raise ValueError("frozen benchmark subset is empty")
    selected_set = set(selected)
    subset_costs = tuple(cost for cost in frozen.costs if cost.dataset in selected_set)
    if tuple(sorted(cost.dataset for cost in subset_costs)) != selected:
        raise ValueError("benchmark subset differs from frozen TRAIN-only selection")

    trials: list[RuntimeTrial] = []
    measurements: list[TrialRuntimeMeasurement] = []
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
        child_cpu_started = _children_cpu_seconds()
        parent_cpu_started = process_time()
        started = monotonic()
        completions = spawn_workers(specs, frozen.bootstrap)
        elapsed = monotonic() - started
        process_cpu_seconds = (
            process_time() - parent_cpu_started
            + _children_cpu_seconds() - child_cpu_started
        )
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
        scientific_evidence = sha256(
            canonical_json_bytes(
                {
                    "completed_datasets": completed,
                    "record_digests": record_digests,
                }
            )
        ).hexdigest()
        worker_utilization = tuple(
            WorkerUtilization(
                completion.shard_index,
                completion.pid,
                completion.elapsed_seconds,
                completion.elapsed_seconds / elapsed,
            )
            for completion in sorted(
                completions, key=lambda item: item.shard_index
            )
        )
        measurement = TrialRuntimeMeasurement(
            worker_count,
            elapsed,
            process_cpu_seconds,
            process_cpu_seconds / (elapsed * (worker_count + 1)),
            worker_utilization,
            scientific_evidence,
        )
        measurements.append(measurement)
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
                measurement.evidence_sha256,
            )
        )
    return MeasuredBenchmark(
        select_runtime(tuple(trials)),
        tuple(measurements),
    )


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


def runtime_measurement_payload(
    measurement: TrialRuntimeMeasurement,
) -> dict[str, object]:
    return {
        "evidence_sha256": measurement.evidence_sha256,
        "process_cpu_seconds": measurement.process_cpu_seconds,
        "process_cpu_utilization": measurement.process_cpu_utilization,
        "scientific_evidence_sha256":
            measurement.scientific_evidence_sha256,
        "telemetry_sha256": measurement.telemetry_sha256,
        "wall_seconds": measurement.wall_seconds,
        "worker_count": measurement.worker_count,
        "worker_utilization": [
            _worker_utilization_payload(worker)
            for worker in measurement.worker_utilization
        ],
    }


def runtime_measurements_from_payload(
    payload: object,
) -> tuple[TrialRuntimeMeasurement, ...]:
    if not isinstance(payload, list):
        raise ValueError("runtime measurements must be a list")
    exact_fields = {
        "evidence_sha256",
        "process_cpu_seconds",
        "process_cpu_utilization",
        "scientific_evidence_sha256",
        "telemetry_sha256",
        "wall_seconds",
        "worker_count",
        "worker_utilization",
    }
    worker_fields = {
        "elapsed_seconds",
        "pid",
        "shard_index",
        "wall_occupancy_fraction",
    }
    measurements = []
    for raw in payload:
        if not isinstance(raw, dict) or set(raw) != exact_fields:
            raise ValueError("runtime measurement fields are invalid")
        raw_workers = raw["worker_utilization"]
        if not isinstance(raw_workers, list):
            raise ValueError("worker utilization must be a list")
        workers = []
        for raw_worker in raw_workers:
            if (
                not isinstance(raw_worker, dict)
                or set(raw_worker) != worker_fields
            ):
                raise ValueError("worker utilization fields are invalid")
            workers.append(
                WorkerUtilization(
                    raw_worker["shard_index"],
                    raw_worker["pid"],
                    raw_worker["elapsed_seconds"],
                    raw_worker["wall_occupancy_fraction"],
                )
            )
        measurements.append(
            TrialRuntimeMeasurement(
                raw["worker_count"],
                raw["wall_seconds"],
                raw["process_cpu_seconds"],
                raw["process_cpu_utilization"],
                tuple(workers),
                raw["scientific_evidence_sha256"],
                raw["telemetry_sha256"],
                raw["evidence_sha256"],
            )
        )
    return tuple(measurements)


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
    runtime_measurements: Sequence[TrialRuntimeMeasurement],
) -> None:
    measured = MeasuredBenchmark(
        benchmark,
        tuple(runtime_measurements),
    )
    roots = tuple(Path(root).resolve(strict=False) for root in trial_roots)
    if len(roots) != len(FROZEN_CONFIG.worker_candidates) or len(set(roots)) != len(roots):
        raise ValueError("benchmark evidence requires exact disjoint trial roots")
    payload = {
        "artifact_kind": "exp419_non_official_runtime_benchmark",
        "benchmark": benchmark_payload(benchmark),
        "context_sha256": context_sha256,
        "git_commit": git_commit,
        "runtime_measurements": [
            runtime_measurement_payload(measurement)
            for measurement in measured.runtime_measurements
        ],
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
    run_trials: Callable[
        [FrozenBenchmarkInputs, Mapping[int, Path]], MeasuredBenchmark
    ]
    revalidate: Callable[[], Any]
    write: Callable[
        [MeasuredBenchmark, str, FrozenBenchmarkInputs, Mapping[int, Path]],
        Any,
    ]


def execute_benchmark(stages: BenchmarkStages) -> int:
    head = stages.clean_git()
    stages.label_guard()
    frozen = stages.freeze_inputs(head)
    roots = stages.prepare_roots()
    measured = stages.run_trials(frozen, roots)
    stages.revalidate()
    stages.write(measured, head, frozen, roots)
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
        measured: MeasuredBenchmark,
        head: str,
        frozen: FrozenBenchmarkInputs,
        roots: Mapping[int, Path],
    ) -> None:
        write_benchmark_evidence(
            args.benchmark_root / BENCHMARK_FILE,
            benchmark=measured.benchmark,
            git_commit=head,
            context_sha256=_context_sha256(frozen.context),
            trial_roots=tuple(roots[count] for count in FROZEN_CONFIG.worker_candidates),
            runtime_measurements=measured.runtime_measurements,
        )

    return BenchmarkStages(
        require_clean_git,
        assert_exp419_label_isolation,
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
