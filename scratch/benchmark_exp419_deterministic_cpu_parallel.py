#!/usr/bin/env python3
"""Local-only frozen-subset runtime benchmark for Exp419."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isclose, isfinite
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import re
import resource
import stat
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
    list_dataset_names,
    _runtime_versions,
    verify_input_snapshots,
)
from experiments.exp417_causal_multihorizon_shadow.runner import RunnerResources
from experiments.exp419_deterministic_cpu_parallel.config import (
    FROZEN_CONFIG,
    require_exact_benchmark_evidence_path,
    require_exact_benchmark_root,
    require_exact_trial_roots,
    seed_workers_for_outer_workers,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    DatasetCost,
    BenchmarkTrainBootstrap,
    BenchmarkTrainContext,
    RuntimeBenchmark,
    RuntimeTrial,
    WorkerSpec,
    benchmark_selection_proof_from_payload,
    benchmark_selection_proof_payload,
)
from experiments.exp419_deterministic_cpu_parallel.coordinator import (
    _context_sha256,
    select_runtime,
)
from experiments.exp419_deterministic_cpu_parallel.planner import (
    build_shard_plan,
    estimate_dataset_cost,
    select_balanced_benchmark_datasets,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    TrainLengthGroupMeta,
)
from experiments.exp419_deterministic_cpu_parallel.workers import (
    spawn_bounded_train_smoke_workers,
)
from scratch.run_exp417_causal_multihorizon_shadow import (
    _backend_audit_path,
)


BENCHMARK_FILE = FROZEN_CONFIG.benchmark_evidence_filename


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


def _static_constants(tree: ast.AST) -> dict[str, str]:
    constants: dict[str, str] = {}
    assignments = [
        node for node in ast.walk(tree)
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
        if not changed:
            break
    return constants


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
            _DIRECT_LABEL_PROJECTION_DENYLIST[0],
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
            _DIRECT_LABEL_DENYLIST[2],
            lowered,
        )
        or _DIRECT_LABEL_DENYLIST[3] in lowered
    )


SOURCE_MANIFEST_RELATIVE = (
    "experiments/exp419_deterministic_cpu_parallel/SOURCE_MANIFEST.json"
)
_SOURCE_MANIFEST_KIND = "exp419_reviewed_execution_source_manifest"
_DIRECT_LABEL_DENYLIST = (
    r"(?:load|read|fetch|get).*labels?",
    r"(?:^|\.)labels?(?:\.|$)",
    r"(?:^|[/\\._-])(?:frozen[/\\._-]*)?(?:test[/\\._-]*)?"
    r"labels?(?:$|[/\\.]|-(?:bin|npy|npz|json|csv|txt)$)",
    "labels_blob",
)
_DIRECT_LABEL_PROJECTION_DENYLIST = (
    r"\b(?:[a-z_][a-z0-9_]*\.)?(?:label|labels_blob)\b",
)
_GUARD_LITERAL_ASSIGNMENTS = (
    "_DIRECT_LABEL_DENYLIST",
    "_DIRECT_LABEL_PROJECTION_DENYLIST",
    "_GUARD_LITERAL_ASSIGNMENTS",
    "_FORBIDDEN_KEYS",
)
_REPORTING_FORBIDDEN_KEYS_SHA256 = (
    "a20c61eb989ed44697812562e4381ed6c4f09b7d9c565512d1f6bcdd92995fec"
)


def _read_regular_no_follow(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"{description} must be a non-symlink regular file"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{description} must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _expected_runtime_source_paths(repo_root: Path) -> tuple[str, ...]:
    root = Path(repo_root)
    package = root / "experiments/exp419_deterministic_cpu_parallel"
    try:
        package_names = sorted(
            entry.name
            for entry in os.scandir(package)
            if entry.name.endswith(".py")
        )
    except OSError as error:
        raise ValueError("Exp419 runtime source directory is unavailable") from error
    relative = [
        f"experiments/exp419_deterministic_cpu_parallel/{name}"
        for name in package_names
    ]
    relative.extend(
        (
            "scratch/benchmark_exp419_deterministic_cpu_parallel.py",
            "scratch/run_exp419_deterministic_cpu_parallel.py",
        )
    )
    return tuple(sorted(relative))


def _validated_relative_source_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source manifest path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or tuple(path.parts) != tuple(part for part in path.parts if part not in {"", ".", ".."})
        or value != path.as_posix()
    ):
        raise ValueError("source manifest path traversal is forbidden")
    return value


def _literal_container_schema(
    node: ast.AST,
) -> tuple[str, tuple[str, ...]]:
    if isinstance(node, ast.Tuple):
        kind, children = "tuple", node.elts
    elif isinstance(node, ast.List):
        kind, children = "list", node.elts
    elif isinstance(node, ast.Set):
        kind, children = "set", node.elts
    elif isinstance(node, ast.Dict):
        kind = "dict"
        children = [
            child
            for pair in zip(node.keys, node.values, strict=True)
            for child in pair
            if child is not None
        ]
    else:
        raise ValueError("guard denylist value must be a literal container")
    leaves: list[str] = []
    for child in children:
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            leaves.append(child.value)
            continue
        if isinstance(child, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            _nested_kind, nested = _literal_container_schema(child)
            leaves.extend(nested)
            continue
        raise ValueError(
            "guard denylist container must contain static string literals"
        )
    return kind, tuple(leaves)


def _guard_literal_nodes(name: str, tree: ast.Module) -> set[ast.AST]:
    benchmark_name = "scratch/benchmark_exp419_deterministic_cpu_parallel.py"
    reporting_name = "experiments/exp419_deterministic_cpu_parallel/reporting.py"
    approved: dict[str, tuple[str, tuple[str, ...] | str]] = {}
    if name == benchmark_name:
        approved = {
            "_DIRECT_LABEL_DENYLIST": ("tuple", _DIRECT_LABEL_DENYLIST),
            "_DIRECT_LABEL_PROJECTION_DENYLIST": (
                "tuple", _DIRECT_LABEL_PROJECTION_DENYLIST
            ),
            "_GUARD_LITERAL_ASSIGNMENTS": (
                "tuple", _GUARD_LITERAL_ASSIGNMENTS
            ),
        }
    elif name == reporting_name:
        approved = {
            "_FORBIDDEN_KEYS": (
                "set", _REPORTING_FORBIDDEN_KEYS_SHA256
            )
        }
    nodes: set[ast.AST] = set()
    seen: set[str] = set()
    for assignment in tree.body:
        target = _assignment_name(assignment)
        value = _assignment_value(assignment)
        if target not in approved or value is None:
            continue
        if target in seen:
            raise ValueError("guard denylist assignment is duplicated")
        seen.add(target)
        kind, leaves = _literal_container_schema(value)
        expected_kind, expected = approved[target]
        if kind != expected_kind:
            raise ValueError("guard denylist container type is invalid")
        if isinstance(expected, str):
            observed = sha256(
                canonical_json_bytes(sorted(leaves))
            ).hexdigest()
            if observed != expected:
                raise ValueError("guard denylist payload is invalid")
        elif leaves != expected:
            raise ValueError("guard denylist payload is invalid")
        nodes.update(ast.walk(value))
    return nodes


def _is_label_loader_name(value: str) -> bool:
    lowered = value.casefold()
    return bool(
        re.search(_DIRECT_LABEL_DENYLIST[0], lowered)
        or re.search(_DIRECT_LABEL_DENYLIST[1], lowered)
    )


def _direct_label_deny_scan(name: str, source: bytes) -> None:
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text, filename=name)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError(f"label guard cannot parse {name}") from error
    constants = _static_constants(tree)
    ignored = _guard_literal_nodes(name, tree)
    for node in ast.walk(tree):
        if node in ignored:
            continue
        if isinstance(node, ast.Import):
            if any(_is_label_loader_name(alias.name) for alias in node.names):
                raise ValueError(f"label loader import in {name}")
        elif isinstance(node, ast.ImportFrom):
            values = [node.module or "", *(alias.name for alias in node.names)]
            if any(_is_label_loader_name(value) for value in values):
                raise ValueError(f"label loader import in {name}")
        elif isinstance(node, ast.Attribute):
            if _is_label_loader_name(_call_name(node)):
                raise ValueError(f"label loader attribute in {name}")
        elif isinstance(node, ast.Call):
            called = _call_name(node.func)
            if _is_label_loader_name(called):
                raise ValueError(f"label loader call in {name}")
            if called.casefold() == "getattr" and len(node.args) >= 2:
                attribute = _static_string(node.args[1], constants)
                if attribute is not None and _is_label_loader_name(attribute):
                    raise ValueError(f"label loader getattr in {name}")
        value = None if isinstance(node, ast.Name) else _static_string(
            node, constants
        )
        if value is None:
            continue
        if _contains_projected_label(value):
            raise ValueError(f"label SQL projection in {name}")
        if _looks_like_label_artifact(value):
            raise ValueError(f"frozen TEST-label artifact in {name}")


def _manifest_payload(repo_root: Path) -> dict[str, object]:
    sources = []
    for relative in _expected_runtime_source_paths(repo_root):
        data = _read_regular_no_follow(
            Path(repo_root) / relative, f"runtime source {relative}"
        )
        _direct_label_deny_scan(relative, data)
        sources.append(
            {"path": relative, "sha256": sha256(data).hexdigest()}
        )
    return {
        "artifact_kind": _SOURCE_MANIFEST_KIND,
        "schema_version": 1,
        "sources": sources,
    }


def propose_source_manifest(
    repo_root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
) -> bytes:
    """Return deterministic reviewed-source bytes; write only when explicitly asked."""
    canonical = canonical_json_bytes(_manifest_payload(Path(repo_root))) + b"\n"
    if output_path is not None:
        Path(output_path).write_bytes(canonical)
    return canonical


def verify_source_manifest(
    repo_root: Path = REPO_ROOT,
    manifest_path: Path | None = None,
) -> str:
    root = Path(repo_root)
    path = (
        root / SOURCE_MANIFEST_RELATIVE
        if manifest_path is None
        else Path(manifest_path)
    )
    raw = _read_regular_no_follow(path, "source manifest")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source manifest is invalid JSON") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"artifact_kind", "schema_version", "sources"}
        or payload["artifact_kind"] != _SOURCE_MANIFEST_KIND
        or payload["schema_version"] != 1
        or not isinstance(payload["sources"], list)
    ):
        raise ValueError("source manifest schema is invalid")
    if raw != canonical_json_bytes(payload) + b"\n":
        raise ValueError("source manifest must be canonical JSON")
    entries = payload["sources"]
    relative_paths = [
        _validated_relative_source_path(
            entry.get("path") if isinstance(entry, dict) else None
        )
        for entry in entries
    ]
    if (
        any(not isinstance(entry, dict) or set(entry) != {"path", "sha256"}
            for entry in entries)
        or len(relative_paths) != len(set(relative_paths))
    ):
        raise ValueError("source manifest contains duplicate or invalid entries")
    expected = _expected_runtime_source_paths(root)
    if tuple(relative_paths) != expected:
        raise ValueError("source manifest runtime source set is missing or extra")
    for entry, relative in zip(entries, relative_paths, strict=True):
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("source manifest digest is invalid")
        data = _read_regular_no_follow(
            root / relative, f"runtime source {relative}"
        )
        if sha256(data).hexdigest() != digest:
            raise ValueError(f"runtime source digest mismatch: {relative}")
        _direct_label_deny_scan(relative, data)
    return sha256(raw).hexdigest()


def assert_exp419_label_isolation(
    repo_root: Path = REPO_ROOT,
    manifest_path: Path | None = None,
) -> str:
    return verify_source_manifest(repo_root, manifest_path)




@dataclass(frozen=True)
class FrozenBenchmarkInputs:
    context: BenchmarkTrainContext
    resources: RunnerResources
    costs: tuple[DatasetCost, ...]
    bootstrap: BenchmarkTrainBootstrap
    paths: tuple[Path, ...]
    source_manifest_sha256: str


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
    source_manifest_sha256: str
    seed_workers: int = 1
    batches_per_seed: int = 2
    input_proof_sha256: str = "0" * 64
    test_values_loaded: bool = False
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
        source_manifest = _sha256_text(
            self.source_manifest_sha256, "source manifest"
        )
        if type(self.seed_workers) is not int or self.seed_workers < 1:
            raise ValueError("runtime seed worker count is invalid")
        if (
            type(self.batches_per_seed) is not int
            or self.batches_per_seed != FROZEN_CONFIG.bounded_train_batches_per_seed
        ):
            raise ValueError("runtime bounded batch limit is not frozen")
        input_proof = _sha256_text(self.input_proof_sha256, "input proof")
        if self.test_values_loaded is not False:
            raise ValueError("runtime smoke must not load TEST values")
        telemetry_payload = {
            "batches_per_seed": self.batches_per_seed,
            "input_proof_sha256": input_proof,
            "process_cpu_seconds": self.process_cpu_seconds,
            "process_cpu_utilization": self.process_cpu_utilization,
            "seed_workers": self.seed_workers,
            "test_values_loaded": self.test_values_loaded,
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
                    "source_manifest_sha256": source_manifest,
                    "telemetry_sha256": telemetry,
                }
            )
        ).hexdigest()
        if self.telemetry_sha256 and self.telemetry_sha256 != telemetry:
            raise ValueError("runtime telemetry digest mismatch")
        if self.evidence_sha256 and self.evidence_sha256 != evidence:
            raise ValueError("runtime evidence digest mismatch")
        object.__setattr__(self, "worker_utilization", workers)
        object.__setattr__(self, "input_proof_sha256", input_proof)
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


def validate_benchmark_source_manifest_binding(
    benchmark: RuntimeBenchmark,
    runtime_measurements: Sequence[TrialRuntimeMeasurement],
    source_manifest_sha256: str,
) -> MeasuredBenchmark:
    expected = _sha256_text(source_manifest_sha256, "source manifest")
    measured = MeasuredBenchmark(benchmark, tuple(runtime_measurements))
    if any(
        measurement.source_manifest_sha256 != expected
        for measurement in measured.runtime_measurements
    ):
        raise ValueError(
            "runtime measurement source manifest digest mismatch"
        )
    return measured


def costs_from_metadata_rows(
    rows: Sequence[tuple[object, object, object]],
    *,
    expected_datasets: Sequence[str],
) -> tuple[DatasetCost, ...]:
    expected = tuple(expected_datasets)
    if (
        not expected
        or any(not isinstance(dataset, str) or not dataset for dataset in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError(
            "expected frozen population must contain unique non-empty dataset names"
        )
    expected_set = set(expected)
    grouped: dict[str, list[TrainLengthGroupMeta]] = {}
    for row in rows:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], str)
        ):
            raise ValueError("TRAIN metadata row is invalid")
        if row[0] not in expected_set:
            continue
        if (
            type(row[1]) is not int
            or row[1] <= 0
            or row[1] % 4
            or type(row[2]) is not int
            or row[2] <= 0
        ):
            raise ValueError("TRAIN metadata row is invalid")
        grouped.setdefault(row[0], []).append(
            TrainLengthGroupMeta(row[0], row[1] // 4, row[2])
        )
    if set(grouped) != expected_set:
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


def freeze_benchmark_inputs(
    args: argparse.Namespace,
    verified_head: str,
    source_manifest_sha256: str,
) -> FrozenBenchmarkInputs:
    paths = (
        args.database,
        args.exp291_prediction,
        args.exp291_manifest,
        args.exp292_prediction,
        args.exp292_manifest,
    )
    lanes = freeze_lane_inputs(*paths[1:])
    context = BenchmarkTrainContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=list_dataset_names(paths[0], lanes),
        lanes=lanes,
        code_commit=verified_head,
        runtime_versions=_runtime_versions(),
    )
    resources = RunnerResources(
        *paths, backend_audit_path=_backend_audit_path(paths[0])
    )
    costs = _load_train_metadata_costs(paths[0], context)
    return FrozenBenchmarkInputs(
        context,
        resources,
        costs,
        BenchmarkTrainBootstrap(resources, context, context.sha256),
        paths,
        _sha256_text(source_manifest_sha256, "source manifest"),
    )


def prepare_trial_roots(root: Path) -> Mapping[int, Path]:
    root = require_exact_benchmark_root(Path(root))
    if root.exists():
        raise ValueError("benchmark root must be new")
    if not root.parent.is_dir():
        raise ValueError("benchmark parent must exist")
    root.mkdir()
    ordered = require_exact_trial_roots(
        tuple(
            root / f"workers-{count}"
            for count in FROZEN_CONFIG.worker_candidates
        )
    )
    return {
        count: path
        for count, path in zip(
            FROZEN_CONFIG.worker_candidates, ordered, strict=True
        )
    }


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


@dataclass(frozen=True)
class BoundedTrainRuntimeSmoke:
    """One non-official TRAIN-only scheduler measurement."""

    worker_count: int
    seed_workers: int
    batches_per_seed: int
    wall_seconds: float
    process_cpu_seconds: float
    completions: tuple[object, ...]
    peak_rss_bytes: int
    input_proof_sha256: str
    test_values_loaded: bool = False


def run_bounded_train_runtime_smoke(
    frozen: FrozenBenchmarkInputs,
    worker_count: int,
    *,
    status_parent: Path | None = None,
) -> BoundedTrainRuntimeSmoke:
    """Measure the frozen outer/inner layout using two normal-TRAIN batches.

    The smoke reconstructs normal TRAIN inputs and never has a TEST loader in
    its call path.  It is scheduler evidence only, not an Exp417 result.
    """
    if type(worker_count) is not int or worker_count not in FROZEN_CONFIG.worker_candidates:
        raise ValueError("bounded runtime smoke worker count is not frozen")
    selection_proof = select_balanced_benchmark_datasets(frozen.costs)
    selected = selection_proof.selected_datasets
    selected_set = set(selected)
    subset_costs = tuple(
        cost for cost in frozen.costs if cost.dataset in selected_set
    )
    if tuple(sorted(cost.dataset for cost in subset_costs)) != selected:
        raise ValueError("benchmark subset differs from frozen TRAIN-only selection")
    plan = build_shard_plan(subset_costs, worker_count)
    seed_workers = seed_workers_for_outer_workers(worker_count)
    root = (
        Path(f"/non-published-exp419-smoke-{worker_count}")
        if status_parent is None
        else Path(status_parent)
    )
    specs = tuple(
        WorkerSpec(
            assignment,
            root / f"worker-{assignment.shard_index}",
            outer_worker_count=worker_count,
            run_id=f"benchmark-{worker_count}",
        )
        for assignment in plan.assignments
    )
    child_cpu_started = _children_cpu_seconds()
    parent_cpu_started = process_time()
    started = monotonic()
    completions = spawn_bounded_train_smoke_workers(
        specs,
        frozen.bootstrap,
        batches_per_seed=FROZEN_CONFIG.bounded_train_batches_per_seed,
    )
    wall_seconds = monotonic() - started
    process_cpu_seconds = (
        process_time() - parent_cpu_started
        + _children_cpu_seconds() - child_cpu_started
    )
    parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    parent_peak_bytes = int(parent_peak) if sys.platform == "darwin" else int(parent_peak) * 1024
    peak_rss_bytes = parent_peak_bytes + sum(
        completion.peak_rss_bytes for completion in completions
    )
    input_proof_sha256 = frozen.context.sha256
    if (
        wall_seconds <= 0.0
        or any(completion.status != "completed" for completion in completions)
        or {dataset for completion in completions for dataset in completion.completed_datasets}
        != set(selected)
    ):
        raise ValueError("bounded TRAIN smoke did not cover the frozen subset")
    return BoundedTrainRuntimeSmoke(
        worker_count=worker_count,
        seed_workers=seed_workers,
        batches_per_seed=FROZEN_CONFIG.bounded_train_batches_per_seed,
        wall_seconds=wall_seconds,
        process_cpu_seconds=process_cpu_seconds,
        completions=tuple(completions),
        peak_rss_bytes=peak_rss_bytes,
        input_proof_sha256=input_proof_sha256,
        test_values_loaded=False,
    )


def _run_trials(
    frozen: FrozenBenchmarkInputs,
    trial_roots: Mapping[int, Path],
) -> MeasuredBenchmark:
    selection_proof = select_balanced_benchmark_datasets(frozen.costs)
    selected = selection_proof.selected_datasets
    if not selected:
        raise ValueError("frozen benchmark subset is empty")
    trials: list[RuntimeTrial] = []
    measurements: list[TrialRuntimeMeasurement] = []
    for worker_count in FROZEN_CONFIG.worker_candidates:
        root = trial_roots[worker_count]
        if root.exists():
            raise ValueError("every benchmark candidate root must be fresh")
        root.mkdir()
        smoke = run_bounded_train_runtime_smoke(
            frozen, worker_count, status_parent=root
        )
        completions = smoke.completions
        elapsed = smoke.wall_seconds
        process_cpu_seconds = smoke.process_cpu_seconds
        completed = tuple(
            sorted(
                dataset
                for completion in completions
                for dataset in completion.completed_datasets
            )
        )
        finite = (
            elapsed > 0.0
            and all(completion.status == "completed" for completion in completions)
            and completed == selected
        )
        parity = finite and not smoke.test_values_loaded
        fallback = "" if finite else "; ".join(
            completion.error
            for completion in completions
            if completion.error
        ) or "incomplete benchmark trial"
        scientific_evidence = sha256(
            canonical_json_bytes(
                {
                    "batches_per_seed": smoke.batches_per_seed,
                    "completed_datasets": completed,
                    "input_proof_sha256": smoke.input_proof_sha256,
                    "seed_workers": smoke.seed_workers,
                    "source_manifest_sha256":
                        frozen.source_manifest_sha256,
                    "test_values_loaded": smoke.test_values_loaded,
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
            frozen.source_manifest_sha256,
            smoke.seed_workers,
            smoke.batches_per_seed,
            smoke.input_proof_sha256,
            smoke.test_values_loaded,
        )
        measurements.append(measurement)
        trials.append(
            RuntimeTrial(
                worker_count,
                elapsed,
                smoke.peak_rss_bytes,
                parity,
                finite,
                fallback,
                selected,
                measurement.evidence_sha256,
            )
        )
    return MeasuredBenchmark(
        select_runtime(tuple(trials), selection_proof),
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
        "selection_proof": benchmark_selection_proof_payload(
            benchmark.selection_proof
        ),
        "status": benchmark.status,
        "benchmark_datasets": list(benchmark.benchmark_datasets),
        "sha256": benchmark.sha256,
    }


def runtime_measurement_payload(
    measurement: TrialRuntimeMeasurement,
) -> dict[str, object]:
    return {
        "batches_per_seed": measurement.batches_per_seed,
        "evidence_sha256": measurement.evidence_sha256,
        "input_proof_sha256": measurement.input_proof_sha256,
        "process_cpu_seconds": measurement.process_cpu_seconds,
        "process_cpu_utilization": measurement.process_cpu_utilization,
        "scientific_evidence_sha256":
            measurement.scientific_evidence_sha256,
        "seed_workers": measurement.seed_workers,
        "source_manifest_sha256": measurement.source_manifest_sha256,
        "test_values_loaded": measurement.test_values_loaded,
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
        "batches_per_seed",
        "evidence_sha256",
        "input_proof_sha256",
        "process_cpu_seconds",
        "process_cpu_utilization",
        "scientific_evidence_sha256",
        "seed_workers",
        "source_manifest_sha256",
        "test_values_loaded",
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
                raw["source_manifest_sha256"],
                raw["seed_workers"],
                raw["batches_per_seed"],
                raw["input_proof_sha256"],
                raw["test_values_loaded"],
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
        "selection_proof",
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
        benchmark_selection_proof_from_payload(payload["selection_proof"]),
    )
    if benchmark.sha256 != payload["sha256"]:
        raise ValueError("benchmark digest mismatch")
    return benchmark


def write_benchmark_evidence(
    path: Path,
    *,
    benchmark: RuntimeBenchmark,
    git_commit: str,
    context_sha256: str,
    source_manifest_sha256: str,
    trial_roots: Sequence[Path],
    runtime_measurements: Sequence[TrialRuntimeMeasurement],
) -> None:
    if select_runtime(
        benchmark.trials, benchmark.selection_proof
    ) != benchmark:
        raise ValueError(
            "benchmark selection is not the frozen passing selection"
        )
    measured = validate_benchmark_source_manifest_binding(
        benchmark, runtime_measurements, source_manifest_sha256
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
        "source_manifest_sha256": _sha256_text(
            source_manifest_sha256, "source manifest"
        ),
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
    label_guard: Callable[[], str]
    freeze_inputs: Callable[[str, str], FrozenBenchmarkInputs]
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
    source_manifest_sha256 = stages.label_guard()
    frozen = stages.freeze_inputs(head, source_manifest_sha256)
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

    def freeze(
        head: str, source_manifest_sha256: str
    ) -> FrozenBenchmarkInputs:
        frozen = freeze_benchmark_inputs(
            args, head, source_manifest_sha256
        )
        state["head"] = head
        state["source_manifest_sha256"] = source_manifest_sha256
        return frozen

    def revalidate() -> None:
        if require_clean_git() != state["head"]:
            raise ValueError("Git HEAD changed during Exp419 benchmark")
        if assert_exp419_label_isolation() != state[
            "source_manifest_sha256"
        ]:
            raise ValueError("Exp419 source manifest changed during benchmark")
        verify_input_snapshots(*paths)

    def write(
        measured: MeasuredBenchmark,
        head: str,
        frozen: FrozenBenchmarkInputs,
        roots: Mapping[int, Path],
    ) -> None:
        evidence_path = require_exact_benchmark_evidence_path(
            args.benchmark_root / BENCHMARK_FILE
        )
        ordered_roots = require_exact_trial_roots(
            tuple(
                roots[count]
                for count in FROZEN_CONFIG.worker_candidates
            )
        )
        write_benchmark_evidence(
            evidence_path,
            benchmark=measured.benchmark,
            git_commit=head,
            context_sha256=frozen.context.sha256,
            source_manifest_sha256=frozen.source_manifest_sha256,
            trial_roots=ordered_roots,
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
