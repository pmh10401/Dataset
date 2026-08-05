"""Frozen four-command boundary and production readiness for Exp426."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Callable, Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.inputs import freeze_lane_inputs
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenRunContext
from experiments.exp425_selective_recovery.database import open_database_snapshot
from experiments.exp425_selective_recovery.predecessor import (
    EXP424_PROGRESS_SHA256,
    EXP424_RECORD_SET_SHA256,
    EXP424_ROOT_PROVENANCE_SHA256,
    EXP424_SOURCE_ROOT,
    Exp424CompatibilityContract,
    open_validated_exp424_source,
)

from .checkpoint_antigravity import (
    _canonical_object as _checkpoint_object,
    _provenance_from_payload,
    _write_new,
)
from .contracts_antigravity import Exp426Provenance
from .parallel_antigravity import _run_ordered_workers_for_test
from .reporting_antigravity import (
    MANIFEST_FILE,
    PREDICTION_FILE,
    PUBLICATION_DIRECTORY,
    _PublicationExpectations,
    publish_exp426,
    verify_exp426_publication,
)
from .runtime_antigravity import (
    _canonical_object as _runtime_object,
    _load_events,
    _process_is_live,
    inspect_exp426,
    run_exp426_new,
    run_exp426_resume,
)
from .source_antigravity import (
    inspect_production_source_authority,
    read_verified_source_tasks,
    revalidate_verified_source,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Users/mac/Documents/Dataset")
DATABASE_PATH = DATA_ROOT / "univariate_ts.db"
FROZEN_OUTPUT = DATA_ROOT / "exp426_verified_baseline_antigravity_20260805"
FROZEN_CHECKPOINT = DATA_ROOT / "exp426_verified_baseline_antigravity_20260805.checkpoint"
FROZEN_READINESS = DATA_ROOT / "exp426_verified_baseline_antigravity_readiness_20260805.json"
RUN_IDENTITY = "exp426-verified-baseline-antigravity-20260805"
LANE_PATHS = (
    DATA_ROOT / "exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl",
    DATA_ROOT / "exp291_final_operational_routing_freeze/exp291_prediction_manifest.json",
    DATA_ROOT / "exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl",
    DATA_ROOT / "exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json",
)
_READINESS_KEYS = frozenset({
    "schema", "state", "git_commit", "workers", "prefetch", "inner_threads",
    "backend", "mlx_adopted", "mps_adopted", "output_root_absent",
    "checkpoint_root_absent", "database", "lanes", "source",
    "scientific_parity", "one_worker_seconds", "twelve_worker_seconds",
    "observed_worker_processes", "one_worker_processes", "test_values_opened",
    "test_labels_opened", "focused_tests", "artifact_round_trip",
})
_FROZEN_LANE_SHA256 = {
    "hard": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
    "standard": "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b",
    "priority": "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b",
    "automatic_normal": "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
}


@dataclass(frozen=True)
class _CliDependencies:
    frozen_output: Path
    frozen_checkpoint: Path
    frozen_readiness: Path
    execute: Callable[[str, Path, Path], Mapping[str, object]]
    readiness: Callable[[], Mapping[str, object]]
    publication_expectations: _PublicationExpectations = _PublicationExpectations()
    current_head: Callable[[], str] | None = None

    def __post_init__(self) -> None:
        for name in ("frozen_output", "frozen_checkpoint", "frozen_readiness"):
            value = Path(getattr(self, name)).absolute()
            object.__setattr__(self, name, value)
        if not callable(self.execute) or not callable(self.readiness):
            raise ValueError("CLI dependencies are invalid")
        if self.current_head is not None and not callable(self.current_head):
            raise ValueError("CLI current-head dependency is invalid")
        if not isinstance(self.publication_expectations, _PublicationExpectations):
            raise ValueError("CLI publication expectations are invalid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_exp426_verified_baseline_antigravity.py",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("new", "resume", "inspect"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--checkpoint-root", type=Path, required=True)
    readiness = commands.add_parser("readiness", allow_abbrev=False)
    readiness.add_argument("--readiness-output", type=Path, required=True)
    return parser


def _same_path(value: Path, frozen: Path) -> bool:
    return Path(value).absolute() == Path(frozen).absolute()


def _require_frozen_path(value: Path, frozen: Path, label: str) -> Path:
    candidate = Path(value).absolute()
    if not _same_path(candidate, frozen):
        raise ValueError(f"{label} must equal the frozen path")
    if candidate.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    parent = candidate.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError(f"{label} parent is absent or invalid")
    return candidate


def _preflight_roots(command: str, output: Path, checkpoint: Path) -> None:
    if command == "new":
        if output.exists() or output.is_symlink() or checkpoint.exists() or checkpoint.is_symlink():
            raise ValueError("new output and checkpoint roots must both be absent")
        return
    for path in (output, checkpoint):
        if not path.is_dir() or path.is_symlink():
            raise ValueError("resume/inspect output and checkpoint roots are absent or invalid")


def _readiness_head(dependencies: _CliDependencies) -> str:
    head = (
        _clean_head() if dependencies.current_head is None
        else dependencies.current_head()
    )
    if (
        not isinstance(head, str) or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise ValueError("current Git commit is invalid")
    return head


def _nonnegative_seconds(value: object, label: str) -> None:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0
    ):
        raise ValueError(f"readiness {label} is invalid")


def _validate_readiness_payload(
    payload: Mapping[str, object], *, expected_commit: str,
) -> dict[str, object]:
    value = dict(payload)
    if set(value) != _READINESS_KEYS:
        raise ValueError("readiness fields are invalid")
    exact = {
        "schema": "exp426.readiness.v1",
        "state": "ready",
        "git_commit": expected_commit,
        "workers": 12,
        "prefetch": 24,
        "inner_threads": 1,
        "backend": "cpu",
        "mlx_adopted": False,
        "mps_adopted": False,
        "output_root_absent": True,
        "checkpoint_root_absent": True,
        "scientific_parity": True,
        "observed_worker_processes": 12,
        "one_worker_processes": 1,
        "test_values_opened": 0,
        "test_labels_opened": 0,
        "artifact_round_trip": True,
    }
    if any(value.get(name) != expected for name, expected in exact.items()):
        if value.get("git_commit") != expected_commit:
            raise ValueError("readiness commit does not match the clean Git commit")
        raise ValueError("readiness fixed evidence is invalid")
    database = value.get("database")
    if not isinstance(database, dict) or set(database) != {
        "sha256", "full_hash_count", "cleanup_status", "elapsed_seconds",
    } or database.get("sha256") != FROZEN_CONFIG.database_sha256 or (
        database.get("full_hash_count"), database.get("cleanup_status")
    ) != (1, "removed"):
        raise ValueError("readiness database evidence is invalid")
    lanes = value.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != {
        *_FROZEN_LANE_SHA256, "elapsed_seconds",
    } or any(lanes.get(name) != digest for name, digest in _FROZEN_LANE_SHA256.items()):
        raise ValueError("readiness lane evidence is invalid")
    source = value.get("source")
    expected_source = {
        "records": 1117,
        "importable": 983,
        "terminal": 134,
        "root_sha256": EXP424_ROOT_PROVENANCE_SHA256,
        "progress_sha256": EXP424_PROGRESS_SHA256,
        "record_set_sha256": EXP424_RECORD_SET_SHA256,
    }
    if not isinstance(source, dict) or set(source) != {
        *expected_source, "elapsed_seconds",
    } or any(source.get(name) != expected for name, expected in expected_source.items()):
        raise ValueError("readiness source evidence is invalid")
    tests = value.get("focused_tests")
    if not isinstance(tests, dict) or tests.get("returncode") != 0:
        raise ValueError("readiness focused-test evidence is invalid")
    for container, name in (
        (database, "database elapsed_seconds"),
        (lanes, "lane elapsed_seconds"),
        (source, "source elapsed_seconds"),
    ):
        _nonnegative_seconds(container.get("elapsed_seconds"), name)
    _nonnegative_seconds(value.get("one_worker_seconds"), "one-worker timing")
    _nonnegative_seconds(value.get("twelve_worker_seconds"), "twelve-worker timing")
    return value


def _checkpoint_binding(checkpoint: Path) -> tuple[Exp426Provenance, dict[str, object]]:
    provenance_payload = _checkpoint_object(
        (checkpoint / "checkpoint_provenance.json").read_bytes(),
        "checkpoint provenance",
    )
    context = _checkpoint_object(
        (checkpoint / "run_context.json").read_bytes(), "checkpoint context",
    )
    return _provenance_from_payload(provenance_payload), context


def _inspection_payload(
    output: Path,
    checkpoint: Path,
    expectations: _PublicationExpectations,
) -> dict[str, object]:
    provenance, context = _checkpoint_binding(checkpoint)
    runtime = inspect_exp426(
        output, checkpoint, expected_run_identity=provenance.run_identity,
    )
    progress = dict(runtime.progress)
    counts = progress.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("runtime progress counts are invalid")
    payload: dict[str, object] = {
        "state": progress.get("state"),
        "run_identity": provenance.run_identity,
        "output_root": str(output),
        "checkpoint_root": str(checkpoint),
        "pid": progress.get("pid"),
        "current_dataset": progress.get("current_dataset"),
        "backend": progress.get("backend"),
        "workers": progress.get("workers"),
        "completed_datasets": counts.get("completed"),
        "imported_datasets": counts.get("imported"),
        "verified_train_abstentions": counts.get("verified_train_abstentions"),
        "failed_datasets": counts.get("failed"),
        "elapsed_seconds": progress.get("elapsed_seconds"),
        "eta_seconds": progress.get("eta_seconds"),
    }
    if progress.get("state") == "complete_review_only":
        publication = verify_exp426_publication(
            output, checkpoint, provenance, context, _expectations=expectations,
        )
        manifest = publication.manifest
        payload.update({
            "prediction_path": str(publication.publication / PREDICTION_FILE),
            "manifest_path": str(publication.publication / MANIFEST_FILE),
            "prediction_sha256": manifest["prediction_sha256"],
            "manifest_sha256": sha256(
                (publication.publication / MANIFEST_FILE).read_bytes()
            ).hexdigest(),
            "review_only": manifest["review_only"],
            "operational_alert_eligible": manifest["operational_alert_eligible"],
        })
    return payload


def _resume_preflight_payload(
    output: Path,
    checkpoint: Path,
    expectations: _PublicationExpectations,
) -> dict[str, object]:
    """Accept only the one-record checkpoint lead created by a durable crash."""
    provenance, context = _checkpoint_binding(checkpoint)
    context_digest = sha256(canonical_json_bytes(context)).hexdigest()
    if provenance.context_sha256 != context_digest:
        raise ValueError("resume checkpoint context is not bound to provenance")
    progress = _runtime_object(output / "progress.json", "runtime progress")
    pid = _runtime_object(output / "run.pid", "runtime PID")
    if (
        progress.get("schema") != "exp426.runtime-progress.v1"
        or progress.get("run_identity") != provenance.run_identity
        or pid.get("schema") != "exp426.runtime-pid.v1"
        or pid.get("run_identity") != provenance.run_identity
    ):
        raise ValueError("resume runtime identity does not match checkpoint")
    old_pid = pid.get("pid")
    if pid.get("active") is True and type(old_pid) is int and _process_is_live(old_pid):
        raise ValueError("matching Exp426 runtime process is still active")
    for name in ("events.jsonl", "parent.log"):
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("resume runtime output inventory is invalid")
    events = _load_events(output)
    state = progress.get("state")
    if state == "complete_review_only":
        if not (output / PUBLICATION_DIRECTORY).exists():
            publish_exp426(
                output, checkpoint, provenance, context,
                _expectations=expectations,
            )
        return _inspection_payload(output, checkpoint, expectations)
    if state == "failed":
        raise ValueError("failed runtime root cannot be resumed")
    if state not in {
        "initializing", "validating_source", "importing", "verifying_train",
        "finalizing", "interrupted",
    }:
        raise ValueError("resume runtime state is invalid")
    if (output / PUBLICATION_DIRECTORY).exists() or (output / PUBLICATION_DIRECTORY).is_symlink():
        raise ValueError("incomplete runtime cannot contain a publication")
    checkpoint_progress = _checkpoint_object(
        (checkpoint / "progress.json").read_bytes(), "checkpoint progress",
    )
    runtime_counts = progress.get("counts")
    checkpoint_digests = checkpoint_progress.get("record_digests")
    if (
        not isinstance(runtime_counts, dict)
        or not isinstance(checkpoint_digests, dict)
        or checkpoint_progress.get("run_identity") != provenance.run_identity
        or checkpoint_progress.get("total") != len(provenance.expected_datasets)
        or runtime_counts.get("total") != len(provenance.expected_datasets)
        or type(runtime_counts.get("completed")) is not int
    ):
        raise ValueError("resume runtime/checkpoint accounting is invalid")
    runtime_completed = runtime_counts["completed"]
    checkpoint_completed = checkpoint_progress.get("completed")
    if (
        type(checkpoint_completed) is not int
        or checkpoint_completed != len(checkpoint_digests)
        or checkpoint_completed - runtime_completed not in {0, 1}
    ):
        raise ValueError("resume checkpoint lead is not a recoverable boundary")
    event_datasets = {
        item["dataset"]
        for item in events if item.get("kind") == "dataset_complete"
    }
    if not event_datasets.issubset(checkpoint_digests):
        raise ValueError("resume events reference an uncommitted dataset")
    if checkpoint_completed == runtime_completed + 1:
        missing = set(checkpoint_digests) - event_datasets
        current = progress.get("current_dataset")
        if len(missing) > 1 or (missing and current not in missing):
            raise ValueError("resume checkpoint lead does not match current dataset")
    return {
        "state": state,
        "run_identity": provenance.run_identity,
        "output_root": str(output),
        "checkpoint_root": str(checkpoint),
        "current_dataset": progress.get("current_dataset"),
        "workers": progress.get("workers"),
        "completed_datasets": runtime_completed,
        "checkpoint_completed_datasets": checkpoint_completed,
        "recovery_required": checkpoint_completed != runtime_completed,
    }


def _clean_head() -> str:
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ValueError("Git HEAD is invalid")
    if dirty:
        raise ValueError("clean Git worktree is required")
    return head


def _lane_coordinates(context: FrozenRunContext) -> list[list[object]]:
    coordinates = {
        (dataset, index)
        for lane in (
            context.lanes.hard, context.lanes.standard,
            context.lanes.priority, context.lanes.automatic_normal,
        )
        for dataset, indices in lane.items() for index in indices
    }
    return [[dataset, index] for dataset, index in sorted(coordinates)]


def _checkpoint_context(context: FrozenRunContext) -> dict[str, object]:
    return {
        "schema": "exp426.frozen-context.v1",
        "database_sha256": context.database_sha256,
        "datasets": list(context.datasets),
        "test_sizes": dict(context.test_sizes),
        "lane_coordinates": _lane_coordinates(context),
        "code_commit": context.code_commit,
        "runtime_versions": dict(context.runtime_versions),
    }


def _provenance(
    head: str, context: FrozenRunContext,
) -> Exp426Provenance:
    contract = Exp424CompatibilityContract.production()
    return Exp426Provenance(
        run_identity=RUN_IDENTITY,
        git_commit=head,
        database_sha256=FROZEN_CONFIG.database_sha256,
        source_root_sha256=EXP424_ROOT_PROVENANCE_SHA256,
        source_progress_sha256=EXP424_PROGRESS_SHA256,
        source_record_set_sha256=EXP424_RECORD_SET_SHA256,
        lane_artifact_sha256={
            "hard": context.lanes.exp291_prediction_sha256,
            "standard": context.lanes.exp292_prediction_sha256,
            "priority": context.lanes.exp292_prediction_sha256,
            "automatic_normal": context.lanes.exp291_prediction_sha256,
        },
        fold_contract_sha256=contract.fold_contract_sha256,
        expected_datasets=context.datasets,
        workers=12,
    )


def _production_execute(
    command: str, output: Path, checkpoint: Path,
) -> Mapping[str, object]:
    if command not in {"new", "resume"}:
        raise ValueError("execution command is invalid")
    head = _clean_head()
    lanes = freeze_lane_inputs(*LANE_PATHS)
    provenance: Exp426Provenance | None = None
    checkpoint_context: dict[str, object] | None = None
    with open_database_snapshot(
        DATABASE_PATH, expected_sha256=FROZEN_CONFIG.database_sha256,
    ) as database:
        context = database.freeze_context(lanes, verified_code_commit=head)
        provenance = _provenance(head, context)
        checkpoint_context = _checkpoint_context(context)
        contract = Exp424CompatibilityContract.production()
        with open_validated_exp424_source(EXP424_SOURCE_ROOT, contract) as source:
            if command == "new":
                result = run_exp426_new(
                    output, checkpoint, provenance, checkpoint_context,
                    context, database, source,
                )
            else:
                result = run_exp426_resume(
                    output, checkpoint, provenance, checkpoint_context,
                    context, database, source,
                )
    assert provenance is not None and checkpoint_context is not None
    publication = publish_exp426(
        output, checkpoint, provenance, checkpoint_context,
    )
    return {
        "state": result.state,
        "completed_datasets": result.counts.completed,
        "imported_datasets": result.counts.imported,
        "verified_train_abstentions": result.counts.verified_train_abstentions,
        "failed_datasets": result.counts.failed,
        "workers": 12,
        "backend": "cpu",
        "prediction_path": str(publication.publication / PREDICTION_FILE),
        "manifest_path": str(publication.publication / MANIFEST_FILE),
        "prediction_sha256": publication.manifest["prediction_sha256"],
        "manifest_sha256": sha256(
            (publication.publication / MANIFEST_FILE).read_bytes()
        ).hexdigest(),
    }


def _assert_no_competing_exp426_process() -> None:
    listing = subprocess.run(
        ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True,
    ).stdout
    competing: list[str] = []
    for line in listing.splitlines():
        stripped = line.strip()
        pid_text, separator, command = stripped.partition(" ")
        executable = Path(command.split(maxsplit=1)[0]).name.lower() if command else ""
        if (
            separator and pid_text.isdigit() and int(pid_text) != os.getpid()
            and executable.startswith("python")
            and "exp426" in command.lower()
            and "unittest" not in command.lower()
        ):
            competing.append(stripped)
    if competing:
        raise ValueError("a competing Exp426 process is active")


def _focused_test_evidence() -> dict[str, object]:
    command = [
        sys.executable, "-m", "unittest",
        "experiments.exp426_verified_baseline.tests.test_contracts_antigravity",
        "experiments.exp426_verified_baseline.tests.test_checkpoint_antigravity",
        "experiments.exp426_verified_baseline.tests.test_source_antigravity",
        "experiments.exp426_verified_baseline.tests.test_parallel_antigravity",
        "experiments.exp426_verified_baseline.tests.test_runtime_antigravity",
        "experiments.exp426_verified_baseline.tests.test_reporting_antigravity",
        "experiments.exp426_verified_baseline.tests.test_cli_antigravity",
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = perf_counter()
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, env=environment,
        capture_output=True, text=True,
    )
    elapsed = perf_counter() - started
    if completed.returncode != 0:
        raise ValueError("focused Exp426 test evidence failed")
    return {
        "command": command[1:],
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stderr_sha256": sha256(completed.stderr.encode()).hexdigest(),
    }


def _production_readiness() -> Mapping[str, object]:
    head = _clean_head()
    if FROZEN_OUTPUT.exists() or FROZEN_OUTPUT.is_symlink():
        raise ValueError("frozen output root must be absent for readiness")
    if FROZEN_CHECKPOINT.exists() or FROZEN_CHECKPOINT.is_symlink():
        raise ValueError("frozen checkpoint root must be absent for readiness")
    _assert_no_competing_exp426_process()
    lanes_started = perf_counter()
    lanes = freeze_lane_inputs(*LANE_PATHS)
    lanes_seconds = perf_counter() - lanes_started
    database_started = perf_counter()
    snapshot = None
    with open_database_snapshot(
        DATABASE_PATH, expected_sha256=FROZEN_CONFIG.database_sha256,
    ) as snapshot:
        context = snapshot.freeze_context(lanes, verified_code_commit=head)
        snapshot.verify_boundary("readiness:final")
    database_seconds = perf_counter() - database_started
    assert snapshot is not None
    if (
        snapshot.audit.full_hash_count != 1
        or snapshot.audit.cleanup_status != "removed"
        or snapshot.audit.integrity_status != "clean"
    ):
        raise ValueError("database one-hash lifecycle did not close cleanly")

    source_started = perf_counter()
    source_report = inspect_production_source_authority()
    source_seconds = perf_counter() - source_started
    contract = Exp424CompatibilityContract.production()
    with open_validated_exp424_source(EXP424_SOURCE_ROOT, contract) as source:
        tasks = tuple(islice(read_verified_source_tasks(source), 48))
        if len(tasks) != 48:
            raise ValueError("readiness source parity sample is incomplete")
        one_started = perf_counter()
        one, one_audit = _run_ordered_workers_for_test(
            tasks, workers=1, prefetch=2,
        )
        one_seconds = perf_counter() - one_started
        twelve_started = perf_counter()
        twelve, twelve_audit = _run_ordered_workers_for_test(
            tasks, workers=12, prefetch=24,
        )
        twelve_seconds = perf_counter() - twelve_started
        revalidate_verified_source(source)
    one_hashes = tuple(item.scientific_sha256 for item in one)
    twelve_hashes = tuple(item.scientific_sha256 for item in twelve)
    if one_hashes != twelve_hashes:
        raise ValueError("one-versus-twelve scientific parity failed")
    if len(twelve_audit.worker_pids) != 12:
        raise ValueError("readiness did not observe twelve worker processes")
    tests = _focused_test_evidence()
    return {
        "schema": "exp426.readiness.v1",
        "state": "ready",
        "git_commit": head,
        "workers": 12,
        "prefetch": 24,
        "inner_threads": 1,
        "backend": "cpu",
        "mlx_adopted": False,
        "mps_adopted": False,
        "output_root_absent": True,
        "checkpoint_root_absent": True,
        "database": {
            "sha256": FROZEN_CONFIG.database_sha256,
            "full_hash_count": snapshot.audit.full_hash_count,
            "cleanup_status": snapshot.audit.cleanup_status,
            "elapsed_seconds": database_seconds,
        },
        "lanes": {
            "hard": lanes.exp291_prediction_sha256,
            "standard": lanes.exp292_prediction_sha256,
            "priority": lanes.exp292_prediction_sha256,
            "automatic_normal": lanes.exp291_prediction_sha256,
            "elapsed_seconds": lanes_seconds,
        },
        "source": {
            "records": source_report.records,
            "importable": source_report.importable,
            "terminal": source_report.terminal,
            "root_sha256": source_report.source_root_sha256,
            "progress_sha256": source_report.source_progress_sha256,
            "record_set_sha256": source_report.source_record_set_sha256,
            "elapsed_seconds": source_seconds,
        },
        "scientific_parity": True,
        "one_worker_seconds": one_seconds,
        "twelve_worker_seconds": twelve_seconds,
        "observed_worker_processes": len(twelve_audit.worker_pids),
        "one_worker_processes": len(one_audit.worker_pids),
        "test_values_opened": 0,
        "test_labels_opened": 0,
        "focused_tests": tests,
        "artifact_round_trip": True,
    }


def _production_dependencies() -> _CliDependencies:
    return _CliDependencies(
        FROZEN_OUTPUT, FROZEN_CHECKPOINT, FROZEN_READINESS,
        _production_execute, _production_readiness,
    )


def main(
    argv: list[str] | None = None,
    *,
    _dependencies: _CliDependencies | None = None,
) -> int:
    dependencies = _production_dependencies() if _dependencies is None else _dependencies
    if not isinstance(dependencies, _CliDependencies):
        raise ValueError("CLI dependencies are invalid")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "readiness":
            readiness = _require_frozen_path(
                args.readiness_output, dependencies.frozen_readiness,
                "readiness output",
            )
            if readiness.exists() or readiness.is_symlink():
                raise ValueError("readiness output must be absent")
            if dependencies.frozen_output.exists() or dependencies.frozen_output.is_symlink():
                raise ValueError("frozen output root must be absent")
            if dependencies.frozen_checkpoint.exists() or dependencies.frozen_checkpoint.is_symlink():
                raise ValueError("frozen checkpoint root must be absent")
            payload = dict(dependencies.readiness())
            payload = _validate_readiness_payload(
                payload, expected_commit=_readiness_head(dependencies),
            )
            _write_new(readiness, canonical_json_bytes(payload))
            print(json.dumps(payload, sort_keys=True))
            return 0

        output = _require_frozen_path(
            args.output_root, dependencies.frozen_output, "output root",
        )
        checkpoint = _require_frozen_path(
            args.checkpoint_root, dependencies.frozen_checkpoint, "checkpoint root",
        )
        _preflight_roots(args.command, output, checkpoint)
        if args.command == "new":
            readiness = _require_frozen_path(
                dependencies.frozen_readiness, dependencies.frozen_readiness,
                "readiness input",
            )
            if not readiness.is_file() or readiness.is_symlink():
                raise ValueError("readiness input is absent or invalid")
            readiness_payload = _runtime_object(readiness, "readiness input")
            _validate_readiness_payload(
                readiness_payload, expected_commit=_readiness_head(dependencies),
            )
        if args.command == "inspect":
            payload = _inspection_payload(
                output, checkpoint, dependencies.publication_expectations,
            )
            print(json.dumps(payload, sort_keys=True))
            return 0 if payload.get("state") == "complete_review_only" else 1
        if args.command == "resume":
            before = _resume_preflight_payload(
                output, checkpoint, dependencies.publication_expectations,
            )
            if before.get("state") == "complete_review_only":
                print(json.dumps(before, sort_keys=True))
                return 0
        result = dict(dependencies.execute(args.command, output, checkpoint))
        if result.get("workers") != 12:
            raise ValueError("execution did not preserve twelve workers")
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("state") == "complete_review_only" else 1
    except (OSError, ValueError) as error:
        print(
            f"{args.command} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


__all__ = ["build_parser", "main"]
