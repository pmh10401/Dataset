#!/usr/bin/env python3
"""Explicit, non-restarting Exp425 runtime command boundary.

Task 5 intentionally exposes only validated inspection without inventing a
launch configuration.  A full ``new``/``resume`` invocation needs the frozen
database snapshot, lane artifacts, provenance and Exp424 source contract; Task
6 supplies that separately gated production assembly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery.runtime import inspect_runtime_output
from experiments.exp425_selective_recovery.runtime import execute_runtime_command, preflight_runtime_paths


_RUNTIME_STATES = frozenset({
    "initializing", "importing", "running", "finalizing", "complete_review_only", "failed",
})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_exp425_selective_recovery.py")
    parser.add_argument("command", choices=("new", "resume", "inspect"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-identity")
    parser.add_argument(
        "--launch-factory",
        help="module:callable returning frozen context/provenance/snapshot/audit and optional importer",
    )
    return parser


def _factory(specification: str) -> object:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("launch factory must be module:callable")
    candidate = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(candidate):
        raise ValueError("launch factory is not callable")
    return candidate


def _launch_paths(output_root: Path, checkpoint_root: Path) -> dict[str, object]:
    output_root, checkpoint_root = output_root.absolute(), checkpoint_root.absolute()
    return {
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "progress_path": str(output_root / "progress.json"),
        "events_path": str(output_root / "events.jsonl"),
        "parent_log_path": str(output_root / "parent.log"),
        "pid_path": str(output_root / "run.pid"),
        "pid": __import__("os").getpid(),
    }


def _nonnegative_integer(payload: object, name: str) -> int:
    value = payload.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"runtime progress {name} is invalid")
    return value


def _terminal_event_datasets(runtime: object) -> tuple[str, ...]:
    """Read the already-preflight-validated terminal event prefix without writes."""
    events_path = runtime.get("events_path")
    run_identity = runtime.get("run_identity")
    if not isinstance(events_path, str) or not isinstance(run_identity, str):
        raise ValueError("runtime event path or identity is invalid")
    raw = Path(events_path).read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("runtime event log is truncated")
    datasets: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        event = json.loads(line)
        if not isinstance(event, dict) or event.get("schema") != "exp425.runtime-event.v1" or event.get("run_identity") != run_identity:
            raise ValueError("runtime event log does not match this run")
        if event.get("event") == "dataset_terminal":
            dataset = event.get("dataset")
            if not isinstance(dataset, str) or dataset in seen:
                raise ValueError("runtime terminal event log is invalid")
            seen.add(dataset)
            datasets.append(dataset)
    return tuple(datasets)


def _validate_progress_payload(runtime: object, checkpoint: object) -> None:
    """Fail closed unless published progress exactly describes inspected records."""
    if not hasattr(runtime, "get"):
        raise ValueError("runtime progress payload is invalid")
    if runtime.get("schema") != "exp425.runtime-progress.v1":
        raise ValueError("runtime progress schema is invalid")
    state = runtime.get("state")
    if state not in _RUNTIME_STATES or runtime.get("stage") != state:
        raise ValueError("runtime progress state is invalid")
    if runtime.get("command") not in {"new", "resume"}:
        raise ValueError("runtime progress command is invalid")
    if not isinstance(runtime.get("run_identity"), str) or not runtime.get("run_identity"):
        raise ValueError("runtime progress identity is invalid")

    total = len(checkpoint.expected_datasets)
    summaries = checkpoint.record_summaries
    if checkpoint.completed_count != len(summaries):
        raise ValueError("checkpoint completion summaries are inconsistent")
    terminal_datasets = _terminal_event_datasets(runtime)
    prefix = summaries[:len(terminal_datasets)]
    if len(prefix) != len(terminal_datasets) or terminal_datasets != tuple(dataset for dataset, _, _, _, _ in prefix):
        raise ValueError("runtime terminal events are not a checkpoint prefix")
    counter_summaries = summaries if state == "complete_review_only" else prefix
    expected = {
        "total_datasets": total,
        "completed_datasets": len(counter_summaries),
        "remaining_datasets": total - len(counter_summaries),
        "imported_datasets": sum(origin == "exp424_import" for _, origin, _, _, _ in counter_summaries),
        "native_datasets": sum(origin == "exp425_native" for _, origin, _, _, _ in counter_summaries),
        "recomputed_datasets": sum(origin == "exp425_native" for _, origin, _, _, _ in counter_summaries),
        "abstained_datasets": sum(status == "abstained" for _, _, status, _, _ in counter_summaries),
        "oof_identity_insufficient_datasets": sum(reason == "oof_identity_insufficient" for _, _, _, reason, _ in counter_summaries),
        "failed_datasets": sum(status == "failed" for _, _, status, _, _ in counter_summaries),
    }
    actual = {name: _nonnegative_integer(runtime, name) for name in expected}
    if actual != expected:
        raise ValueError("runtime progress counters do not match checkpoint summaries")
    if actual["imported_datasets"] + actual["native_datasets"] != actual["completed_datasets"]:
        raise ValueError("runtime progress counters do not conserve population")
    if actual["remaining_datasets"] + actual["completed_datasets"] != actual["total_datasets"]:
        raise ValueError("runtime progress counters do not conserve population")
    if actual["recomputed_datasets"] != actual["native_datasets"]:
        raise ValueError("runtime progress recomputation counter is invalid")
    if actual["oof_identity_insufficient_datasets"] > actual["abstained_datasets"]:
        raise ValueError("runtime progress abstention counter is invalid")
    current_dataset = runtime.get("current_dataset")
    if current_dataset is not None and (not isinstance(current_dataset, str) or state != "running"):
        raise ValueError("runtime progress current dataset is invalid")
    if state == "complete_review_only":
        if (
            len(terminal_datasets) != len(summaries)
            or actual["completed_datasets"] != total
            or actual["remaining_datasets"] != 0
            or actual["failed_datasets"] != 0
            or current_dataset is not None
            or runtime.get("current_phase") is not None
        ):
            raise ValueError("complete runtime progress is inconsistent")


def _inspect_pair(output_root: Path, checkpoint_root: Path, run_identity: str | None) -> tuple[object, object]:
    """Read and cross-bind the only two durable roots without opening a run."""
    output_root, checkpoint_root = output_root.absolute(), checkpoint_root.absolute()
    # This existing preflight path uses only the read-only runtime/checkpoint
    # validators and additionally proves the append-only terminal evidence is
    # compatible with the checkpoint inventory.
    preflight_runtime_paths(
        "resume", output_root=output_root, checkpoint_root=checkpoint_root,
        run_identity=run_identity,
    )
    checkpoint = Exp425CheckpointStore.inspect_read_only(
        checkpoint_root, expected_run_identity=run_identity,
    )
    runtime = inspect_runtime_output(output_root, expected_run_identity=run_identity)
    if runtime.get("run_identity") != checkpoint.run_identity:
        raise ValueError("runtime output and checkpoint run identities do not match")
    expected_paths = _launch_paths(output_root, checkpoint_root)
    for name in ("checkpoint_root", "output_root", "progress_path", "events_path", "parent_log_path", "pid_path"):
        if runtime.get(name) != expected_paths[name]:
            raise ValueError(f"runtime output does not bind the requested {name}")
    if runtime.get("total_datasets") != len(checkpoint.expected_datasets):
        raise ValueError("runtime output total does not match checkpoint universe")
    _validate_progress_payload(runtime, checkpoint)
    return runtime, checkpoint


def _inspection_payload(runtime: object, checkpoint: object) -> dict[str, object]:
    assert isinstance(runtime, dict) or hasattr(runtime, "get")
    return {
        "checkpoint_root": runtime.get("checkpoint_root"),
        "output_root": runtime.get("output_root"),
        "progress_path": runtime.get("progress_path"),
        "events_path": runtime.get("events_path"),
        "parent_log_path": runtime.get("parent_log_path"),
        "pid_path": runtime.get("pid_path"),
        "pid": runtime.get("pid"),
        "completed_datasets": checkpoint.completed_count,
        "expected_datasets": len(checkpoint.expected_datasets),
        "state": runtime.get("state"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        try:
            runtime, checkpoint = _inspect_pair(
                args.output_root, args.checkpoint_root, args.run_identity,
            )
        except (OSError, ValueError) as error:
            print(f"inspect failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 2
        complete = runtime.get("state") == "complete_review_only" and checkpoint.completed_count == len(checkpoint.expected_datasets)
        print(json.dumps(_inspection_payload(runtime, checkpoint), sort_keys=True))
        return 0 if complete else 1
    try:
        # Reject lifecycle mistakes before importing a launch module that may
        # open the frozen database or initialize MPS.
        if args.command == "resume":
            runtime, checkpoint = _inspect_pair(
                args.output_root, args.checkpoint_root, args.run_identity,
            )
            if runtime.get("state") == "complete_review_only" and checkpoint.completed_count == len(checkpoint.expected_datasets):
                print(json.dumps(_inspection_payload(runtime, checkpoint), sort_keys=True))
                return 0
        else:
            preflight_runtime_paths(
                args.command, output_root=args.output_root,
                checkpoint_root=args.checkpoint_root, run_identity=args.run_identity,
            )
    except (OSError, ValueError) as error:
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if not args.launch_factory:
        print("new/resume require --launch-factory module:callable", file=sys.stderr)
        return 2
    # These are printed before the factory can open a snapshot or invoke MPS.
    paths = _launch_paths(args.output_root, args.checkpoint_root)
    print(json.dumps(paths, sort_keys=True), flush=True)
    launch: object | None = None
    try:
        factory = _factory(args.launch_factory)
    except (ImportError, ValueError) as error:
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    try:
        launch = factory()
    except Exception as error:
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if not isinstance(launch, dict):
        print(f"{args.command} failed: ValueError: launch factory must return a mapping", file=sys.stderr)
        return 2
    try:
        provenance = launch["provenance"]
        context = launch["context"]
        snapshot = launch["snapshot"]
        audit = launch["audit"]
        adoption = launch.get("adoption")
    except KeyError as error:
        snapshot = launch.get("snapshot")
        close = getattr(snapshot, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    try:
        result = execute_runtime_command(
            args.command,
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
            provenance=provenance, context=context,
            snapshot=snapshot, audit=audit, adoption=adoption, importer=launch.get("importer"),
        )
    except (OSError, ValueError) as error:
        close = getattr(snapshot, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0 if result.state == "complete_review_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
