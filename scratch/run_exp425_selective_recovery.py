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
        launch = _factory(args.launch_factory)()
        if not isinstance(launch, dict):
            raise ValueError("launch factory must return a mapping")
        result = execute_runtime_command(
            args.command,
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
            provenance=launch["provenance"], context=launch["context"],
            snapshot=launch["snapshot"], audit=launch["audit"], importer=launch.get("importer"),
        )
    except (KeyError, ImportError, OSError, ValueError) as error:
        if isinstance(launch, dict):
            snapshot = launch.get("snapshot")
            close = getattr(snapshot, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        # Factory shape is an argument/contract error; failures after a valid
        # factory launch (MPS, integrity, DB, guard) are runtime failures.
        return 2 if isinstance(error, (KeyError, ImportError)) else 1
    return 0 if result.state == "complete_review_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
