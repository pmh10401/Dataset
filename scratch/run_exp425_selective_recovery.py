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
from experiments.exp425_selective_recovery.runtime import execute_runtime_command


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        try:
            checkpoint = Exp425CheckpointStore.inspect_read_only(
                args.checkpoint_root, expected_run_identity=args.run_identity
            )
            runtime = inspect_runtime_output(args.output_root, expected_run_identity=args.run_identity)
        except (OSError, ValueError) as error:
            print(f"inspect failed: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        complete = runtime.get("state") == "complete_review_only" and checkpoint.completed_count == len(checkpoint.expected_datasets)
        print(json.dumps({"checkpoint_root": str(args.checkpoint_root.absolute()), "output_root": str(args.output_root.absolute()), "completed_datasets": checkpoint.completed_count, "expected_datasets": len(checkpoint.expected_datasets), "state": runtime.get("state")}, sort_keys=True))
        return 0 if complete else 1
    if not args.launch_factory:
        print("new/resume require --launch-factory module:callable", file=sys.stderr)
        return 2
    # These are printed before the factory can open a snapshot or invoke MPS.
    paths = {
        "checkpoint_root": str(args.checkpoint_root.absolute()),
        "output_root": str(args.output_root.absolute()),
        "progress_path": str((args.output_root / "progress.json").absolute()),
        "events_path": str((args.output_root / "events.jsonl").absolute()),
        "parent_log_path": str((args.output_root / "parent.log").absolute()),
        "pid_path": str((args.output_root / "run.pid").absolute()),
        "pid": __import__("os").getpid(),
    }
    print(json.dumps(paths, sort_keys=True), flush=True)
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
        print(f"{args.command} failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2 if isinstance(error, (KeyError, ImportError, ValueError)) else 1
    return 0 if result.state == "complete_review_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
