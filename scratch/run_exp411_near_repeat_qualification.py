#!/usr/bin/env python3
"""Explicit representative smoke or separately authorized Exp411 full runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp411_near_repeat_qualification.runner import (
    run_full_qualification,
    run_representative_smoke,
)
from experiments.exp411_near_repeat_qualification.benchmark import (
    run_fixed_train_benchmark,
)
from experiments.exp403_408.artifact_io import canonical_json_bytes


def _new_output_dir(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if result.exists():
        raise argparse.ArgumentTypeError("output directory must not exist")
    if not result.parent.is_dir():
        raise argparse.ArgumentTypeError("output parent must exist")
    return result


def _existing_file(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {result}")
    return result


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _checkpoint_dir(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if result.exists() and not result.is_dir():
        raise argparse.ArgumentTypeError(
            "checkpoint path must be a directory or new path"
        )
    if not result.exists() and not result.parent.is_dir():
        raise argparse.ArgumentTypeError("checkpoint parent must exist")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Exp411 smoke or explicitly authorized full qualification."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--representative-smoke", action="store_true")
    mode.add_argument("--full-qualification", action="store_true")
    mode.add_argument("--acceleration-benchmark", action="store_true")
    parser.add_argument("--db-path", type=_existing_file, metavar="PATH")
    parser.add_argument("--population-manifest", type=_existing_file, metavar="PATH")
    parser.add_argument("--checkpoint-dir", type=_checkpoint_dir, metavar="PATH")
    parser.add_argument("--benchmark-report", type=_path, metavar="PATH")
    parser.add_argument("--output-dir", type=_new_output_dir, metavar="NEW_PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.acceleration_benchmark and args.output_dir is None:
        parser.error("--output-dir is required")
    if args.acceleration_benchmark:
        if args.db_path is None or args.benchmark_report is None:
            parser.error(
                "--acceleration-benchmark requires --benchmark-report "
                "and --db-path"
            )
        if (
            args.population_manifest is not None
            or args.checkpoint_dir is not None
            or args.output_dir is not None
        ):
            parser.error(
                "population, checkpoint, and output paths are forbidden "
                "in benchmark mode"
            )
        if args.benchmark_report.exists():
            parser.error("benchmark report must not already exist")
        if not args.benchmark_report.parent.is_dir():
            parser.error("benchmark report parent must exist")
        report = run_fixed_train_benchmark(args.db_path)
        _atomic_json_write(args.benchmark_report, report)
        print(json.dumps(dict(report), sort_keys=True))
        return 0
    if args.representative_smoke:
        if (
            args.db_path is not None
            or args.population_manifest is not None
            or args.checkpoint_dir is not None
            or args.benchmark_report is not None
        ):
            parser.error("real input paths are forbidden in smoke mode")
        hashes = run_representative_smoke(args.output_dir)
    else:
        if (
            args.db_path is None
            or args.population_manifest is None
            or args.checkpoint_dir is None
            or args.benchmark_report is None
        ):
            parser.error(
                "--full-qualification requires --db-path, "
                "--population-manifest, --checkpoint-dir, "
                "and --benchmark-report"
            )
        hashes = run_full_qualification(
            args.db_path,
            args.population_manifest,
            args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            benchmark_report=args.benchmark_report,
        )
    print(json.dumps(dict(hashes), sort_keys=True))
    return 0


def _atomic_json_write(path: Path, payload: object) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError("benchmark report appeared during write")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
