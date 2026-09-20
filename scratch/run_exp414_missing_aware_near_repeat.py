#!/usr/bin/env python3
"""Explicit Exp414 audit, smoke, benchmark, or full qualification CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp414_missing_aware_near_repeat.benchmark import (
    run_fixed_train_benchmark,
)
from experiments.exp414_missing_aware_near_repeat.inputs import (
    scan_missing_inventory,
)
from experiments.exp414_missing_aware_near_repeat.runner import (
    _load_population_manifest,
    _require_inventory,
    run_full_qualification,
    run_representative_smoke,
)


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


def _existing_dir(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if not result.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {result}")
    return result


def _checkpoint_dir(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if result.exists() and not result.is_dir():
        raise argparse.ArgumentTypeError(
            "checkpoint path must be a directory or new path"
        )
    if not result.exists() and not result.parent.is_dir():
        raise argparse.ArgumentTypeError("checkpoint parent must exist")
    return result


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    return path != root and path.is_relative_to(root)


def _require_temporary_output(path: Path) -> None:
    roots = {
        Path("/private/tmp").resolve(strict=False),
        Path("/tmp").resolve(strict=False),
        Path(tempfile.gettempdir()).resolve(strict=False),
    }
    if not any(_is_within(path, root) for root in roots):
        raise ValueError("smoke output must be below a temporary root")


def _require_disjoint_execution_roots(paths: tuple[Path, ...]) -> None:
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise ValueError(
                    "full checkpoint and output roots must be disjoint"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one explicitly selected Exp414 execution mode."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-audit", action="store_true")
    mode.add_argument("--representative-smoke", action="store_true")
    mode.add_argument("--acceleration-benchmark", action="store_true")
    mode.add_argument("--full-qualification", action="store_true")
    parser.add_argument("--db-path", type=_existing_file, metavar="PATH")
    parser.add_argument("--population-manifest", type=_existing_file, metavar="PATH")
    parser.add_argument(
        "--exp411-checkpoint-dir",
        type=_existing_dir,
        metavar="PATH",
    )
    parser.add_argument("--checkpoint-dir", type=_checkpoint_dir, metavar="PATH")
    parser.add_argument("--benchmark-report", type=_path, metavar="PATH")
    parser.add_argument("--output-dir", type=_new_output_dir, metavar="NEW_PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    real_paths = (
        args.db_path,
        args.population_manifest,
        args.exp411_checkpoint_dir,
        args.checkpoint_dir,
        args.benchmark_report,
    )
    if args.input_audit:
        if args.db_path is None or args.population_manifest is None:
            parser.error(
                "--input-audit requires --db-path and --population-manifest"
            )
        if (
            args.exp411_checkpoint_dir is not None
            or args.checkpoint_dir is not None
            or args.benchmark_report is not None
            or args.output_dir is not None
        ):
            parser.error(
                "checkpoint, benchmark, and output paths are forbidden "
                "in input audit mode"
            )
        datasets = _load_population_manifest(args.population_manifest)
        inventory = scan_missing_inventory(args.db_path, datasets)
        _require_inventory(inventory)
        print(canonical_json_bytes(asdict(inventory)).decode("ascii"))
        return 0
    if args.representative_smoke:
        if any(value is not None for value in real_paths):
            parser.error("real input paths are forbidden in smoke mode")
        if args.output_dir is None:
            parser.error("--output-dir is required")
        try:
            _require_temporary_output(args.output_dir)
        except ValueError as error:
            parser.error(str(error))
        result = run_representative_smoke(args.output_dir)
    elif args.acceleration_benchmark:
        if args.db_path is None or args.benchmark_report is None:
            parser.error(
                "--acceleration-benchmark requires --db-path "
                "and --benchmark-report"
            )
        if (
            args.population_manifest is not None
            or args.exp411_checkpoint_dir is not None
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
        result = run_fixed_train_benchmark(args.db_path)
        _atomic_json_write(args.benchmark_report, result)
    else:
        missing = [
            flag
            for flag, value in (
                ("--db-path", args.db_path),
                ("--population-manifest", args.population_manifest),
                ("--exp411-checkpoint-dir", args.exp411_checkpoint_dir),
                ("--checkpoint-dir", args.checkpoint_dir),
                ("--benchmark-report", args.benchmark_report),
                ("--output-dir", args.output_dir),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "--full-qualification requires " + ", ".join(missing)
            )
        try:
            _require_disjoint_execution_roots(
                (
                    args.exp411_checkpoint_dir,
                    args.checkpoint_dir,
                    args.output_dir,
                )
            )
        except ValueError as error:
            parser.error(str(error))
        result = run_full_qualification(
            args.db_path,
            args.population_manifest,
            args.output_dir,
            exp411_checkpoint_dir=args.exp411_checkpoint_dir,
            checkpoint_dir=args.checkpoint_dir,
            benchmark_report=args.benchmark_report,
        )
    print(json.dumps(dict(result), sort_keys=True))
    return 0


def _atomic_json_write(path: Path, payload: object) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(raw_temporary)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        temporary.unlink()
        _fsync_directory(path.parent)
        published = False
    except Exception:
        if published:
            path.unlink(missing_ok=True)
            try:
                _fsync_directory(path.parent)
            except OSError:
                pass
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
