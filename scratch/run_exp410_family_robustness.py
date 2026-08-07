#!/usr/bin/env python3
"""Explicit smoke or separately authorized full entrypoint for Exp410."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp410_family_robustness.runner import (
    run_full_audit,
    run_smoke,
)


def _new_output_dir(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if result.exists():
        raise argparse.ArgumentTypeError("output directory must not exist")
    if not result.parent.is_dir():
        raise argparse.ArgumentTypeError("output parent must exist")
    return result


def _existing_path(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if not result.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {result}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Exp410 synthetic smoke or an explicitly authorized full audit."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--representative-smoke", action="store_true")
    mode.add_argument("--full-audit", action="store_true")
    parser.add_argument(
        "--input-dir",
        type=_existing_path,
        metavar="PATH",
    )
    parser.add_argument(
        "--db-path",
        type=_existing_path,
        metavar="PATH",
    )
    parser.add_argument(
        "--output-dir",
        type=_new_output_dir,
        metavar="NEW_PATH",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.representative_smoke:
        if args.input_dir is not None or args.db_path is not None:
            parser.error("--input-dir and --db-path are forbidden in smoke mode")
        hashes = run_smoke(args.output_dir)
    else:
        if args.input_dir is None or args.db_path is None:
            parser.error("--full-audit requires --input-dir and --db-path")
        if not args.input_dir.is_dir() or not args.db_path.is_file():
            parser.error("--input-dir must be a directory and --db-path a file")
        hashes = run_full_audit(
            args.input_dir,
            args.db_path,
            args.output_dir,
        )
    print(json.dumps(dict(hashes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
