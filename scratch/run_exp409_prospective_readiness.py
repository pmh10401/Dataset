#!/usr/bin/env python3
"""Smoke-only command line entrypoint for Exp409 readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp409_prospective_readiness.runner import run_smoke


def _dataset_limit(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "dataset limit must be an integer in 4..20"
        ) from error
    if not 4 <= result <= 20:
        raise argparse.ArgumentTypeError("dataset limit must be in 4..20")
    return result


def _new_output_dir(value: str) -> Path:
    result = Path(value).expanduser().resolve(strict=False)
    if result.exists():
        raise argparse.ArgumentTypeError("output directory must not exist")
    if not result.parent.is_dir():
        raise argparse.ArgumentTypeError("output parent must exist")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only the authorized Exp409 representative readiness smoke."
    )
    parser.add_argument(
        "--representative-smoke",
        action="store_true",
        required=True,
        help="authorize the preregistered representative smoke",
    )
    parser.add_argument(
        "--dataset-limit",
        type=_dataset_limit,
        required=True,
        metavar="4..20",
    )
    parser.add_argument(
        "--output-dir",
        type=_new_output_dir,
        required=True,
        metavar="NEW_PATH",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hashes = run_smoke(args.output_dir, limit=args.dataset_limit)
    print(json.dumps(dict(hashes), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
