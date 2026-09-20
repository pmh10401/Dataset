"""Run one bounded, normal-TRAIN-only Exp421 acceleration diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.exp421_mps_batched_acceleration.benchmark import (
    BenchmarkBoundaryError,
    MAX_MEASURED_REPEATS,
    run_train_only_benchmark,
)
from experiments.exp421_mps_batched_acceleration.config import FROZEN_CONFIG


def _official_artifact_path(path: Path) -> bool:
    components = [part for part in str(path).lower().replace("\\", "/").split("/") if part]
    return any(
        component == "exp419"
        or component.startswith("exp419_")
        or component.startswith("exp419-")
        or component in {"exp419_prediction", "exp419_checkpoint"}
        for component in components
    )


def _database_argument(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if _official_artifact_path(path):
        raise argparse.ArgumentTypeError("database path must not target official Exp419 artifacts")
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError("database must be an existing file")
    return path


def _output_root_argument(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if _official_artifact_path(path):
        raise argparse.ArgumentTypeError("output root must not target official Exp419 artifacts")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise argparse.ArgumentTypeError("output root must be a fresh or empty directory")
        return path
    if not path.parent.exists() or not path.parent.is_dir():
        raise argparse.ArgumentTypeError("output root parent must already exist")
    return path


def _dataset_argument(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("dataset must be a non-empty TRAIN dataset name")
    name = value.strip()
    lowered = name.lower()
    if any(token in lowered for token in ("test", "label", "ground_truth", "groundtruth", "location")):
        raise argparse.ArgumentTypeError("dataset must not contain TEST or evaluation fields")
    return name


def _repeats_argument(value: str) -> int:
    try:
        repeats = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("repeats must be an integer") from exc
    if not 1 <= repeats <= MAX_MEASURED_REPEATS:
        raise argparse.ArgumentTypeError(
            f"repeats must be between 1 and {MAX_MEASURED_REPEATS}"
        )
    return repeats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_exp421_mps_batched_acceleration.py",
        description=(
            "Run a fresh, bounded Exp421 normal-TRAIN diagnostic. "
            "This command never starts the official Exp419 run."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--database", required=True, type=_database_argument, help="read-only SQLite database")
    parser.add_argument(
        "--output-root",
        required=True,
        type=_output_root_argument,
        help="fresh or empty directory for this diagnostic attempt",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_dataset_argument,
        help="optional normal-TRAIN dataset name; repeat for more names",
    )
    parser.add_argument(
        "--repeats",
        type=_repeats_argument,
        default=1,
        help=f"bounded measured timing repeats (1-{MAX_MEASURED_REPEATS}; default: 1)",
    )
    return parser


def _summary(manifest: object, output_root: Path) -> dict[str, object]:
    attempt_root = Path(str(getattr(manifest, "attempt_root", output_root))).expanduser().resolve(strict=False)
    bundle = getattr(manifest, "policy_bundle", None)
    reasons = getattr(bundle, "reasons", {})
    if not isinstance(reasons, dict):
        try:
            reasons = dict(reasons)
        except (TypeError, ValueError):
            reasons = {}
    return {
        "manifest_path": str(attempt_root / "manifest.json"),
        "policy_reasons": {str(key): str(value) for key, value in reasons.items()},
        "state": str(getattr(manifest, "state", "unknown")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # Keep this diagnostic one-process and avoid competing CPU threads with MPS.
        torch.set_num_threads(1)
        manifest = run_train_only_benchmark(
            database=args.database,
            output_root=args.output_root,
            config=FROZEN_CONFIG,
            dataset_names=tuple(args.dataset) if args.dataset is not None else None,
            repeats=args.repeats,
        )
    except Exception as exc:
        parser.exit(1, f"Exp421 stopped before a valid manifest: {exc}\n")
    print(json.dumps(_summary(manifest, args.output_root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
