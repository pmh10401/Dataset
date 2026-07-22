from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from experiments.exp239_248_diverse_score_sources.foundation import database_hash
from .cache_pipeline import cache_all, dataset_names, source_config
from .config import EXTERNAL_STATUS_PATH, EXPERIMENT_ID, HEARTBEAT_PATH, OUTPUT_ROOT, SOURCE
from .evaluation import evaluate_experiment
from .independent_reaggregate import reaggregate
from .prediction import predict_experiment
from .reporting import write_manifest, write_report


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_status(
    root: Path,
    status: str,
    phase: str,
    done: int,
    total: int,
    current: str | None = None,
    errors: int = 0,
    publish_global: bool = True,
) -> None:
    now = time.time()
    payload = {
        "id": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "label": "Exp255 - change-point features",
        "source": SOURCE,
        "status": status,
        "phase": phase,
        "pid": os.getpid(),
        "child_pid": os.getpid(),
        "done": int(done),
        "datasets_done": int(done),
        "expected": int(total),
        "expected_datasets": int(total),
        "progress_percent": 100 * done / max(1, total),
        "current_dataset": current,
        "errors": int(errors),
        "updated_unix": now,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "retrospective_score_source_qualification",
        "comparison_eligible": False,
        "contains_test_labels": phase in {"evaluate", "completed"},
        "detail_csv": str(root / "results.csv"),
        "stdout_log": str(root / "full.log"),
    }
    _atomic_json(root / "status.json", payload)
    if publish_global:
        _atomic_json(EXTERNAL_STATUS_PATH, payload)
        _atomic_json(HEARTBEAT_PATH, payload)


def selected_names(limit: int | None) -> tuple[str, ...]:
    names = dataset_names()
    return names[:limit] if limit else names


def audit(root: Path, names: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        root / "audit.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config": source_config(),
            "database_sha256": database_hash(),
            "datasets": len(names),
            "full_population": len(names) == 1117,
            "python": sys.version,
            "ruptures_version": importlib.metadata.version("ruptures"),
            "contains_test_labels": False,
            "retrospective": True,
        },
    )


def run(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    names = selected_names(args.dataset_limit)
    total = len(names)
    publish_global = root.resolve() == OUTPUT_ROOT.resolve()
    write_status(root, "running", args.stage, 0, total, publish_global=publish_global)
    if args.stage in {"audit", "all"}:
        audit(root, names)
        write_status(root, "running", "audit", 0, total, publish_global=publish_global)
    if args.stage in {"cache", "all"}:
        write_status(root, "running", "cache", 0, total, publish_global=publish_global)

        def progress(done: int, expected: int, current: str, errors: int) -> None:
            write_status(
                root,
                "running",
                "cache",
                done,
                expected,
                current,
                errors,
                publish_global=publish_global,
            )
            print(f"progress {done}/{expected} last={current} errors={errors}", flush=True)

        cache_all(
            root,
            workers=args.workers,
            names=names,
            reset=args.reset_cache,
            progress=progress,
        )
    if args.stage in {"predict", "all"}:
        write_status(root, "running", "predict", 0, total, publish_global=publish_global)
        predict_experiment(root, names=names)
        write_status(root, "running", "predict", total, total, publish_global=publish_global)
    if args.stage in {"evaluate", "all"}:
        write_status(root, "running", "evaluate", 0, total, publish_global=publish_global)
        evaluate_experiment(root)
        result = reaggregate(root)
        if not result["passed"]:
            raise RuntimeError("independent reaggregation failed")
        write_report(root)
        write_manifest(root, " ".join(map(str, sys.argv)))
        write_status(root, "completed", "completed", total, total, publish_global=publish_global)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--stage",
        choices=("audit", "cache", "predict", "evaluate", "all"),
        default="all",
    )
    value.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    value.add_argument("--dataset-limit", type=int)
    value.add_argument("--workers", type=int, default=7)
    value.add_argument("--reset-cache", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
