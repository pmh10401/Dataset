from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp239_248_diverse_score_sources.foundation import database_hash
from experiments.exp251_sktime_statistical_features.cache_pipeline import cache_all, dataset_names, source_config
from experiments.exp251_sktime_statistical_features.config import (
    DB_PATH, EXPERIMENT, EXTERNAL_STATUS_PATH, HEARTBEAT_PATH, OUTPUT_ROOT, SOURCE,
)
from experiments.exp251_sktime_statistical_features.evaluation import evaluate_experiment
from experiments.exp251_sktime_statistical_features.independent_reaggregate import reaggregate
from experiments.exp251_sktime_statistical_features.prediction import predict_experiment
from experiments.exp251_sktime_statistical_features.reporting import write_manifest, write_report


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_status(root: Path, status: str, phase: str, done: int, total: int,
                 current: str | None = None, errors: int = 0,
                 publish_external: bool = False) -> None:
    now = time.time()
    payload = {
        "id": f"experiment_{EXPERIMENT}_{SOURCE}",
        "experiment_id": f"experiment_{EXPERIMENT}_{SOURCE}",
        "label": "Exp251 - sktime statistical features", "source": SOURCE,
        "status": status, "phase": phase, "pid": os.getpid(), "child_pid": os.getpid(),
        "done": int(done), "datasets_done": int(done), "expected": int(total),
        "expected_datasets": int(total), "progress_percent": 100.0 * done / max(1, total),
        "current_dataset": current, "errors": int(errors), "updated_unix": now,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "retrospective_score_source_qualification", "comparison_eligible": False,
        "contains_test_labels": phase == "completed",
    }
    _atomic_json(root / "status.json", payload)
    if publish_external:
        _atomic_json(EXTERNAL_STATUS_PATH, payload)
        _atomic_json(HEARTBEAT_PATH, payload)


def selected_names(limit: int | None, list_file: Path | None) -> tuple[str, ...]:
    population = dataset_names()
    if list_file is None:
        return population[:limit] if limit else population
    requested = tuple(line.strip() for line in list_file.read_text().splitlines() if line.strip())
    if len(requested) != len(set(requested)):
        raise ValueError("dataset list contains duplicates")
    unknown = sorted(set(requested) - set(population))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return requested


def audit(root: Path, names: tuple[str, ...]) -> None:
    _atomic_json(root / "audit.json", {
        "experiment": EXPERIMENT, "source": SOURCE, "config": source_config(),
        "database_path": str(DB_PATH), "database_sha256": database_hash(),
        "datasets": len(names), "full_population": len(names) == 1117,
        "python": sys.version, "sktime_version": importlib.metadata.version("sktime"),
        "numpy_version": importlib.metadata.version("numpy"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "contains_test_labels": False, "retrospective": True,
    })


def run(args: argparse.Namespace) -> None:
    names = selected_names(args.dataset_limit, args.dataset_list_file)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    total = len(names)
    publish_external = total == 1117
    write_status(root, "running", args.stage, 0, total, publish_external=publish_external)
    if args.stage in ("audit", "all"):
        audit(root, names)
    if args.stage in ("cache", "all"):
        def progress(done: int, expected: int, current: str, errors: int) -> None:
            write_status(root, "running", "cache", done, expected, current, errors,
                         publish_external=publish_external)
            print(f"progress {done}/{expected} experiment=251 last={current} errors={errors}", flush=True)
        cache_all(root, workers=args.workers, names=names, reset=args.reset_cache, progress=progress)
    if args.stage in ("predict", "all"):
        write_status(root, "running", "predict", 0, total, publish_external=publish_external)
        predict_experiment(root, names=names)
        write_status(root, "running", "predict", total, total, publish_external=publish_external)
    if args.stage in ("evaluate", "all"):
        write_status(root, "running", "evaluate", 0, total, publish_external=publish_external)
        evaluate_experiment(root)
        result = reaggregate(root)
        if not result["passed"]:
            raise RuntimeError("independent reaggregation failed")
        write_report(root)
        write_manifest(root, " ".join(map(str, sys.argv)))
        write_status(root, "completed", "completed", total, total,
                     publish_external=publish_external)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--stage", choices=("audit", "cache", "predict", "evaluate", "all"), default="all")
    value.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    value.add_argument("--dataset-limit", type=int)
    value.add_argument("--dataset-list-file", type=Path)
    value.add_argument("--workers", type=int, default=7)
    value.add_argument("--reset-cache", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
