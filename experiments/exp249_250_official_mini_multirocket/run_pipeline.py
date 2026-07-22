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
from experiments.exp249_250_official_mini_multirocket.cache_pipeline import (
    cache_experiment,
    dataset_names,
    source_config,
    source_root,
)
from experiments.exp249_250_official_mini_multirocket.config import (
    DB_PATH,
    EXTERNAL_STATUS_PATH,
    HEARTBEAT_PATH,
    MODEL_CONFIGS,
    OUTPUT_ROOT,
)
from experiments.exp249_250_official_mini_multirocket.evaluation import evaluate_experiment
from experiments.exp249_250_official_mini_multirocket.independent_reaggregate import reaggregate
from experiments.exp249_250_official_mini_multirocket.prediction import predict_experiment
from experiments.exp249_250_official_mini_multirocket.reporting import write_manifest, write_report


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
    experiment: int,
    status: str,
    phase: str,
    done: int,
    total: int,
    current: str | None = None,
    errors: int = 0,
) -> None:
    config = MODEL_CONFIGS[int(experiment)]
    now = time.time()
    payload = {
        "id": f"experiment_{experiment}_{config.source}",
        "experiment_id": f"experiment_{experiment}_{config.source}",
        "label": f"Exp{experiment} - {config.source.replace('_', ' ')}",
        "source": config.source,
        "status": status,
        "phase": phase,
        "pid": os.getpid(),
        "child_pid": os.getpid(),
        "done": int(done),
        "datasets_done": int(done),
        "expected": int(total),
        "expected_datasets": int(total),
        "progress_percent": 100.0 * done / max(1, total),
        "current_dataset": current,
        "errors": int(errors),
        "updated_unix": now,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "retrospective_score_source_qualification",
        "comparison_eligible": False,
        "contains_test_labels": phase == "completed",
    }
    _atomic_json(source_root(root, experiment) / "status.json", payload)
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


def audit(root: Path, experiment: int, names: tuple[str, ...]) -> None:
    payload = {
        "experiment": int(experiment),
        "config": source_config(experiment),
        "database_path": str(DB_PATH),
        "database_sha256": database_hash(),
        "datasets": len(names),
        "full_population": len(names) == 1117,
        "python": sys.version,
        "aeon_version": importlib.metadata.version("aeon"),
        "numpy_version": importlib.metadata.version("numpy"),
        "scikit_learn_version": importlib.metadata.version("scikit-learn"),
        "contains_test_labels": False,
        "retrospective": True,
    }
    base = source_root(root, experiment)
    base.mkdir(parents=True, exist_ok=True)
    _atomic_json(base / "audit.json", payload)


def run(args: argparse.Namespace) -> None:
    experiment = int(args.experiment)
    if experiment not in MODEL_CONFIGS:
        raise ValueError(f"unsupported experiment: {experiment}")
    names = selected_names(args.dataset_limit, args.dataset_list_file)
    root = Path(args.output_root)
    total = len(names)
    write_status(root, experiment, "running", args.stage, 0, total)
    if args.stage in ("audit", "all"):
        audit(root, experiment, names)
    if args.stage in ("cache", "all"):
        def progress(done: int, expected: int, current: str, errors: int) -> None:
            write_status(root, experiment, "running", "cache", done, expected, current, errors)
            print(
                f"progress {done}/{expected} experiment={experiment} "
                f"last={current} errors={errors}",
                flush=True,
            )

        cache_experiment(
            root,
            experiment,
            workers=args.workers,
            names=names,
            reset=args.reset_cache,
            progress=progress,
        )
    if args.stage in ("predict", "all"):
        write_status(root, experiment, "running", "predict", 0, total)
        predict_experiment(root, experiment, names=names)
        write_status(root, experiment, "running", "predict", total, total)
    if args.stage in ("evaluate", "all"):
        write_status(root, experiment, "running", "evaluate", 0, total)
        evaluate_experiment(root, experiment)
        result = reaggregate(root, experiment)
        if not result["passed"]:
            raise RuntimeError("independent reaggregation failed")
        write_report(root, experiment)
        write_manifest(root, experiment, " ".join(map(str, sys.argv)))
        write_status(root, experiment, "completed", "completed", total, total)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--experiment", type=int, choices=tuple(MODEL_CONFIGS), required=True)
    value.add_argument("--stage", choices=("audit", "cache", "predict", "evaluate", "all"), default="all")
    value.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    value.add_argument("--dataset-limit", type=int)
    value.add_argument("--dataset-list-file", type=Path)
    value.add_argument("--workers", type=int, default=7)
    value.add_argument("--reset-cache", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
