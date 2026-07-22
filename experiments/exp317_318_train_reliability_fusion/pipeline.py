from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .config import EXPERIMENT_IDS, OUTPUT_ROOTS
from .evaluation import evaluate_exp318
from .exp317_audit import run_exp317_audit
from .exp318_prediction import run_exp318_prediction
from .independent_reaggregate import independent_reaggregate


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _status_writer(root: Path, experiment_id: str, phase: str, started: float):
    def publish(done: int, total: int, dataset: str | None = None, *, status: str = "running") -> None:
        elapsed = max(time.time() - started, 1e-9)
        rate = done / elapsed * 60.0 if done else None
        eta = (total - done) / rate * 60.0 if rate else None
        _atomic_json(root / "status.json", {
            "id": experiment_id,
            "experiment_id": experiment_id,
            "status": status,
            "phase": phase,
            "datasets_done": int(done),
            "datasets_total": int(total),
            "progress_percent": 100.0 * done / total if total else 0.0,
            "current_dataset": dataset,
            "elapsed_seconds": elapsed,
            "recent_rate_per_minute": rate,
            "eta_seconds": eta,
            "updated_at_unix": time.time(),
            "automatic_policy_changed": False,
            "research_shadow_only": True,
        })
    return publish


def run_experiment(
    experiment: int, output_root: Path, dataset_limit: int | None = None, *,
    reliability_root: Path | None = None,
) -> None:
    root = Path(output_root)
    started = time.time()
    total = int(dataset_limit or 1117)
    publish = _status_writer(root, EXPERIMENT_IDS[experiment], "audit" if experiment == 317 else "predict", started)
    publish(0, total)
    try:
        if experiment == 317:
            run_exp317_audit(root, dataset_limit, progress_callback=publish)
        elif experiment == 318:
            run_exp318_prediction(
                root, reliability_root or OUTPUT_ROOTS[317], dataset_limit,
                progress_callback=publish,
            )
            publish(total, total, status="evaluating")
            evaluate_exp318(root)
            independent_reaggregate(root)
        else:
            raise ValueError(f"unsupported experiment: {experiment}")
        publish(total, total, status="completed")
    except Exception as exc:
        publish(0, total, status="failed")
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))
        status["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_json(root / "status.json", status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=(317, 318), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--reliability-root", type=Path)
    args = parser.parse_args()
    run_experiment(
        args.experiment,
        args.output_root or OUTPUT_ROOTS[args.experiment],
        args.dataset_limit,
        reliability_root=args.reliability_root,
    )


if __name__ == "__main__":
    main()
