from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .cache import build_distance_cache
from .evaluation import evaluate_predictions
from .prediction import freeze_predictions


OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp282_284_prototype_distance_verifier")
EXPERIMENT_IDS = {
    "282": "experiment_282_prototype_distance_cache",
    "283": "experiment_283_prototype_distance_prediction",
    "284": "experiment_284_prototype_distance_evaluation",
}


def _publish(root: Path, stage: str, status: str, done: int, started: float) -> None:
    experiment_id = EXPERIMENT_IDS[stage]
    payload = build_status(experiment_id, status, stage, done, 1117, None, errors=[], timestamps=[],
                           stdout_log=root / "logs" / f"{experiment_id}.log", started_unix=started)
    payload.update({"autonomous_metric_scope": "Exp253 Hard unchanged", "review_metric_scope": "Prototype-distance review verifier"})
    publish_status(payload, output_root=root)


def run_stage(stage: str, *, output_root: Path) -> None:
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(exist_ok=True)
    started = time.time(); _publish(root, stage, "running", 0, started)
    if stage == "282": build_distance_cache(root)
    elif stage == "283": freeze_predictions(root)
    elif stage == "284": evaluate_predictions(root / "exp283_frozen_predictions.jsonl", root / "exp283_prediction_manifest.json", output_root=root)
    else: raise ValueError(f"unknown stage: {stage}")
    _publish(root, stage, "completed", 1117, started)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=tuple(EXPERIMENT_IDS), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args()
    run_stage(args.stage, output_root=args.output_root)


if __name__ == "__main__": main()
