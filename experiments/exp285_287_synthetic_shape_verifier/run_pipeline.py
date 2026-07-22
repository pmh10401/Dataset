from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .evaluation import evaluate_predictions
from .prediction import freeze_predictions, generation_audit


OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp285_287_synthetic_shape_verifier")
IDS = {"285": "experiment_285_synthetic_shape_generation_audit", "286": "experiment_286_synthetic_shape_prediction", "287": "experiment_287_synthetic_shape_evaluation"}


def run_stage(stage: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(exist_ok=True); started = time.time(); experiment_id = IDS[stage]
    def publish(status, done):
        payload = build_status(experiment_id, status, stage, done, 1117, None, errors=[], timestamps=[], stdout_log=root / "logs" / f"{experiment_id}.log", started_unix=started)
        payload.update({"autonomous_metric_scope": "Exp253 Hard unchanged", "review_metric_scope": "Synthetic-shape review verifier"}); publish_status(payload, output_root=root)
    publish("running", 0)
    if stage == "285": generation_audit(root)
    elif stage == "286": freeze_predictions(root)
    elif stage == "287": evaluate_predictions(root / "exp286_frozen_predictions.jsonl", root / "exp286_prediction_manifest.json", output_root=root)
    publish("completed", 1117)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=tuple(IDS), required=True); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(); run_stage(args.stage, args.output_root)


if __name__ == "__main__": main()
