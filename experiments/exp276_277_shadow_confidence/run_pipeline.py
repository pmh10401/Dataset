from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .evaluation import evaluate_confidence_predictions
from .freeze import freeze_confidence_predictions


OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp276_277_shadow_confidence")
EXPERIMENT_IDS = {
    "276": "experiment_276_recovery_shadow_confidence_freeze",
    "277": "experiment_277_recovery_shadow_confidence_evaluation",
}


def _publish(root: Path, stage: str, status: str, done: int, expected: int, started: float) -> None:
    experiment_id = EXPERIMENT_IDS[stage]
    payload = build_status(
        experiment_id, status, stage, done, expected, None,
        errors=[], timestamps=[], stdout_log=root / "logs" / f"{experiment_id}.log",
        started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp253 Hard unchanged",
        "review_metric_scope": "Exp272 Recovery Shadow confidence only",
    })
    publish_status(payload, output_root=root)


def run_stage(stage: str, *, output_root: Path, dataset_limit: int | None = None) -> None:
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(exist_ok=True)
    expected = dataset_limit or 1117
    started = time.time(); _publish(root, stage, "running", 0, expected, started)
    if stage == "276":
        freeze_confidence_predictions(output_root=root, expected_datasets=expected, dataset_limit=dataset_limit)
    elif stage == "277":
        prediction = root / "exp276_frozen_confidence.jsonl"
        manifest = root / "exp276_prediction_manifest.json"
        datasets = tuple(json.loads(line)["dataset"] for line in prediction.read_text().splitlines() if line)
        if len(datasets) != expected:
            raise ValueError("Exp276 prediction population does not match evaluation")
        evaluate_confidence_predictions(prediction, manifest, expected_datasets=datasets, output_root=root)
    else:
        raise ValueError(f"unknown stage: {stage}")
    _publish(root, stage, "completed", expected, expected, started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp276-277 review confidence experiments")
    parser.add_argument("--stage", choices=tuple(EXPERIMENT_IDS), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    args = parser.parse_args()
    run_stage(args.stage, output_root=args.output_root, dataset_limit=args.dataset_limit)


if __name__ == "__main__":
    main()
