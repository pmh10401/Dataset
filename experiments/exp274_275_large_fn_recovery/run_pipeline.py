from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .config import OUTPUT_ROOT
from .evaluation import evaluate_predictions
from .freeze import freeze_predictions


EXPERIMENT_IDS = {
    "274": "experiment_274_large_train_multisource_fn_recovery",
    "275": "experiment_275_recovery_shadow_precision_guard",
}


def _prediction_datasets(path: Path) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["dataset"]
        for line in path.read_text(encoding="utf-8").splitlines() if line
    )


def _publish(root: Path, stage: str, status: str, done: int, expected: int, started: float) -> None:
    experiment_id = EXPERIMENT_IDS[stage]
    log_path = root / "logs" / f"{experiment_id}.log"
    payload = build_status(
        experiment_id, status, stage, done, expected, None,
        errors=[], timestamps=[], stdout_log=log_path, started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp253 Hard unchanged",
        "review_metric_scope": "Exp274/275 review-only candidates",
    })
    publish_status(payload, output_root=root)


def run_stage(stage: str, *, output_root: Path, dataset_limit: int | None = None) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    expected = dataset_limit or 1117
    started = time.time()
    _publish(root, stage, "running", 0, expected, started)
    if stage == "274":
        freeze_predictions(
            output_root=root, expected_datasets=expected, dataset_limit=dataset_limit,
        )
    elif stage == "275":
        prediction = root / "exp274_frozen_predictions.jsonl"
        manifest = root / "exp274_prediction_manifest.json"
        datasets = _prediction_datasets(prediction)
        if len(datasets) != expected:
            raise ValueError("Exp274 prediction population does not match requested evaluation")
        evaluate_predictions(
            prediction, manifest, expected_datasets=datasets, output_root=root,
        )
    else:
        raise ValueError(f"unknown stage: {stage}")
    _publish(root, stage, "completed", expected, expected, started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp274-275 review-only recovery experiments")
    parser.add_argument("--stage", choices=tuple(EXPERIMENT_IDS), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Compatibility option; replay is deterministic")
    args = parser.parse_args()
    del args.workers
    run_stage(args.stage, output_root=args.output_root, dataset_limit=args.dataset_limit)


if __name__ == "__main__":
    main()

