from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .contracts import OUTPUT_ROOT, candidate_rows
from .evaluation import evaluate_verifier_predictions
from .event_audit import run_event_applicability_audit
from .risk_audit import run_risk_resolution_audit
from .stability import run_calibration_stability
from .verifier import freeze_verifier_predictions


EXPERIMENT_IDS = {
    "278": "experiment_278_train_risk_resolution_audit",
    "279": "experiment_279_train_calibration_stability",
    "280": "experiment_280_event_merge_applicability_audit",
    "281": "experiment_281_train_only_two_stage_verifier",
}


def _publish(root: Path, stage: str, status: str, done: int, started: float) -> None:
    experiment_id = EXPERIMENT_IDS[stage]
    payload = build_status(
        experiment_id, status, stage, done, 1117, None, errors=[], timestamps=[],
        stdout_log=root / "logs" / f"{experiment_id}.log", started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp253 Hard unchanged",
        "review_metric_scope": "TRAIN-only recovery validation; review-only",
    })
    publish_status(payload, output_root=root)


def run_stage(stage: str, *, output_root: Path) -> None:
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(exist_ok=True)
    started = time.time(); _publish(root, stage, "running", 0, started)
    if stage == "278":
        run_risk_resolution_audit(root)
    elif stage == "279":
        run_calibration_stability(root)
    elif stage == "280":
        run_event_applicability_audit(root)
    elif stage == "281":
        prediction, manifest = freeze_verifier_predictions(root)
        _, datasets = candidate_rows()
        evaluate_verifier_predictions(
            prediction, manifest, expected_datasets=datasets, output_root=root,
        )
    else:
        raise ValueError(f"unknown stage: {stage}")
    _publish(root, stage, "completed", 1117, started)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp278-281 TRAIN-only recovery validation")
    parser.add_argument("--stage", choices=tuple(EXPERIMENT_IDS), required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(); run_stage(args.stage, output_root=args.output_root)


if __name__ == "__main__":
    main()
