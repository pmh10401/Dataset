from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp290_train_raw_exact_normal_guard.contracts import verify_prediction_rows

from .contracts import EXPERIMENT_ID, freeze_rows, sha256_file, verify_rows
from .evaluation import evaluate_rows
from .routing import route_dataset


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
EXP290_ROOT = DATA_ROOT / "exp290_train_raw_exact_normal_guard"
EXP272_ROOT = DATA_ROOT / "exp267_273_zero_support_fn_recovery"
OUTPUT_ROOT = DATA_ROOT / "exp291_final_operational_routing_freeze"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _load_exp272_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_path = EXP272_ROOT / "exp272_frozen_predictions.jsonl"
    manifest_path = EXP272_ROOT / "exp272_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("Exp272 prediction hash mismatch")
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1117 or any(row.get("contains_test_labels") is not False for row in rows):
        raise ValueError("Exp272 prediction contract mismatch")
    return rows, manifest


def run_prediction(root: Path) -> dict[str, object]:
    exp290_rows = verify_prediction_rows(
        EXP290_ROOT / "exp290_frozen_predictions.jsonl",
        EXP290_ROOT / "exp290_prediction_manifest.json",
    )
    exp272_rows, exp272_manifest = _load_exp272_rows()
    by_dataset_272 = {str(row["dataset"]): row for row in exp272_rows}
    if len(exp290_rows) != 1117 or len(by_dataset_272) != 1117:
        raise ValueError("input coverage mismatch")
    predictions = [route_dataset(row, by_dataset_272[str(row["dataset"])]) for row in exp290_rows]
    count_fields = {
        "automatic_hard_indices": 2218,
        "automatic_normal_exact_indices": 35,
        "standard_review_indices": 806,
        "priority_review_indices": 31,
        "shadow_review_indices": 2,
        "recovery_shadow_review_indices": 42,
        "research_only_recovery_priority_indices": 137,
    }
    for field, expected in count_fields.items():
        observed = sum(len(row[field]) for row in predictions)  # type: ignore[arg-type]
        if observed != expected:
            raise ValueError(f"final routing count mismatch: {field}={observed}/{expected}")
    manifest = freeze_rows(
        predictions,
        root / "exp291_frozen_operational_routing.jsonl",
        root / "exp291_prediction_manifest.json",
        metadata={
            "exp290_prediction_sha256": sha256_file(EXP290_ROOT / "exp290_frozen_predictions.jsonl"),
            "exp272_prediction_sha256": exp272_manifest["prediction_sha256"],
            "default_review_excludes_recovery_priority": True,
            "recovery_priority_scope": "research_only",
            "lane_counts": count_fields,
        },
    )
    return manifest


def run_evaluation(root: Path) -> dict[str, object]:
    prediction_path = root / "exp291_frozen_operational_routing.jsonl"
    prediction_rows = verify_rows(prediction_path, root / "exp291_prediction_manifest.json")
    if len(prediction_rows) != 1117:
        raise ValueError("Exp291 prediction coverage mismatch")
    labels = {
        str(row["dataset"]): load_test_labels_for_evaluation(str(row["dataset"]))
        for row in prediction_rows
    }
    evaluated, summary = evaluate_rows(prediction_rows, labels)
    expected = {
        "automatic_hard_count": 2218,
        "automatic_hard_tp": 1915,
        "automatic_hard_fp": 303,
        "automatic_normal_exact_count": 35,
        "automatic_normal_exact_tp": 0,
        "automatic_normal_exact_fp": 35,
    }
    for field, value in expected.items():
        if int(summary[field]) != value:
            raise ValueError(f"Exp291 metric parity mismatch: {field}")
    _write_csv(root / "exp291_results.csv", evaluated)
    _write_csv(root / "exp291_summary.csv", [summary])
    transitions = [
        {
            "dataset": row["dataset"],
            "automatic_hard_count": row["automatic_hard_count"],
            "automatic_normal_exact_count": row["automatic_normal_exact_count"],
            "default_review_count": row["default_review_count"],
            "research_only_recovery_priority_count": row["research_only_recovery_priority_count"],
            "lane_overlap": 0,
        }
        for row in evaluated
    ]
    _write_csv(root / "exp291_lane_counts.csv", transitions)
    (root / "exp291_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "prediction_sha256": sha256_file(prediction_path),
                "labels_used_after_prediction_freeze": True,
                "autonomous_and_review_metrics_separated": True,
                "rows": len(evaluated),
                "retrospective_integration": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _publish(root: Path, status: str, phase: str, done: int, started: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID, status, phase, done, 1117, None,
        errors=[], timestamps=[],
        stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log",
        started_unix=started,
    )
    payload.update(
        {
            "autonomous_metric_scope": "Exp290 guarded Hard",
            "review_metric_scope": "Exp272 approved review lanes; human-assisted only",
            "retrospective": True,
            "prospective": False,
        }
    )
    publish_status(payload, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.time()
    _publish(args.output_root, "running", args.phase, 0, started)
    if args.phase in ("predict", "all"):
        run_prediction(args.output_root)
    if args.phase in ("evaluate", "all"):
        run_evaluation(args.output_root)
    _publish(args.output_root, "completed", args.phase, 1117, started)


if __name__ == "__main__":
    main()
