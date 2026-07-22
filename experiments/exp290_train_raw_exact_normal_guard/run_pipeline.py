from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .contracts import EXPERIMENT_ID, freeze_prediction_rows, sha256_file, verify_prediction_rows
from .evaluation import evaluate_rows
from .prediction import build_prediction_rows


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP253_ROOT = DATA_ROOT / "exp253_guarded_standard_to_hard_expansion"
OUTPUT_ROOT = DATA_ROOT / "exp290_train_raw_exact_normal_guard"


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


def load_exp253_predictions() -> tuple[list[dict[str, object]], dict[str, object]]:
    path = EXP253_ROOT / "frozen_predictions.jsonl"
    manifest_path = EXP253_ROOT / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest.get("prediction_sha256"):
        raise ValueError("Exp253 prediction hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1117:
        raise ValueError(f"Exp253 coverage mismatch: {len(rows)}/1117")
    if sum(len(row["final_hard_indices"]) for row in rows) != 2253:
        raise ValueError("Exp253 Hard count mismatch")
    if any(row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False for row in rows):
        raise ValueError("Exp253 prediction contains labels or evaluation metrics")
    return rows, manifest


def run_prediction(root: Path) -> dict[str, object]:
    baseline_rows, baseline_manifest = load_exp253_predictions()
    prediction_rows = build_prediction_rows(baseline_rows, db_path=DB_PATH)
    manifest = freeze_prediction_rows(
        prediction_rows,
        root / "exp290_frozen_predictions.jsonl",
        root / "exp290_prediction_manifest.json",
    )
    removed = sum(len(row["train_raw_exact_normal_indices"]) for row in prediction_rows)
    audit = {
        **manifest,
        "database_path": str(DB_PATH),
        "database_sha256": sha256_file(DB_PATH),
        "exp253_prediction_path": str(EXP253_ROOT / "frozen_predictions.jsonl"),
        "exp253_prediction_sha256": baseline_manifest["prediction_sha256"],
        "baseline_hard_count": 2253,
        "train_raw_exact_normal_count": removed,
        "guarded_hard_count": 2253 - removed,
        "prediction_sql_reads_test_labels": False,
    }
    (root / "exp290_prediction_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def run_evaluation(root: Path) -> dict[str, object]:
    prediction_path = root / "exp290_frozen_predictions.jsonl"
    manifest_path = root / "exp290_prediction_manifest.json"
    prediction_rows = verify_prediction_rows(prediction_path, manifest_path)
    if len(prediction_rows) != 1117:
        raise ValueError(f"Exp290 prediction coverage mismatch: {len(prediction_rows)}/1117")
    labels = {
        str(row["dataset"]): load_test_labels_for_evaluation(str(row["dataset"]))
        for row in prediction_rows
    }
    evaluated, summary = evaluate_rows(prediction_rows, labels)
    if int(summary["baseline_hard_count"]) != 2253:
        raise ValueError("Exp290 evaluation baseline Hard mismatch")
    _write_csv(root / "exp290_results.csv", evaluated)
    _write_csv(root / "exp290_summary.csv", [summary])

    removed_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for row in evaluated:
        removed = list(map(int, row["train_raw_exact_normal_indices"]))  # type: ignore[arg-type]
        transition_rows.append(
            {
                "dataset": row["dataset"],
                "baseline_hard_count": row["baseline_hard_count"],
                "guarded_hard_count": row["guarded_hard_count"],
                "train_raw_exact_normal_count": row["train_raw_exact_normal_count"],
                "removed_tp": row["train_raw_exact_normal_tp"],
                "removed_fp": row["train_raw_exact_normal_fp"],
                "retrospective_counterfactual": True,
            }
        )
        dataset_labels = labels[str(row["dataset"])]
        match_map = row["exact_train_indices_by_candidate"]  # type: ignore[assignment]
        for index in removed:
            removed_rows.append(
                {
                    "dataset": row["dataset"],
                    "candidate_index": index,
                    "matching_train_indices": match_map[str(index)],  # type: ignore[index]
                    "true_label": int(dataset_labels[index]),
                    "outcome": "TP_removed" if int(dataset_labels[index]) == 1 else "FP_removed",
                    "match_definition": "sqlite_values_blob_byte_exact",
                    "retrospective_counterfactual": True,
                }
            )
    _write_csv(root / "exp290_lane_transitions.csv", transition_rows)
    if removed_rows:
        _write_csv(root / "exp290_removed_exact_matches.csv", removed_rows)
    else:
        (root / "exp290_removed_exact_matches.csv").write_text(
            "dataset,candidate_index,matching_train_indices,true_label,outcome,match_definition,retrospective_counterfactual\n",
            encoding="utf-8",
        )
    evaluation_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_used_after_prediction_freeze": True,
        "rows": len(evaluated),
        "retrospective_counterfactual": True,
        "autonomous_metric_scope": "guarded_hard_indices",
    }
    (root / "exp290_evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _publish(root: Path, status: str, phase: str, done: int, started: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID,
        status,
        phase,
        done,
        1117,
        None,
        errors=[],
        timestamps=[],
        stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log",
        started_unix=started,
    )
    payload.update(
        {
            "autonomous_metric_scope": "Exp253 Hard with TRAIN raw exact normal guard",
            "review_metric_scope": "unchanged",
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
    root = args.output_root
    started = time.time()
    _publish(root, "running", args.phase, 0, started)
    if args.phase in ("predict", "all"):
        run_prediction(root)
    if args.phase in ("evaluate", "all"):
        run_evaluation(root)
    _publish(root, "completed", args.phase, 1117, started)


if __name__ == "__main__":
    main()
