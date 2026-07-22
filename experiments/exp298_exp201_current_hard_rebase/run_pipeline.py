from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp292_standard_review_evidence_tiering.contracts import (
    freeze_prediction_rows,
    sha256_file,
    verify_prediction_rows,
)

from .core import EXPERIMENT_ID, evaluate_predictions, rebase_candidate_row, summarize_evaluated


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp298_exp201_current_hard_rebase"
EXP201_PATH = DATA_ROOT / "exp201_205_large_zero_f1_recovery" / "exp201_frozen_predictions.jsonl"
EXP292_PATH = DATA_ROOT / "exp292_standard_review_evidence_tiering" / "exp292_frozen_review_tiers.jsonl"
EXP201_SHA256 = "cd84e293d18784873ed839613cd407fc00c424e2a4cd9cab7a5cccd1490cc44f"
EXP292_SHA256 = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
EXPECTED_DATASETS = 251


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _validate_inputs() -> None:
    for path, expected, label in (
        (EXP201_PATH, EXP201_SHA256, "Exp201"),
        (EXP292_PATH, EXP292_SHA256, "Exp292"),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} input hash mismatch: {actual} != {expected}")


def run_prediction(root: Path) -> dict[str, object]:
    _validate_inputs()
    exp201 = _load_jsonl(EXP201_PATH)
    exp292 = {str(row["dataset"]): row for row in _load_jsonl(EXP292_PATH)}
    if len(exp201) != EXPECTED_DATASETS or len({str(row["dataset"]) for row in exp201}) != EXPECTED_DATASETS:
        raise ValueError("Exp201 coverage mismatch")
    if len(exp292) != 1117:
        raise ValueError("Exp292 coverage mismatch")
    predictions = [rebase_candidate_row(row, exp292[str(row["dataset"])]) for row in exp201]
    predictions.sort(key=lambda row: str(row["dataset"]))
    return freeze_prediction_rows(
        predictions,
        root / "exp298_frozen_predictions.jsonl",
        root / "exp298_prediction_manifest.json",
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "exp201_sha256": EXP201_SHA256,
            "exp292_sha256": EXP292_SHA256,
            "rows": EXPECTED_DATASETS,
            "autonomous_policy_changed": False,
            "retrospective_counterfactual": True,
            "candidate_rule_frozen_before_evaluation": True,
        },
    )


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp298_frozen_predictions.jsonl"
    manifest_path = root / "exp298_prediction_manifest.json"
    predictions = verify_prediction_rows(prediction_path, manifest_path)
    if len(predictions) != EXPECTED_DATASETS:
        raise ValueError("Exp298 prediction coverage mismatch")
    labels = {str(row["dataset"]): load_test_labels_for_evaluation(str(row["dataset"])) for row in predictions}
    evaluated = evaluate_predictions(predictions, labels)
    summaries = summarize_evaluated(evaluated)
    _write_csv(root / "exp298_results.csv", evaluated)
    _write_csv(root / "exp298_summary.csv", summaries)
    all_row = next(row for row in summaries if row["size_stratum"] == "100+")
    evaluation_manifest = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True,
        "datasets": EXPECTED_DATASETS,
        "autonomous_policy_changed": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    (root / "exp298_evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Exp298 Exp201 Candidate Rebase",
        "",
        "Frozen Exp201 candidates rebased against current Exp292 automatic Hard. Retrospective shadow evaluation only.",
        "",
        f"- Large datasets: {all_row['datasets']}",
        f"- Original Exp201 candidates: {all_row['source_exp201_candidates']}",
        f"- Already absorbed into current Hard: {all_row['already_in_current_hard']}",
        f"- Residual candidates: {all_row['residual_candidates']} (TP {all_row['residual_tp']}, FP {all_row['residual_fp']})",
        f"- Residual precision: {float(all_row['residual_precision']):.3%}",
        f"- Current Hard: TP {all_row['baseline_hard_tp']}, FP {all_row['baseline_hard_fp']}, mean F1 {float(all_row['baseline_mean_hard_f1']):.6f}",
        f"- Counterfactual union: TP {all_row['counterfactual_tp']}, FP {all_row['counterfactual_fp']}, mean F1 {float(all_row['counterfactual_mean_f1']):.6f}",
        f"- Zero-F1: {all_row['zero_f1_before']} -> {all_row['zero_f1_after']} (fixed {all_row['zero_f1_fixed']})",
        f"- Retrospective 80% point gate: {all_row['retrospective_80pct_point_gate']}",
        "",
        "No candidate is promoted to autonomous Hard by this experiment. TEST labels were loaded only after prediction freeze.",
    ]
    (root / "RESULT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summaries


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None,
        errors=errors or [], timestamps=[], stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log", started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp292 automatic Hard unchanged",
        "review_metric_scope": "Exp201 residual candidate retrospective shadow diagnostic",
        "retrospective": True,
        "prospective": False,
    })
    publish_status(payload, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.time()
    _publish(args.output_root, "running", args.phase, 0, started)
    try:
        if args.phase in ("predict", "all"):
            run_prediction(args.output_root)
        if args.phase in ("evaluate", "all"):
            run_evaluation(args.output_root)
    except Exception as exc:
        _publish(args.output_root, "failed", args.phase, 0, started, [f"{type(exc).__name__}: {exc}"])
        raise
    _publish(args.output_root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__":
    main()
