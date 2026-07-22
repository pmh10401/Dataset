from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp292_standard_review_evidence_tiering.contracts import freeze_prediction_rows, sha256_file, verify_prediction_rows

from .core import ALPHAS, EXPERIMENT_ID, PRIMARY_ALPHA, build_consensus_row, evaluate_predictions, summarize_evaluated


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp300_independent_evidence_consensus"
INPUTS = {
    "multirocket": (
        DATA_ROOT / "exp249_250_official_mini_multirocket" / "experiment_250_official_multirocket" / "frozen_predictions.jsonl",
        "b07ef8a3093446701229e82f2794f47da81bbfb7f855f6e9b36650d112cd0800",
        "kernels_1024_seed_intersection",
    ),
    "structural": (
        DATA_ROOT / "exp254_sktime_structural_features" / "frozen_predictions.jsonl",
        "fd796d0fcba5a0413790b0a17bea263fd9fb3ca5273e5c88a8746d476ae4d997",
        "structural_combined_440",
    ),
    "change_point": (
        DATA_ROOT / "exp255_change_point_features" / "frozen_predictions.jsonl",
        "de964382304df87a52bf99e589ddfd8df8c63eebe832fe92ec4a22b60cbdec4e",
        "change_point_combined_33",
    ),
}
BASELINE_PATH = DATA_ROOT / "exp298_exp201_current_hard_rebase" / "exp298_frozen_predictions.jsonl"
BASELINE_SHA256 = "9de5eb0cd3226b0734e26179e246ad3fa59b3d95c472d78b424af5e80b65a727"
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
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _source_map(path: Path, variant: str, required: set[str]) -> dict[tuple[str, float], dict[str, object]]:
    result = {}
    for row in _load_jsonl(path):
        dataset = str(row["dataset"])
        if dataset in required and str(row["variant"]) == variant and float(row["alpha"]) in ALPHAS:
            result[(dataset, float(row["alpha"]))] = row
    expected = EXPECTED_DATASETS * len(ALPHAS)
    if len(result) != expected:
        raise ValueError(f"source coverage mismatch variant={variant}: {len(result)} != {expected}")
    return result


def run_prediction(root: Path) -> dict[str, object]:
    if sha256_file(BASELINE_PATH) != BASELINE_SHA256:
        raise ValueError("Exp298 baseline hash mismatch")
    for name, (path, expected_hash, _) in INPUTS.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{name} input hash mismatch")
    baselines = _load_jsonl(BASELINE_PATH)
    if len(baselines) != EXPECTED_DATASETS or len({str(row["dataset"]) for row in baselines}) != EXPECTED_DATASETS:
        raise ValueError("large baseline coverage mismatch")
    required = {str(row["dataset"]) for row in baselines}
    sources = {name: _source_map(path, variant, required) for name, (path, _, variant) in INPUTS.items()}
    predictions: list[dict[str, object]] = []
    for baseline in baselines:
        dataset = str(baseline["dataset"])
        for alpha in ALPHAS:
            predictions.append(build_consensus_row(
                sources["multirocket"][(dataset, alpha)],
                sources["structural"][(dataset, alpha)],
                sources["change_point"][(dataset, alpha)],
                baseline,
            ))
    predictions.sort(key=lambda row: (str(row["dataset"]), float(row["alpha"])))
    return freeze_prediction_rows(
        predictions, root / "exp300_frozen_predictions.jsonl", root / "exp300_prediction_manifest.json",
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "input_sha256": {name: expected for name, (_, expected, _) in INPUTS.items()},
            "baseline_sha256": BASELINE_SHA256,
            "datasets": EXPECTED_DATASETS,
            "rows": EXPECTED_DATASETS * len(ALPHAS),
            "primary_alpha": PRIMARY_ALPHA,
            "consensus_rules": ["2of3", "3of3"],
            "autonomous_policy_changed": False,
        },
    )


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp300_frozen_predictions.jsonl"
    predictions = verify_prediction_rows(prediction_path, root / "exp300_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS * len(ALPHAS):
        raise ValueError("Exp300 prediction coverage mismatch")
    labels = {str(row["dataset"]): load_test_labels_for_evaluation(str(row["dataset"])) for row in predictions}
    evaluated = evaluate_predictions(predictions, labels)
    summaries = summarize_evaluated(evaluated)
    _write_csv(root / "exp300_results.csv", evaluated)
    _write_csv(root / "exp300_summary.csv", summaries)
    primary = next(row for row in summaries if row["alpha"] == PRIMARY_ALPHA and row["size_stratum"] == "100+")
    (root / "exp300_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True,
        "test_metrics_selected_policy": False,
        "retrospective": True,
        "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Exp300 Independent Evidence Consensus",
        "",
        "Frozen official MultiROCKET, structural-pattern and change-point evidence on 251 large datasets.",
        "Current Exp292 automatic Hard remains unchanged.",
        "",
        f"## Primary alpha {PRIMARY_ALPHA}",
        "",
        f"- 2-of-3 residual: {primary['residual_2of3_count']} (TP {primary['residual_2of3_tp']}, FP {primary['residual_2of3_fp']}, precision {float(primary['residual_2of3_precision']):.3%})",
        f"- 3-of-3 residual: {primary['residual_3of3_count']} (TP {primary['residual_3of3_tp']}, FP {primary['residual_3of3_fp']}, precision {float(primary['residual_3of3_precision']):.3%})",
        f"- 2-of-3 Mean F1: {float(primary['baseline_hard_mean_f1']):.6f} -> {float(primary['counterfactual_2of3_mean_f1']):.6f}",
        f"- 3-of-3 Mean F1: {float(primary['baseline_hard_mean_f1']):.6f} -> {float(primary['counterfactual_3of3_mean_f1']):.6f}",
        f"- Zero-F1 fixed: 2-of-3 {primary['zero_f1_fixed_2of3']}, 3-of-3 {primary['zero_f1_fixed_3of3']}",
        f"- 80% gates: 2-of-3 {primary['gate_2of3_80pct']}, 3-of-3 {primary['gate_3of3_80pct']}",
        "",
        "Other alphas are sensitivity outputs only. No source, alpha or consensus rule is selected from TEST metrics.",
    ]
    (root / "RESULT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summaries


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None, errors=errors or [], timestamps=[], stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log", started_unix=started)
    payload.update({"autonomous_metric_scope": "Exp292 automatic Hard unchanged", "review_metric_scope": "Independent 2-of-3/3-of-3 retrospective shadow", "retrospective": True, "prospective": False})
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
