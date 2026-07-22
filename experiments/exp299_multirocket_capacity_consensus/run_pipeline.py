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
OUTPUT_ROOT = DATA_ROOT / "exp299_multirocket_capacity_consensus"
EXP250_PATH = DATA_ROOT / "exp249_250_official_mini_multirocket" / "experiment_250_official_multirocket" / "frozen_predictions.jsonl"
EXP298_PATH = DATA_ROOT / "exp298_exp201_current_hard_rebase" / "exp298_frozen_predictions.jsonl"
EXP250_SHA256 = "b07ef8a3093446701229e82f2794f47da81bbfb7f855f6e9b36650d112cd0800"
EXP298_SHA256 = "9de5eb0cd3226b0734e26179e246ad3fa59b3d95c472d78b424af5e80b65a727"
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


def run_prediction(root: Path) -> dict[str, object]:
    if sha256_file(EXP250_PATH) != EXP250_SHA256:
        raise ValueError("Exp250 input hash mismatch")
    if sha256_file(EXP298_PATH) != EXP298_SHA256:
        raise ValueError("Exp298 input hash mismatch")
    large = _load_jsonl(EXP298_PATH)
    if len(large) != EXPECTED_DATASETS or len({str(row["dataset"]) for row in large}) != EXPECTED_DATASETS:
        raise ValueError("Exp298 large-dataset coverage mismatch")
    required_datasets = {str(row["dataset"]) for row in large}
    source: dict[tuple[str, float, str], dict[str, object]] = {}
    for row in _load_jsonl(EXP250_PATH):
        dataset = str(row["dataset"])
        variant = str(row["variant"])
        alpha = float(row["alpha"])
        if dataset in required_datasets and variant in {"kernels_512_seed_intersection", "kernels_1024_seed_intersection"}:
            source[(dataset, alpha, variant)] = row
    expected_source_rows = EXPECTED_DATASETS * len(ALPHAS) * 2
    if len(source) != expected_source_rows:
        raise ValueError(f"Exp250 required source coverage mismatch: {len(source)} != {expected_source_rows}")
    predictions: list[dict[str, object]] = []
    for baseline in large:
        dataset = str(baseline["dataset"])
        for alpha in ALPHAS:
            predictions.append(build_consensus_row(
                source[(dataset, alpha, "kernels_512_seed_intersection")],
                source[(dataset, alpha, "kernels_1024_seed_intersection")],
                baseline,
            ))
    predictions.sort(key=lambda row: (str(row["dataset"]), float(row["alpha"])))
    return freeze_prediction_rows(
        predictions, root / "exp299_frozen_predictions.jsonl", root / "exp299_prediction_manifest.json",
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "exp250_sha256": EXP250_SHA256,
            "exp298_sha256": EXP298_SHA256,
            "rows": EXPECTED_DATASETS * len(ALPHAS),
            "datasets": EXPECTED_DATASETS,
            "primary_alpha": PRIMARY_ALPHA,
            "sensitivity_alphas": [alpha for alpha in ALPHAS if alpha != PRIMARY_ALPHA],
            "autonomous_policy_changed": False,
        },
    )


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp299_frozen_predictions.jsonl"
    predictions = verify_prediction_rows(prediction_path, root / "exp299_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS * len(ALPHAS):
        raise ValueError("Exp299 prediction row coverage mismatch")
    labels = {str(row["dataset"]): load_test_labels_for_evaluation(str(row["dataset"])) for row in predictions}
    evaluated = evaluate_predictions(predictions, labels)
    summaries = summarize_evaluated(evaluated)
    _write_csv(root / "exp299_results.csv", evaluated)
    _write_csv(root / "exp299_summary.csv", summaries)
    primary = next(row for row in summaries if row["alpha"] == PRIMARY_ALPHA and row["size_stratum"] == "100+")
    (root / "exp299_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True,
        "datasets": EXPECTED_DATASETS,
        "primary_alpha": PRIMARY_ALPHA,
        "test_metrics_selected_policy": False,
        "retrospective": True,
        "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Exp299 MultiROCKET Capacity Consensus",
        "",
        "Official MultiROCKET 512/1024 seed-intersection capacity blocks on the frozen 251 large-dataset population.",
        "Current Exp292 automatic Hard is unchanged; residual consensus is retrospective shadow evidence only.",
        "",
        f"## Primary alpha {PRIMARY_ALPHA}",
        "",
        f"- Residual consensus: {primary['consensus_residual_count']} (TP {primary['consensus_residual_tp']}, FP {primary['consensus_residual_fp']})",
        f"- Residual precision: {float(primary['consensus_residual_precision']):.3%}",
        f"- Counterfactual mean F1: {float(primary['baseline_hard_mean_f1']):.6f} -> {float(primary['counterfactual_mean_f1']):.6f}",
        f"- Zero-F1: {primary['zero_f1_before']} -> {primary['zero_f1_after']} (fixed {primary['zero_f1_fixed']})",
        f"- Retrospective 80% point gate: {primary['retrospective_80pct_point_gate']}",
        "",
        "Other alphas are sensitivity outputs only. No alpha or capacity is selected from TEST metrics.",
    ]
    (root / "RESULT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summaries


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None, errors=errors or [], timestamps=[], stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log", started_unix=started)
    payload.update({"autonomous_metric_scope": "Exp292 automatic Hard unchanged", "review_metric_scope": "MultiROCKET capacity consensus retrospective shadow", "retrospective": True, "prospective": False})
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
