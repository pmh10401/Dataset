from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status

from .config import EXPECTED_DATASETS, EXP291_PATH, EXP291_SHA256, EXPERIMENT_ID, OUTPUT_ROOT, SOURCE_SPECS
from .contracts import freeze_prediction_rows, sha256_file, verify_prediction_rows
from .evaluation import evaluate_review_tiers
from .routing import tier_standard_review
from .sources import load_all_source_support


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


def run_prediction(root: Path) -> dict[str, object]:
    if sha256_file(EXP291_PATH) != EXP291_SHA256:
        raise ValueError("Exp291 input hash mismatch")
    exp291_rows = _load_jsonl(EXP291_PATH)
    if len(exp291_rows) != EXPECTED_DATASETS or len({str(row["dataset"]) for row in exp291_rows}) != EXPECTED_DATASETS:
        raise ValueError("Exp291 dataset coverage mismatch")
    all_support, source_manifests = load_all_source_support()
    predictions: list[dict[str, object]] = []
    source_names = [spec.name for spec in SOURCE_SPECS]
    for baseline in exp291_rows:
        dataset = str(baseline["dataset"])
        family_support = {
            source: all_support[source].get(dataset) if dataset in all_support[source] else None
            for source in source_names
        }
        predictions.append(tier_standard_review(baseline, family_support))
    tier_counts = {
        "review_priority_1": sum(len(row["review_priority_1_indices"]) for row in predictions),
        "review_priority_2": sum(len(row["review_priority_2_indices"]) for row in predictions),
        "review_priority_3": sum(len(row["review_priority_3_indices"]) for row in predictions),
    }
    expected = {"review_priority_1": 17, "review_priority_2": 57, "review_priority_3": 732}
    if tier_counts != expected:
        raise ValueError(f"frozen review tier count mismatch: {tier_counts} != {expected}")
    if sum(tier_counts.values()) != 806:
        raise ValueError("Standard review partition count mismatch")
    return freeze_prediction_rows(
        predictions,
        root / "exp292_frozen_review_tiers.jsonl",
        root / "exp292_prediction_manifest.json",
        metadata={
            "exp291_sha256": EXP291_SHA256,
            "source_manifests": source_manifests,
            "tier_counts": tier_counts,
            "alpha": 0.005,
            "automatic_policy_changed": False,
            "review_tier_policy_frozen_before_evaluation": True,
        },
    )


def _cumulative_metrics(evaluated: list[dict[str, object]], tiers: tuple[str, ...]) -> dict[str, object]:
    count = sum(sum(int(row[f"{tier}_count"]) for tier in tiers) for row in evaluated)
    tp = sum(sum(int(row[f"{tier}_tp"]) for tier in tiers) for row in evaluated)
    fp = sum(sum(int(row[f"{tier}_fp"]) for tier in tiers) for row in evaluated)
    return {"tiers": "+".join(tiers), "count": count, "tp": tp, "fp": fp, "precision": tp / count if count else 0.0}


def run_evaluation(root: Path) -> dict[str, object]:
    prediction_path = root / "exp292_frozen_review_tiers.jsonl"
    predictions = verify_prediction_rows(prediction_path, root / "exp292_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS:
        raise ValueError("Exp292 prediction coverage mismatch")
    labels = {str(row["dataset"]): load_test_labels_for_evaluation(str(row["dataset"])) for row in predictions}
    evaluated, summary = evaluate_review_tiers(predictions, labels)
    if (int(summary["automatic_hard_count"]), int(summary["automatic_hard_tp"]), int(summary["automatic_hard_fp"])) != (2218, 1915, 303):
        raise ValueError("Exp291 automatic Hard parity mismatch")
    cumulative = [
        _cumulative_metrics(evaluated, ("review_priority_1",)),
        _cumulative_metrics(evaluated, ("review_priority_1", "review_priority_2")),
        _cumulative_metrics(evaluated, ("review_priority_1", "review_priority_2", "review_priority_3")),
    ]
    _write_csv(root / "exp292_results.csv", evaluated)
    _write_csv(root / "exp292_summary.csv", [summary])
    _write_csv(root / "exp292_tier_counts.csv", cumulative)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True,
        "review_metrics_human_assisted_only": True,
        "automatic_policy_changed": False,
        "retrospective": True,
        "rows": len(evaluated),
    }
    (root / "exp292_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Exp292 Standard Review Evidence Tiering",
        "",
        "Retrospective human-assisted review ordering only. Exp291 automatic Hard is unchanged.",
        "",
        f"- Automatic Hard: {summary['automatic_hard_count']} (TP {summary['automatic_hard_tp']}, FP {summary['automatic_hard_fp']})",
    ]
    for tier in ("review_priority_1", "review_priority_2", "review_priority_3"):
        report.append(f"- {tier}: {summary[tier + '_count']} (TP {summary[tier + '_tp']}, FP {summary[tier + '_fp']}, precision {float(summary[tier + '_precision']):.3%})")
    report.extend(["", "No tier is promoted to autonomous Hard. TEST labels were used only after the prediction hash was frozen."])
    (root / "RESULT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def _publish(root: Path, status: str, phase: str, done: int, started: float, *, errors: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None,
        errors=errors or [], timestamps=[], stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log", started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp291 Hard unchanged",
        "review_metric_scope": "Standard review P1/P2/P3 evidence ordering; human-assisted only",
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
        _publish(args.output_root, "failed", args.phase, 0, started, errors=[f"{type(exc).__name__}: {exc}"])
        raise
    _publish(args.output_root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__":
    main()

