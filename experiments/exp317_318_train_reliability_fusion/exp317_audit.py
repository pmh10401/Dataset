from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import ALPHAS, EXPERIMENT_IDS, SEED, SOURCE_GROUPS
from .contracts import freeze_jsonl
from .reliability import audit_source_reliability
from .source_loader import load_verified_sources


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (list, dict, tuple)) else value
                for key, value in row.items()
            })


def audit_one_source(
    dataset: str, source: str, alpha: float, train_scores,
) -> dict[str, object]:
    audit = audit_source_reliability(train_scores, alpha=float(alpha), seed=SEED)
    return {
        "experiment_id": EXPERIMENT_IDS[317],
        "dataset": str(dataset),
        "source": str(source),
        "method_group": SOURCE_GROUPS[str(source)],
        "alpha": float(alpha),
        "n_train_scores": int(len(np.asarray(train_scores).reshape(-1))),
        "available": bool(audit.available),
        "abstention_reason": str(audit.reason),
        "split_false_alert_counts": list(audit.split_false_alert_counts),
        "split_audit_counts": list(audit.split_audit_counts),
        "split_wilson_upper": list(audit.split_wilson_upper),
        "max_wilson_upper": audit.max_wilson_upper,
        "n_audit_min": int(audit.n_audit_min),
        "weight": float(audit.weight),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_family_name": False,
        "uses_dataset_name_for_policy": False,
        "uses_test_length_for_policy": False,
        "retrospective": True,
    }


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), float(row["alpha"]))].append(row)
    output = []
    for (source, alpha), selected in sorted(grouped.items()):
        weights = [float(row["weight"]) for row in selected if row["available"]]
        output.append({
            "experiment_id": EXPERIMENT_IDS[317],
            "source": source,
            "method_group": SOURCE_GROUPS[source],
            "alpha": alpha,
            "datasets": len(selected),
            "available_datasets": sum(bool(row["available"]) for row in selected),
            "abstained_datasets": sum(not bool(row["available"]) for row in selected),
            "mean_weight": float(np.mean(weights)) if weights else 0.0,
            "median_weight": float(np.median(weights)) if weights else 0.0,
            "min_weight": float(np.min(weights)) if weights else 0.0,
            "max_weight": float(np.max(weights)) if weights else 0.0,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        })
    return output


def run_exp317_audit(
    output_root: Path, dataset_limit: int | None = None, *,
    source_roots: dict[str, Path] | None = None,
    progress_callback=None,
) -> dict[str, object]:
    sources, source_hashes, frozen_datasets = load_verified_sources(source_roots)
    datasets = sorted(frozen_datasets)
    if dataset_limit is not None:
        datasets = datasets[: int(dataset_limit)]
    rows = []
    for position, dataset in enumerate(datasets, start=1):
        for source in sorted(sources):
            train_scores = sources[source][dataset].get("train_oof_scores", [])
            for alpha in ALPHAS:
                rows.append(audit_one_source(dataset, source, alpha, train_scores))
        if progress_callback is not None:
            progress_callback(position, len(datasets), dataset)
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
    manifest = freeze_jsonl(
        rows,
        root / "exp317_frozen_reliability.jsonl",
        root / "exp317_reliability_manifest.json",
        experiment_id=EXPERIMENT_IDS[317],
        metadata={
            "datasets": len(datasets),
            "full_source_coverage_validated": True,
            "full_source_dataset_count": len(frozen_datasets),
            "dataset_limit": dataset_limit,
            "source_prediction_sha256": source_hashes,
            "alphas": list(ALPHAS),
            "seed": SEED,
            "retrospective": True,
            "prospective_validated": False,
            "automatic_policy_changed": False,
        },
    )
    _write_csv(root / "exp317_reliability_audit.csv", rows)
    _write_csv(root / "exp317_summary.csv", _summarize(rows))
    return manifest
