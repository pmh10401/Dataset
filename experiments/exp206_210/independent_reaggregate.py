from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp206_210.core import parse_indices, sha256_file


def bootstrap_paired_diff(
    left_values: np.ndarray, right_values: np.ndarray, iterations: int = 1000, seed: int = 20260716
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(left_values)
    if n == 0:
        return 0.0, 0.0
    diffs = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(np.mean(left_values[idx]) - np.mean(right_values[idx])))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def independently_reaggregate(
    *,
    output_root: Path,
    stage_files: dict[str, Path],
    result_path: Path,
    summary_path: Path,
    bootstrap_iterations: int = 1000,
) -> None:
    rows: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    per_dataset: dict[str, dict[str, set[int]]] = defaultdict(dict)

    # 1. Load results from all active prediction/evaluation stages (208 and 209)
    for stage, path in sorted(stage_files.items()):
        if not path.exists():
            raise FileNotFoundError(f"required stage result is missing: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            stage_rows = list(csv.DictReader(handle))
        for row in stage_rows:
            if int(row["candidate_count"]) != int(row["candidate_tp"]) + int(row["candidate_fp"]):
                raise ValueError(f"candidate arithmetic mismatch stage={stage} dataset={row['dataset_name']}")
            rows.append({"source_stage": stage, **row})
            per_dataset[row["dataset_name"]][f"{stage}:{row['variant']}"] = parse_indices(
                row["candidate_indices"]
            )

    # 2. Compute pairwise Jaccard candidate overlaps
    keys = sorted({key for stages in per_dataset.values() for key in stages})
    for dataset, stage_sets in sorted(per_dataset.items()):
        for left_pos, left in enumerate(keys):
            if left not in stage_sets:
                continue
            for right in keys[left_pos + 1 :]:
                if right not in stage_sets:
                    continue
                a, b = stage_sets[left], stage_sets[right]
                union = a | b
                overlaps.append(
                    {
                        "dataset_name": dataset,
                        "left": left,
                        "right": right,
                        "intersection": len(a & b),
                        "union": len(union),
                        "jaccard": len(a & b) / max(1, len(union)),
                        "left_unique": len(a - b),
                        "right_unique": len(b - a),
                    }
                )

    _write_csv(result_path, rows)
    overlap_path = output_root / "exp210_candidate_overlap.csv"
    _write_csv(overlap_path, overlaps)

    # 3. Stratified summaries with Bootstrapped F1 intervals
    summary_rows = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        scope = str(row["metric_scope"])
        grouped[(str(row["source_stage"]), str(row["variant"]), str(row["size_stratum"]), scope)].append(row)
        grouped[(str(row["source_stage"]), str(row["variant"]), "ALL", scope)].append(row)

    for (stage, variant, stratum, scope), group in sorted(grouped.items()):
        candidate_count = sum(int(row["candidate_count"]) for row in group)
        tp = sum(int(row["candidate_tp"]) for row in group)
        fp = sum(int(row["candidate_fp"]) for row in group)
        
        cf_f1s = np.array([float(row["counterfactual_f1"]) for row in group])
        base_f1s = np.array([float(row["baseline_hard_f1"]) for row in group])
        boot_low, boot_high = bootstrap_paired_diff(cf_f1s, base_f1s, iterations=bootstrap_iterations)

        workloads = np.array([int(row["candidate_count"]) for row in group], dtype=float)

        summary_rows.append(
            {
                "source_stage": stage,
                "variant": variant,
                "size_stratum": stratum,
                "metric_scope": scope,
                "datasets": len(group),
                "candidate_count": candidate_count,
                "candidate_tp": tp,
                "candidate_fp": fp,
                "candidate_precision": tp / max(1, candidate_count),
                "mean_candidate_recall": float(np.mean([float(row["candidate_recall"]) for row in group])),
                "mean_candidate_f1": float(np.mean([float(row["candidate_f1"]) for row in group])),
                "baseline_mean_hard_f1": float(np.mean(base_f1s)),
                "counterfactual_mean_f1": float(np.mean(cf_f1s)),
                "paired_f1_bootstrap_low": boot_low,
                "paired_f1_bootstrap_high": boot_high,
                "zero_f1_before": sum(int(row["zero_f1_before"]) for row in group),
                "zero_f1_after": sum(int(row["zero_f1_after"]) for row in group),
                "zero_f1_fixed": sum(int(row["zero_f1_fixed"]) for row in group),
                "zero_f1_new": sum(int(row["zero_f1_new"]) for row in group),
                "workload_median": float(np.median(workloads)) if len(workloads) > 0 else 0.0,
                "workload_p90": float(np.quantile(workloads, 0.90)) if len(workloads) > 0 else 0.0,
                "workload_p95": float(np.quantile(workloads, 0.95)) if len(workloads) > 0 else 0.0,
                "workload_max": int(np.max(workloads)) if len(workloads) > 0 else 0,
                "selected_from_test_metrics": 0,
                "retrospective_counterfactual": 1,
            }
        )
    _write_csv(summary_path, summary_rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(output_root: Path, experiment_id: str, files: list[Path], input_hashes: dict[str, str]) -> None:
    rows = [
        {
            "file": str(path),
            "sha256": sha256_file(path),
            "experiment_id": experiment_id,
            "retrospective_or_prospective": "retrospective",
            "autonomous_or_human_assisted": "mixed_shadow_review_research_diagnostic",
            "known_limitations": "not prospective; inherited feature provenance unresolved",
        }
        for path in files if path.exists()
    ]
    manifest_path = output_root / f"{experiment_id}_manifest.json"
    manifest_path.write_text(
        json.dumps({"files": rows, "input_hashes": input_hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
