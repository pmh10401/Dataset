#!/usr/bin/env python3
"""Read-only retrospective diagnosis of Exp202 alternative-source candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/Users/minho/Documents/timesries project")
DATA = Path("/Users/minho/Documents/Dataset")
OUT = Path(__file__).resolve().parent
ALPHA = 0.01
SOURCES = ("spectral", "shapelet", "interval")


def labels_for(connection: sqlite3.Connection, dataset: str) -> np.ndarray:
    rows = connection.execute(
        """
        SELECT CAST(i.label AS INTEGER)
        FROM instances i JOIN datasets d ON d.id=i.dataset_id
        WHERE d.name=? AND i.split='TEST'
        ORDER BY i.instance_index
        """,
        (dataset,),
    ).fetchall()
    return np.asarray([int(row[0]) for row in rows], dtype=np.int8)


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positive) * len(negative)))


def main() -> None:
    manifest = {
        row["dataset"]: row
        for row in csv.DictReader(
            (ROOT / "outputs/exp162_full_coverage_alternative_score_cache/02_cache_manifest.csv").open()
        )
    }
    prediction_path = DATA / "exp201_205_large_zero_f1_recovery/exp202_frozen_predictions.jsonl"
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    all_large = {
        row["dataset"]: row for row in predictions if row["variant"] == "alternative_3of3_all_large"
    }
    exp137_rows = {
        row["dataset_name"]: row
        for row in csv.DictReader(
            (DATA / "experiment_137_operational_triage_results.csv").open()
        )
    }
    details = []
    source_totals = defaultdict(Counter)
    dataset_auc = defaultdict(list)
    raw_agreement_totals = Counter()
    raw_agreement_by_stratum = defaultdict(Counter)
    overlap_lane_totals = Counter()
    overlap_by_lane = defaultdict(Counter)
    total_test_normal = 0
    with sqlite3.connect(DATA / "univariate_ts.db") as connection:
        for dataset, prediction in sorted(all_large.items()):
            labels = labels_for(connection, dataset)
            total_test_normal += int(np.sum(labels == 0))
            cache_path = ROOT / manifest[dataset]["file_path"]
            with np.load(cache_path, allow_pickle=False) as archive:
                thresholds = {}
                test_scores = {}
                source_sets = {}
                for source in SOURCES:
                    train = np.asarray(archive[f"{source}_train"], dtype=float)
                    test = np.asarray(archive[f"{source}_test"], dtype=float)
                    threshold = float(np.quantile(train[np.isfinite(train)], 1.0 - ALPHA, method="higher"))
                    thresholds[source] = threshold
                    test_scores[source] = test
                    selected = set(np.where(np.isfinite(test) & (test > threshold))[0].tolist())
                    source_sets[source] = selected
                    metrics = source_totals[source]
                    metrics["candidates"] += len(selected)
                    metrics["tp"] += sum(int(labels[index]) for index in selected)
                    metrics["fp"] += sum(1 - int(labels[index]) for index in selected)
                    auc = rank_auc(labels, test)
                    if np.isfinite(auc):
                        dataset_auc[source].append(auc)

                agreement = source_sets["spectral"] & source_sets["shapelet"] & source_sets["interval"]
                predicted = {int(index) for index in prediction["candidate_indices"]}
                baseline = exp137_rows[dataset]
                lane_sets = {
                    lane: {
                        int(token)
                        for token in baseline[key].replace(",", " ").split()
                        if token
                    }
                    for lane, key in (
                        ("hard", "hard_alert_indices"),
                        ("standard", "standard_review_indices"),
                        ("priority", "priority_review_indices"),
                    )
                }
                existing_lanes = set().union(*lane_sets.values())
                if predicted != agreement - existing_lanes:
                    raise ValueError(f"prediction/raw agreement mismatch dataset={dataset}")
                raw_agreement_totals["candidates"] += len(agreement)
                raw_agreement_totals["tp"] += sum(int(labels[index]) for index in agreement)
                raw_agreement_totals["fp"] += sum(1 - int(labels[index]) for index in agreement)
                stratum_totals = raw_agreement_by_stratum[prediction["size_stratum"]]
                stratum_totals["candidates"] += len(agreement)
                stratum_totals["tp"] += sum(int(labels[index]) for index in agreement)
                stratum_totals["fp"] += sum(1 - int(labels[index]) for index in agreement)
                overlap = agreement & existing_lanes
                overlap_lane_totals["candidates"] += len(overlap)
                overlap_lane_totals["tp"] += sum(int(labels[index]) for index in overlap)
                overlap_lane_totals["fp"] += sum(1 - int(labels[index]) for index in overlap)
                for lane, lane_indices in lane_sets.items():
                    lane_overlap = agreement & lane_indices
                    overlap_by_lane[lane]["candidates"] += len(lane_overlap)
                    overlap_by_lane[lane]["tp"] += sum(int(labels[index]) for index in lane_overlap)
                    overlap_by_lane[lane]["fp"] += sum(1 - int(labels[index]) for index in lane_overlap)
                anomaly_indices = np.where(labels == 1)[0]
                for index in sorted(predicted):
                    row = {
                        "dataset": dataset,
                        "size_stratum": prediction["size_stratum"],
                        "candidate_index": index,
                        "true_label": int(labels[index]),
                        "nearest_anomaly_distance": int(np.min(np.abs(anomaly_indices - index))) if len(anomaly_indices) else -1,
                        "n_test": len(labels),
                        "anomaly_indices": " ".join(map(str, anomaly_indices.tolist())),
                    }
                    for source in SOURCES:
                        threshold = thresholds[source]
                        score = float(test_scores[source][index])
                        row[f"{source}_score"] = score
                        row[f"{source}_threshold"] = threshold
                        row[f"{source}_ratio"] = score / threshold if threshold != 0 else float("inf")
                    details.append(row)

        duplicate_trace_groups = []
        for dataset, indices in {
            "StarLightCurves_normal_1": [444, 685, 882, 1052],
            "TwoPatterns_normal_3": [447, 452, 911],
            "ElectricDevices_normal_7": [263, 1197],
            "PhalangesOutlinesCorrect_normal_0": [433, 809],
        }.items():
            placeholders = ",".join("?" for _ in indices)
            rows = connection.execute(
                f"""
                SELECT i.instance_index, i.values_blob
                FROM instances i JOIN datasets d ON d.id=i.dataset_id
                WHERE d.name=? AND i.split='TEST'
                  AND i.instance_index IN ({placeholders})
                ORDER BY i.instance_index
                """,
                [dataset, *indices],
            ).fetchall()
            hashes = [hashlib.sha256(blob).hexdigest() for _, blob in rows]
            duplicate_trace_groups.append(
                {
                    "dataset": dataset,
                    "indices": [int(index) for index, _ in rows],
                    "sha256": hashes,
                    "all_values_blob_identical": len(set(hashes)) == 1,
                }
            )

    fields = list(details[0]) if details else []
    with (OUT / "exp202_candidate_diagnosis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)

    concentration = Counter(row["dataset"] for row in details)
    summary = {
        "agreement_candidates": len(details),
        "agreement_tp": sum(row["true_label"] for row in details),
        "agreement_fp": sum(1 - row["true_label"] for row in details),
        "active_datasets": len(concentration),
        "candidate_concentration": dict(concentration.most_common()),
        "raw_3of3_before_exp137_lane_exclusion": dict(raw_agreement_totals),
        "raw_3of3_by_size_stratum": {
            stratum: {
                **dict(counts),
                "precision": counts["tp"] / max(1, counts["candidates"]),
            }
            for stratum, counts in sorted(raw_agreement_by_stratum.items())
        },
        "total_test_normal_instances": total_test_normal,
        "raw_3of3_test_normal_false_alert_rate": (
            raw_agreement_totals["fp"] / max(1, total_test_normal)
        ),
        "raw_3of3_overlap_with_existing_exp137_lanes": dict(overlap_lane_totals),
        "raw_3of3_overlap_by_exp137_lane": {
            lane: dict(counts) for lane, counts in overlap_by_lane.items()
        },
        "nearest_anomaly_distance": {
            "min": min((row["nearest_anomaly_distance"] for row in details), default=None),
            "median": float(np.median([row["nearest_anomaly_distance"] for row in details])) if details else None,
            "max": max((row["nearest_anomaly_distance"] for row in details), default=None),
        },
        "individual_sources": {
            source: {
                **dict(source_totals[source]),
                "precision": source_totals[source]["tp"] / max(1, source_totals[source]["candidates"]),
                "mean_dataset_auc_high_score_is_anomaly": float(np.mean(dataset_auc[source])),
                "median_dataset_auc_high_score_is_anomaly": float(np.median(dataset_auc[source])),
                "datasets_auc_below_0_5": sum(value < 0.5 for value in dataset_auc[source]),
                "datasets_with_auc": len(dataset_auc[source]),
            }
            for source in SOURCES
        },
        "duplicate_trace_groups": duplicate_trace_groups,
        "retrospective_diagnostic_only": True,
        "must_not_select_policy_from_this_report": True,
    }
    (OUT / "exp202_candidate_diagnosis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
