from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp251_sktime_statistical_features.evaluation import VIEWS


def compare_summary(result_rows, summary_rows, *, views: tuple[str, ...] = VIEWS):
    grouped = defaultdict(list)
    for row in result_rows:
        grouped[(row["variant"], float(row["alpha"]))].append(row)
    indexed = {(row["variant"], float(row["alpha"])): row for row in summary_rows}
    mismatches = []
    for key, group in grouped.items():
        summary = indexed.get(key)
        if summary is None:
            mismatches.append({"variant": key[0], "alpha": key[1], "field": "row", "error": "missing"})
            continue
        for view in views:
            tp = sum(int(row[f"{view}_tp"]) for row in group)
            fp = sum(int(row[f"{view}_fp"]) for row in group)
            expected = {
                f"{view}_count": tp + fp, f"{view}_tp": tp, f"{view}_fp": fp,
                f"{view}_precision": tp / max(1, tp + fp),
                f"{view}_mean_f1": float(np.mean([float(row[f"{view}_f1"]) for row in group])),
            }
            for field, value in expected.items():
                actual = float(summary[field])
                if not np.isclose(actual, float(value), rtol=0.0, atol=1e-12):
                    mismatches.append({"variant": key[0], "alpha": key[1], "field": field,
                                       "expected": value, "actual": actual})
    if len(indexed) != len(grouped):
        mismatches.append({"field": "summary_rows", "expected": len(grouped), "actual": len(indexed)})
    return mismatches


def reaggregate(root: Path) -> dict[str, object]:
    with (root / "results.csv").open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    mismatches = compare_summary(results, summaries)
    payload = {"experiment": 251, "passed": not mismatches, "result_rows": len(results),
               "summary_rows": len(summaries), "mismatches": mismatches}
    (root / "independent_reaggregation.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload
