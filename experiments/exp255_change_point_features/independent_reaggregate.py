from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation import VIEWS


def reaggregate(root: Path) -> dict[str, object]:
    results = list(csv.DictReader((root / "results.csv").open(newline="", encoding="utf-8")))
    summaries = list(csv.DictReader((root / "summary.csv").open(newline="", encoding="utf-8")))
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in results:
        grouped[(row["variant"], float(row["alpha"]))].append(row)
    indexed = {(row["variant"], float(row["alpha"])): row for row in summaries}
    mismatches = []
    for key, rows in grouped.items():
        summary = indexed.get(key)
        if summary is None:
            mismatches.append({"variant": key[0], "alpha": key[1], "field": "missing_summary"})
            continue
        for view in VIEWS:
            tp = sum(int(row[f"{view}_tp"]) for row in rows)
            fp = sum(int(row[f"{view}_fp"]) for row in rows)
            expected = {
                f"{view}_count": tp + fp,
                f"{view}_tp": tp,
                f"{view}_fp": fp,
                f"{view}_precision": tp / (tp + fp) if tp + fp else 0.0,
                f"{view}_mean_f1": float(np.mean([float(row[f"{view}_f1"]) for row in rows])),
            }
            for field, value in expected.items():
                if not np.isclose(float(summary[field]), float(value), rtol=0, atol=1e-12):
                    mismatches.append(
                        {
                            "variant": key[0],
                            "alpha": key[1],
                            "field": field,
                            "expected": value,
                            "actual": summary[field],
                        }
                    )
    payload = {
        "passed": not mismatches,
        "result_rows": len(results),
        "summary_rows": len(summaries),
        "mismatches": mismatches,
    }
    (root / "independent_reaggregation.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n"
    )
    return payload

