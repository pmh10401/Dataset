from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import VIEWS


def reaggregate(output_root: Path) -> dict[str, object]:
    with (output_root / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (output_root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 1:
        raise ValueError("summary.csv must contain exactly one row")
    summary = summary_rows[0]
    mismatches: list[dict[str, object]] = []
    for view in VIEWS:
        tp = sum(int(row[f"{view}_tp"]) for row in rows)
        fp = sum(int(row[f"{view}_fp"]) for row in rows)
        fn = sum(int(row[f"{view}_fn"]) for row in rows)
        expected = {
            f"{view}_count": tp + fp,
            f"{view}_tp": tp,
            f"{view}_fp": fp,
            f"{view}_fn": fn,
            f"{view}_precision": tp / (tp + fp) if tp + fp else 0.0,
            f"{view}_recall": tp / (tp + fn) if tp + fn else 0.0,
            f"{view}_mean_f1": float(np.mean([float(row[f"{view}_f1"]) for row in rows])),
        }
        for field, value in expected.items():
            actual = float(summary[field])
            if not np.isclose(actual, float(value), rtol=0.0, atol=1e-12):
                mismatches.append({"field": field, "expected": value, "actual": actual})
    payload = {
        "passed": not mismatches,
        "datasets": len(rows),
        "summary_rows": len(summary_rows),
        "mismatches": mismatches,
    }
    (output_root / "independent_reaggregation.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return payload
