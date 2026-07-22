from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.exp237_new_score_source_qualification.config import OUTPUT_ROOT


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    with (Path(output_root) / "experiment_237_results.csv").open(newline="", encoding="utf-8") as handle:
        results = list(csv.DictReader(handle))
    with (Path(output_root) / "experiment_237_summary.csv").open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    groups: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in results:
        groups.setdefault((row["source"], float(row["alpha"])), []).append(row)
    mismatches = []
    for summary in summaries:
        key = (summary["source"], float(summary["alpha"]))
        group = groups.get(key, [])
        calculated = {
            "datasets": len(group),
            "candidates": sum(int(row["candidate_count"]) for row in group),
            "tp": sum(int(row["tp"]) for row in group),
            "fp": sum(int(row["fp"]) for row in group),
            "mean_f1": float(np.mean([float(row["f1"]) for row in group])) if group else 0.0,
        }
        for field, value in calculated.items():
            expected = float(summary[field]) if field == "mean_f1" else int(summary[field])
            if not np.isclose(value, expected, atol=1e-12, rtol=0.0):
                mismatches.append({"source": key[0], "alpha": key[1], "field": field})
    report = {
        "passed": not mismatches,
        "result_rows": len(results),
        "summary_rows": len(summaries),
        "datasets": len({row["dataset"] for row in results}),
        "mismatches": mismatches,
    }
    path = Path(output_root) / "exp237_independent_reaggregation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mismatches:
        raise ValueError(f"Exp237 independent reaggregation mismatch: {len(mismatches)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
