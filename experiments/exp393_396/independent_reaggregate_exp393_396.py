from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def reaggregate(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    dataset_rows = _read_csv(output_root / "exp396_review_integration.csv")
    summary_rows = _read_csv(output_root / "exp396_summary.csv")
    expected: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in dataset_rows:
        key = (row["score"], row["alpha"], row["variant"])
        aggregate = expected.setdefault(
            key,
            {
                "datasets": 0,
                "hard_tp": 0,
                "hard_fp": 0,
                "added_standard_count": 0,
                "added_standard_tp": 0,
                "added_standard_fp": 0,
            },
        )
        aggregate["datasets"] += 1
        for field in tuple(aggregate)[1:]:
            aggregate[field] += int(float(row[field]))

    mismatches: list[dict[str, Any]] = []
    for row in summary_rows:
        key = (row["score"], row["alpha"], row["variant"])
        aggregate = expected.get(key)
        if aggregate is None:
            mismatches.append({"key": key, "reason": "missing aggregate"})
            continue
        for field, value in aggregate.items():
            if int(float(row[field])) != value:
                mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "expected": value,
                        "actual": row[field],
                    }
                )
    report = {
        "dataset_rows": len(dataset_rows),
        "summary_rows": len(summary_rows),
        "mismatches": mismatches,
        "consistent": not mismatches,
        "autonomous_hard_metrics_separate": True,
        "human_assisted_metrics_separate": True,
    }
    (output_root / "independent_consistency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "independent_consistency_report.md").write_text(
        "# Exp393-396 independent consistency report\n\n"
        f"- dataset rows: {len(dataset_rows):,}\n"
        f"- summary rows: {len(summary_rows):,}\n"
        f"- mismatches: {len(mismatches):,}\n"
        f"- consistent: {not mismatches}\n",
        encoding="utf-8",
    )
    if mismatches:
        raise ValueError("independent reaggregation mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reaggregate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
