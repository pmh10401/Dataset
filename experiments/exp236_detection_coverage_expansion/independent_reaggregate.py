from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


INTEGER_FIELDS = (
    "datasets", "hard_alerts", "hard_tp", "hard_fp", "hard_fn",
    "standard_review_candidates", "standard_review_tp", "standard_review_fp",
    "priority_review_candidates", "priority_review_tp", "priority_review_fp",
    "new_hard_tp", "new_hard_fp", "new_standard_review_tp", "new_standard_review_fp",
)
FLOAT_FIELDS = (
    "hard_precision", "hard_recall", "mean_hard_f1",
    "standard_review_precision", "priority_review_precision",
)


def _integer(row: dict[str, str], field: str) -> int:
    return int(float(row.get(field, 0) or 0))


def reaggregate(rows: list[dict[str, str]]) -> dict[str, object]:
    hard_tp = sum(_integer(row, "hard_tp") for row in rows)
    hard_fp = sum(_integer(row, "hard_fp") for row in rows)
    hard_fn = sum(_integer(row, "hard_fn") for row in rows)
    standard_tp = sum(_integer(row, "standard_review_tp") for row in rows)
    standard_fp = sum(_integer(row, "standard_review_fp") for row in rows)
    priority_tp = sum(_integer(row, "priority_review_tp") for row in rows)
    priority_fp = sum(_integer(row, "priority_review_fp") for row in rows)
    return {
        "datasets": len(rows),
        "hard_alerts": hard_tp + hard_fp,
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_fn": hard_fn,
        "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
        "hard_recall": hard_tp / max(1, hard_tp + hard_fn),
        "mean_hard_f1": sum(float(row["hard_f1"]) for row in rows) / max(1, len(rows)),
        "standard_review_candidates": standard_tp + standard_fp,
        "standard_review_tp": standard_tp,
        "standard_review_fp": standard_fp,
        "standard_review_precision": standard_tp / max(1, standard_tp + standard_fp),
        "priority_review_candidates": priority_tp + priority_fp,
        "priority_review_tp": priority_tp,
        "priority_review_fp": priority_fp,
        "priority_review_precision": priority_tp / max(1, priority_tp + priority_fp),
        "new_hard_tp": sum(_integer(row, "new_hard_tp") for row in rows),
        "new_hard_fp": sum(_integer(row, "new_hard_fp") for row in rows),
        "new_standard_review_tp": sum(_integer(row, "new_standard_review_tp") for row in rows),
        "new_standard_review_fp": sum(_integer(row, "new_standard_review_fp") for row in rows),
    }


def verify(output_root: Path) -> dict[str, object]:
    with (output_root / "experiment_236_results.csv").open(newline="", encoding="utf-8") as handle:
        actual = reaggregate(list(csv.DictReader(handle)))
    with (output_root / "experiment_236_summary.csv").open(newline="", encoding="utf-8") as handle:
        stored = next(csv.DictReader(handle))
    errors: list[str] = []
    for field in INTEGER_FIELDS:
        if int(actual[field]) != _integer(stored, field):
            errors.append(field)
    for field in FLOAT_FIELDS:
        if abs(float(actual[field]) - float(stored[field])) > 1e-12:
            errors.append(field)
    report = {"passed": not errors, "errors": errors, "independent_summary": actual}
    (output_root / "exp236_independent_reaggregation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.output_root)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
