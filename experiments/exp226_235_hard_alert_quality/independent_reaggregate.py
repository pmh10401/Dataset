from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


INTEGER_FIELDS = (
    "datasets", "hard_alerts", "hard_tp", "hard_fp", "hard_fn",
    "hard_zero_f1_datasets", "review_candidates", "review_tp", "review_fp",
    "research_candidates", "research_tp", "research_fp",
)
FLOAT_FIELDS = (
    "hard_precision", "hard_recall", "mean_hard_f1",
    "review_precision", "research_precision",
)


def _integer(row: dict[str, object], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / max(1, denominator)


def reaggregate_result_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["experiment_id"]), str(row["variant"]))].append(row)

    output: list[dict[str, object]] = []
    for (experiment_id, variant), group in sorted(groups.items()):
        hard_tp = sum(_integer(row, "hard_tp") for row in group)
        hard_fp = sum(_integer(row, "hard_fp") for row in group)
        hard_fn = sum(_integer(row, "hard_fn") for row in group)
        review_tp = sum(_integer(row, "review_tp") for row in group)
        review_fp = sum(_integer(row, "review_fp") for row in group)
        research_tp = sum(_integer(row, "research_tp") for row in group)
        research_fp = sum(_integer(row, "research_fp") for row in group)
        output.append({
            "experiment_id": experiment_id,
            "variant": variant,
            "datasets": len(group),
            "hard_alerts": hard_tp + hard_fp,
            "hard_tp": hard_tp,
            "hard_fp": hard_fp,
            "hard_fn": hard_fn,
            "hard_precision": _ratio(hard_tp, hard_tp + hard_fp),
            "hard_recall": _ratio(hard_tp, hard_tp + hard_fn),
            "mean_hard_f1": sum(float(row.get("hard_f1", 0) or 0) for row in group) / len(group),
            "hard_zero_f1_datasets": sum(float(row.get("hard_f1", 0) or 0) == 0 for row in group),
            "review_candidates": review_tp + review_fp,
            "review_tp": review_tp,
            "review_fp": review_fp,
            "review_precision": _ratio(review_tp, review_tp + review_fp),
            "research_candidates": research_tp + research_fp,
            "research_tp": research_tp,
            "research_fp": research_fp,
            "research_precision": _ratio(research_tp, research_tp + research_fp),
        })
    return output


def verify_experiment(output_root: Path, experiment: int, tolerance: float = 1e-12) -> dict[str, object]:
    result_path = output_root / "results" / f"experiment_{experiment}_results.csv"
    summary_path = output_root / "results" / f"experiment_{experiment}_summary.csv"
    with result_path.open(newline="", encoding="utf-8") as handle:
        independent = reaggregate_result_rows(list(csv.DictReader(handle)))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        stored = list(csv.DictReader(handle))
    expected = {(row["experiment_id"], row["variant"]): row for row in independent}
    actual = {(row["experiment_id"], row["variant"]): row for row in stored}
    errors: list[str] = []
    if set(expected) != set(actual):
        errors.append("summary key mismatch")
    for key in sorted(set(expected) & set(actual)):
        for field in INTEGER_FIELDS:
            if int(expected[key][field]) != _integer(actual[key], field):
                errors.append(f"{key}:{field}")
        for field in FLOAT_FIELDS:
            if abs(float(expected[key][field]) - float(actual[key][field])) > tolerance:
                errors.append(f"{key}:{field}")
    return {
        "experiment": experiment,
        "variants": len(independent),
        "passed": not errors,
        "errors": errors,
        "result_file": str(result_path),
        "summary_file": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiments", default="226-235")
    args = parser.parse_args()
    left, right = (int(value) for value in args.experiments.split("-", 1))
    reports = [verify_experiment(args.output_root, number) for number in range(left, right + 1)]
    report = {"passed": all(item["passed"] for item in reports), "experiments": reports}
    destination = args.output_root / "independent_reaggregation.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
