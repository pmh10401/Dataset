from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from experiments.exp175_176 import diagnostic_core as core


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs/exp175_176_large_train_miss_diagnostic"


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reaggregate(output_dir: Path, enforce_expected: bool = True) -> dict[str, object]:
    output_dir = Path(output_dir)
    datasets = read_rows(output_dir / "04_exp175_dataset_diagnostic.csv")
    misses = read_rows(output_dir / "05_exp176_missed_anomaly_diagnostic.csv")
    cause_counts = Counter(row["cause_category"] for row in misses)
    unknown = set(cause_counts) - set(core.CAUSE_CATEGORIES)
    unique_misses = {(row["dataset"], int(row["candidate_index"])) for row in misses}
    result = {
        "datasets": len(datasets),
        "true_anomalies": sum(int(row["anomaly_count"]) for row in datasets),
        "hard_tp": sum(int(row["hard_tp"]) for row in datasets),
        "hard_fp": sum(int(row["hard_fp"]) for row in datasets),
        "hard_misses": len(misses),
        "dataset_hard_fn_sum": sum(int(row["hard_fn"]) for row in datasets),
        "unique_missed_indices": len(unique_misses),
        "cause_counts": {category: cause_counts[category] for category in core.CAUSE_CATEGORIES},
        "cause_count_sum": sum(cause_counts.values()),
        "unknown_cause_categories": sorted(unknown),
    }
    checks = {
        "dataset_fn_matches_anomaly_rows": result["dataset_hard_fn_sum"] == result["hard_misses"],
        "missed_indices_unique": result["unique_missed_indices"] == result["hard_misses"],
        "cause_sum_matches": result["cause_count_sum"] == result["hard_misses"],
        "cause_categories_known": not unknown,
        "anomaly_identity": result["hard_tp"] + result["hard_misses"] == result["true_anomalies"],
    }
    if enforce_expected:
        checks.update(
            {
                "expected_datasets": result["datasets"] == 639,
                "expected_anomalies": result["true_anomalies"] == 3440,
                "expected_hard_misses": result["hard_misses"] == 2107,
            }
        )
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    if not result["all_checks_passed"]:
        raise ValueError(f"independent reaggregation failed: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = reaggregate(args.output_dir)
    path = args.output_dir / "11_independent_reaggregation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
