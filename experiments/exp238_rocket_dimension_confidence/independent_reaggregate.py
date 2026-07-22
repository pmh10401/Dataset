from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp226_235_hard_alert_quality.database import (
    load_test_labels_for_evaluation,
)


FIELDS = (
    "hard",
    "confidence_a",
    "confidence_b",
    "confidence_c",
    "review_priority_1",
    "review_priority_2",
    "review_priority_3",
    "rocket_256",
    "rocket_512",
    "rocket_1024",
    "rocket_256_seed2",
    "rocket_stable",
)


def main() -> None:
    root = Path("/Users/minho/Documents/Dataset/exp238_rocket_dimension_confidence")
    predictions = [
        json.loads(line)
        for line in (root / "exp238_frozen_predictions.jsonl").read_text().splitlines()
    ]
    with (root / "experiment_238_summary.csv").open(newline="") as handle:
        reported = {row["alpha"]: row for row in csv.DictReader(handle)}

    violations: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, object]] = {}
    for row in predictions:
        alpha = str(row["alpha"])
        bucket = aggregate.setdefault(
            alpha,
            {"datasets": 0, "abstained_datasets": 0, "metrics": {}},
        )
        bucket["datasets"] = int(bucket["datasets"]) + 1
        bucket["abstained_datasets"] = int(bucket["abstained_datasets"]) + int(
            row["calibration_abstained"]
        )

        hard = set(row["hard_indices"])
        hard_parts = [set(row[f"confidence_{grade}_indices"]) for grade in "abc"]
        standard = set(row["standard_indices"])
        review_parts = [
            set(row[f"review_priority_{priority}_indices"]) for priority in (1, 2, 3)
        ]
        if set.union(*hard_parts) != hard or any(
            left & right
            for index, left in enumerate(hard_parts)
            for right in hard_parts[index + 1 :]
        ):
            violations.append({"dataset": row["dataset"], "alpha": alpha, "type": "hard_partition"})
        if set.union(*review_parts) != standard or any(
            left & right
            for index, left in enumerate(review_parts)
            for right in review_parts[index + 1 :]
        ):
            violations.append({"dataset": row["dataset"], "alpha": alpha, "type": "review_partition"})
        if any(row[f"{lane}_lane_changed"] for lane in ("hard", "standard", "priority")):
            violations.append({"dataset": row["dataset"], "alpha": alpha, "type": "lane_changed"})

        truth = set(
            np.flatnonzero(load_test_labels_for_evaluation(row["dataset"]) == 1)
            .astype(int)
            .tolist()
        )
        metrics = bucket["metrics"]
        assert isinstance(metrics, dict)
        for field in FIELDS:
            indices = set(row[f"{field}_indices"])
            if any(index < 0 or index >= int(row["n_test"]) for index in indices):
                violations.append({"dataset": row["dataset"], "alpha": alpha, "type": f"{field}_range"})
            item = metrics.setdefault(field, {"count": 0, "tp": 0, "fp": 0})
            item["count"] += len(indices)
            item["tp"] += len(indices & truth)
            item["fp"] += len(indices - truth)

    mismatches: list[dict[str, object]] = []
    for alpha, bucket in aggregate.items():
        source = reported[alpha]
        for field, values in bucket["metrics"].items():
            values["precision"] = values["tp"] / max(1, values["count"])
            for metric in ("count", "tp", "fp"):
                expected = int(source[f"{field}_{metric}"])
                if values[metric] != expected:
                    mismatches.append(
                        {
                            "alpha": alpha,
                            "field": field,
                            "metric": metric,
                            "independent": values[metric],
                            "reported": expected,
                        }
                    )

    output = {
        "schema": "exp238_independent_reaggregation_v1",
        "prediction_rows": len(predictions),
        "dataset_count": len({row["dataset"] for row in predictions}),
        "alphas": aggregate,
        "partition_or_range_violations": violations,
        "summary_mismatches": mismatches,
        "passed": not violations and not mismatches and len(predictions) == 1117 * 4,
    }
    (root / "exp238_independent_reaggregation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"passed": output["passed"], "rows": len(predictions), "violations": len(violations), "mismatches": len(mismatches)}))


if __name__ == "__main__":
    main()
