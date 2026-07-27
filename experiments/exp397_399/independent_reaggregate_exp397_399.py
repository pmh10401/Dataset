from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def compare_summary(
    candidate_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        key = (str(row["variant"]), float(row["alpha"]))
        grouped.setdefault(key, []).append(row)

    computed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for summary in summary_rows:
        key = (str(summary["variant"]), float(summary["alpha"]))
        rows = grouped.get(key, [])
        workload_type = str(summary["workload_type"])
        if workload_type == "global_k":
            selected = sorted(rows, key=lambda row: int(row["rank"]))[
                : int(summary["workload_k"])
            ]
        elif workload_type == "per_dataset_top1":
            by_dataset: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                by_dataset.setdefault(str(row["dataset"]), []).append(row)
            selected = [
                min(values, key=lambda row: int(row["rank"]))
                for values in by_dataset.values()
                if values
            ]
        else:
            computed.append(
                (summary, {"error": "unknown workload type", "key": key})
            )
            continue
        reviewed = len(selected)
        tp = sum(_truth(row["is_true_anomaly"]) for row in selected)
        fp = reviewed - tp
        precision = tp / reviewed if reviewed else 0.0
        total_tp = sum(_truth(row["is_true_anomaly"]) for row in rows)
        expected = {
            "reviewed": int(summary["reviewed"]),
            "tp": int(summary["tp_at_k"]),
            "fp": int(summary["fp_at_k"]),
            "precision": float(summary["precision_at_k"]),
        }
        actual = {
            "reviewed": reviewed,
            "tp": tp,
            "fp": fp,
            "precision": precision,
            "total_candidates": len(rows),
            "total_tp": total_tp,
            "total_fp": len(rows) - total_tp,
            "capture_rate": tp / total_tp if total_tp else 0.0,
        }
        computed.append((summary, {"expected": expected, "actual": actual}))

    baseline_precision = {
        (
            float(summary["alpha"]),
            str(summary["workload_type"]),
            int(summary["workload_k"]),
        ): float(data["actual"]["precision"])
        for summary, data in computed
        if str(summary["variant"]) == "exp292_baseline" and "actual" in data
    }
    mismatches: list[dict[str, Any]] = []
    for summary, data in computed:
        if "error" in data:
            mismatches.append(data)
            continue
        expected = data["expected"]
        actual = data["actual"]
        baseline = baseline_precision.get(
            (
                float(summary["alpha"]),
                str(summary["workload_type"]),
                int(summary["workload_k"]),
            ),
            0.0,
        )
        extended_mismatch = (
            "total_standard_review_candidates" in summary
            and (
                int(summary["total_standard_review_candidates"])
                != actual["total_candidates"]
                or int(summary["total_standard_review_tp"]) != actual["total_tp"]
                or int(summary["total_standard_review_fp"]) != actual["total_fp"]
                or not _close(
                    float(summary["review_tp_capture_rate"]),
                    actual["capture_rate"],
                )
                or not _close(
                    float(summary["baseline_precision_at_k"]),
                    baseline,
                )
                or not _close(
                    float(summary["precision_lift_vs_exp292"]),
                    actual["precision"] - baseline,
                )
            )
        )
        if (
            expected["reviewed"] != actual["reviewed"]
            or expected["tp"] != actual["tp"]
            or expected["fp"] != actual["fp"]
            or not _close(expected["precision"], actual["precision"])
            or extended_mismatch
        ):
            mismatches.append(
                {
                    "variant": key[0],
                    "alpha": key[1],
                    "workload_type": workload_type,
                    "workload_k": int(summary["workload_k"]),
                    "expected": expected,
                    "actual": actual,
                }
            )
    return mismatches


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_independent_reaggregate(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    candidate_rows = _read_csv(output_root / "exp398_candidate_evaluation.csv")
    summary_rows = _read_csv(output_root / "exp398_precision_at_k.csv")
    mismatches = compare_summary(candidate_rows, summary_rows)
    report = {
        "experiment_id": "experiment_399_independent_review_ranking_audit",
        "candidate_rows": len(candidate_rows),
        "summary_rows": len(summary_rows),
        "mismatches": mismatches,
        "consistent": not mismatches,
        "retrospective_counterfactual": True,
        "eligible_for_deployment": False,
    }
    (output_root / "independent_consistency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_independent_reaggregate(args.output_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
