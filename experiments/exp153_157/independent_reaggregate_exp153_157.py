"""Read-only consistency checks for Exp153--157 generated artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import exp153_157_core as core
import exp153_157_reporting as reporting

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/exp153_157_train_only_improvements"

ARTIFACTS = (
    ("exp154", "05_exp154_predictions.jsonl", "07_exp154_results.csv"),
    ("exp155", "11_exp155_predictions.jsonl", "13_exp155_results.csv"),
    ("exp156", "16_exp156_predictions.jsonl", "18_exp156_results.csv"),
    ("exp157", "21_exp157_predictions.jsonl", "23_exp157_results.csv"),
)


def check_prediction_rows(rows):
    overlap = union_mismatch = forbidden = 0
    for row in rows:
        hard = core.parse_indices(row.get("hard_indices"))
        standard = core.parse_indices(row.get("standard_review_indices"))
        priority = core.parse_indices(row.get("priority_review_indices"))
        candidate = core.parse_indices(row.get("candidate_indices"))
        overlap += int(bool(hard & standard or hard & priority or standard & priority))
        union_mismatch += int(candidate != hard | standard | priority)
        forbidden += int(bool(reporting.FORBIDDEN_PREDICTION_KEYS & set(row)))
    return overlap, union_mismatch, forbidden


def aggregate_result_rows(rows, group_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        hard_tp = sum(int(float(row.get("hard_tp", 0) or 0)) for row in group)
        hard_fp = sum(int(float(row.get("hard_fp", 0) or 0)) for row in group)
        output.append(
            {
                **{field: value for field, value in zip(group_fields, key)},
                "datasets": len(group),
                "hard_alerts": sum(int(float(row.get("hard_count", 0) or 0)) for row in group),
                "hard_tp": hard_tp,
                "hard_fp": hard_fp,
                "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
                "standard_review_candidates": sum(int(float(row.get("standard_review_count", 0) or 0)) for row in group),
                "priority_review_candidates": sum(int(float(row.get("priority_review_count", 0) or 0)) for row in group),
                "mean_hard_f1": sum(float(row.get("hard_f1", 0) or 0) for row in group) / max(1, len(group)),
            }
        )
    return output


def run(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    consistency = []
    reaggregated = []
    for label, prediction_name, result_name in ARTIFACTS:
        prediction_path = output_dir / prediction_name
        result_path = output_dir / result_name
        if not prediction_path.exists() or not result_path.exists():
            consistency.append({"artifact": label, "status": "missing"})
            continue
        prediction_rows = reporting.read_jsonl(prediction_path)
        result_rows = reporting.read_rows(result_path)
        overlap, union_mismatch, forbidden = check_prediction_rows(prediction_rows)
        consistency.append(
            {
                "artifact": label,
                "status": "present",
                "prediction_rows": len(prediction_rows),
                "result_rows": len(result_rows),
                "lane_overlap_rows": overlap,
                "candidate_union_mismatch_rows": union_mismatch,
                "forbidden_prediction_field_rows": forbidden,
                "prediction_sha256": core.sha256_file(prediction_path),
                "result_sha256": core.sha256_file(result_path),
            }
        )
        group_fields = ["experiment_id", "variant"]
        if label != "exp157":
            group_fields.append("alpha")
        else:
            group_fields.append("risk_target_per_run")
        for row in aggregate_result_rows(result_rows, group_fields):
            reaggregated.append({"artifact": label, **row})

    reporting.write_rows(output_dir / "29_independent_reaggregation.csv", reaggregated)
    reporting.write_rows(output_dir / "29a_consistency_checks.csv", consistency)
    (output_dir / "30_consistency_report.md").write_text(
        "# Exp153--157 independent consistency report\n\n"
        "This report is read-only. It does not select alpha or any policy variant.\n\n"
        "```json\n"
        + json.dumps(consistency, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    return consistency, reaggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(Path(args.output_dir))


if __name__ == "__main__":
    main()
