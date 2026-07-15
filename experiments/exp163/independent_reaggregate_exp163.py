#!/usr/bin/env python3
"""Read-only independent integrity checks for frozen Exp163 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiments.exp163.alternative_tail_core import ALPHAS, VARIANTS, parse_indices
from experiments.exp163.exp163_io import FORBIDDEN_PREDICTION_KEYS, get_sha256
from experiments.exp163.run_experiment_163_alternative_source_tail_agreement_review import (
    EXP160_EXPECTED_HASH,
    EXP160_PRED_PATH,
    EXP162_CONTRACT_EXPECTED_HASH,
    EXP162_MANIFEST_EXPECTED_HASH,
    load_and_validate_inputs,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_predictions(output_dir: Path) -> dict:
    prediction_path = output_dir / "02_exp163_predictions.jsonl"
    manifest_path = output_dir / "03_prediction_manifest.json"
    rows = _read_jsonl(prediction_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exp160_rows, _, _, datasets = load_and_validate_inputs()
    baseline = {
        (str(row["dataset_name"]), float(row["alpha"])): row for row in exp160_rows
    }

    errors = {
        "duplicate_key": 0,
        "missing_key": 0,
        "forbidden_prediction_field": 0,
        "lane_overlap": 0,
        "lane_union_mismatch": 0,
        "lane_count_mismatch": 0,
        "index_range": 0,
        "hard_baseline_mismatch": 0,
        "priority_baseline_mismatch": 0,
        "target_gate_mismatch": 0,
        "source_agreement_mismatch": 0,
        "added_standard_mismatch": 0,
        "variant_subset_mismatch": 0,
        "prediction_hash_mismatch": 0,
        "manifest_row_count_mismatch": 0,
        "input_hash_mismatch": 0,
    }
    keys = [(row["dataset_name"], float(row["alpha"]), row["variant"]) for row in rows]
    errors["duplicate_key"] = len(keys) - len(set(keys))
    expected_keys = {
        (dataset, alpha, variant)
        for dataset in datasets
        for alpha in ALPHAS
        for variant in VARIANTS
    }
    errors["missing_key"] = len(expected_keys - set(keys))
    row_map = {key: row for key, row in zip(keys, rows)}

    target_rows = 0
    added_standard = 0
    candidate_zero_after = 0
    for key, row in row_map.items():
        dataset, alpha, variant = key
        base = baseline[(dataset, alpha)]
        forbidden = set(row) & FORBIDDEN_PREDICTION_KEYS
        errors["forbidden_prediction_field"] += int(bool(forbidden))
        hard = parse_indices(row["hard_indices"])
        standard = parse_indices(row["standard_review_indices"])
        priority = parse_indices(row["priority_review_indices"])
        candidate = parse_indices(row["candidate_indices"])
        added = parse_indices(row["added_standard_indices"])
        errors["lane_overlap"] += int(bool(hard & standard or hard & priority or standard & priority))
        errors["lane_union_mismatch"] += int(candidate != hard | standard | priority)
        errors["lane_count_mismatch"] += int(
            len(candidate) != int(row["candidate_count"])
            or len(hard) != int(row["hard_count"])
            or len(standard) != int(row["standard_review_count"])
            or len(priority) != int(row["priority_review_count"])
        )
        n_test = int(row["n_test"])
        errors["index_range"] += int(any(index < 0 or index >= n_test for index in candidate))
        base_hard = parse_indices(base["hard_indices"])
        base_priority = parse_indices(base["priority_review_indices"])
        base_standard = parse_indices(base["standard_review_indices"])
        errors["hard_baseline_mismatch"] += int(hard != base_hard)
        errors["priority_baseline_mismatch"] += int(priority != base_priority)
        target = int(base["n_train"]) >= 5 and int(base["candidate_count"]) == 0
        errors["target_gate_mismatch"] += int(int(row["candidate_zero_target"]) != int(target))
        target_rows += int(target)
        source_sets = {
            source: {int(index) for index in indices}
            for source, indices in row["source_candidate_indices"].items()
        }
        union = set().union(*source_sets.values())
        required = 2 if variant == "tail_rank_2of3" else 3
        expected_agreement = {
            index for index in union if sum(index in values for values in source_sets.values()) >= required
        }
        agreement = parse_indices(row["agreement_indices"])
        errors["source_agreement_mismatch"] += int(agreement != expected_agreement)
        expected_added = (agreement - base_hard - base_priority - base_standard) if target else set()
        errors["added_standard_mismatch"] += int(added != expected_added)
        added_standard += len(added)
        candidate_zero_after += int(not candidate)

    for dataset in datasets:
        for alpha in ALPHAS:
            two = parse_indices(row_map[(dataset, alpha, "tail_rank_2of3")]["agreement_indices"])
            three = parse_indices(row_map[(dataset, alpha, "tail_rank_3of3")]["agreement_indices"])
            errors["variant_subset_mismatch"] += int(not three <= two)

    errors["prediction_hash_mismatch"] = int(
        get_sha256(prediction_path) != manifest.get("prediction_sha256")
    )
    errors["manifest_row_count_mismatch"] = int(
        len(rows) != int(manifest.get("row_count", -1))
    )
    expected_inputs = {
        "exp160_predictions": EXP160_EXPECTED_HASH,
        "exp162_cache_manifest": EXP162_MANIFEST_EXPECTED_HASH,
        "exp162_source_contract": EXP162_CONTRACT_EXPECTED_HASH,
    }
    errors["input_hash_mismatch"] = int(manifest.get("input_hashes") != expected_inputs)
    return {
        "prediction_sha256": get_sha256(prediction_path),
        "rows": len(rows),
        "datasets": len({row["dataset_name"] for row in rows}),
        "target_rows_including_variants": target_rows,
        "added_standard_candidates_including_variants": added_standard,
        "candidate_zero_rows_after": candidate_zero_after,
        "errors": errors,
        "passed": not any(errors.values()),
    }


def validate_evaluation(output_dir: Path) -> dict:
    result_path = output_dir / "04_exp163_results.csv"
    prediction_path = output_dir / "02_exp163_predictions.jsonl"
    with result_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    predictions = _read_jsonl(prediction_path)
    result_map = {
        (row["dataset_name"], float(row["alpha"]), row["variant"]): row
        for row in result_rows
    }
    errors = {"result_row_count_mismatch": int(len(result_rows) != len(predictions)), "metric_mismatch": 0}
    from experiments.exp153_157.exp153_157_bridge import load_test_labels

    labels_cache = {}
    for prediction in predictions:
        dataset = prediction["dataset_name"]
        labels = labels_cache.setdefault(dataset, np.asarray(load_test_labels(dataset), dtype=int))
        result = result_map[(dataset, float(prediction["alpha"]), prediction["variant"])]
        for prefix, field in (
            ("hard", "hard_indices"),
            ("standard_review", "standard_review_indices"),
            ("priority_review", "priority_review_indices"),
            ("added_standard", "added_standard_indices"),
            ("combined", "candidate_indices"),
        ):
            indices = sorted(parse_indices(prediction[field]))
            tp = int(np.sum(labels[indices])) if indices else 0
            fp = len(indices) - tp
            fn = int(np.sum(labels)) - tp
            errors["metric_mismatch"] += int(
                int(float(result[f"{prefix}_tp"])) != tp
                or int(float(result[f"{prefix}_fp"])) != fp
                or int(float(result[f"{prefix}_fn"])) != fn
            )
    return {"rows": len(result_rows), "errors": errors, "passed": not any(errors.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/exp163_alternative_source_tail_agreement_review")
    parser.add_argument("--include-evaluation", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    report = {"prediction": validate_predictions(output_dir)}
    if args.include_evaluation:
        report["evaluation"] = validate_evaluation(output_dir)
    destination = output_dir / "11_independent_validation.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(section["passed"] for section in report.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
