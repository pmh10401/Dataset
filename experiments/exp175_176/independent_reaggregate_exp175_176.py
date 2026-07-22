from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.exp175_176 import diagnostic_core as core


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs/exp175_176_large_train_miss_diagnostic"


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_indices(value: object) -> set[int]:
    text = str(value or "").replace(",", " ").strip()
    return {int(token) for token in text.split()} if text else set()


def _same_number(actual: object, expected: object, tolerance: float = 1e-10) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return str(actual) == str(expected)


def row_matches(actual: dict[str, object], expected: dict[str, object], fields: tuple[str, ...]) -> bool:
    return all(field in actual and _same_number(actual[field], expected[field]) for field in fields)


def completion_matches(actual: dict[str, object], expected: dict[str, object]) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, (dict, list, bool)):
            if actual[key] != value:
                return False
        elif not _same_number(actual[key], value):
            return False
    return True


def verify_published_summaries(output_dir: Path, misses: list[dict[str, str]]) -> dict[str, bool]:
    prediction_manifest = json.loads(
        (output_dir / "03_frozen_prediction_manifest.json").read_text(encoding="utf-8")
    )
    frozen_path = output_dir / prediction_manifest["frozen_prediction_path"]
    if sha256_file(frozen_path) != prediction_manifest["frozen_prediction_sha256"]:
        raise ValueError("frozen prediction hash mismatch during independent reaggregation")
    database_path = Path(prediction_manifest["database_path"])
    baseline_path = Path(prediction_manifest["baseline_path"])
    if not baseline_path.is_absolute():
        baseline_path = output_dir.parents[1] / baseline_path
    if sha256_file(database_path) != prediction_manifest["database_sha256"]:
        raise ValueError("database snapshot drift during independent reaggregation")
    if sha256_file(baseline_path) != prediction_manifest["baseline_sha256"]:
        raise ValueError("baseline snapshot drift during independent reaggregation")

    baseline = {
        row["dataset_name"]: row
        for row in read_rows(baseline_path)
        if int(row["train_normal_count"]) >= 20
    }
    frozen: dict[str, dict[str, object]] = {}
    with frozen_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frozen[row["dataset"]] = row

    alpha_acc = {alpha: Counter() for alpha in core.ALPHAS}
    alpha_workloads = {alpha: [] for alpha in core.ALPHAS}
    alpha_new_workloads = {alpha: [] for alpha in core.ALPHAS}
    rank_acc = {(method, depth): Counter() for method in ("any_source", "mean_rank") for depth in core.RANK_DEPTHS}
    rank_workloads = {(method, depth): [] for method in ("any_source", "mean_rank") for depth in core.RANK_DEPTHS}
    completion_acc = Counter()
    with sqlite3.connect(database_path) as connection:
        for dataset in sorted(frozen):
            labels = np.asarray(
                [
                    int(row[0])
                    for row in connection.execute(
                        """
                        SELECT CAST(i.label AS INTEGER)
                        FROM instances i JOIN datasets d ON d.id=i.dataset_id
                        WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index
                        """,
                        (dataset,),
                    ).fetchall()
                ],
                dtype=np.int8,
            )
            base = baseline[dataset]
            hard = parse_indices(base["hard_alert_indices"])
            all_lanes = hard | parse_indices(base["standard_review_indices"]) | parse_indices(base["priority_review_indices"])
            truth = set(np.flatnonzero(labels != 0).astype(int).tolist())
            missed = truth - hard
            completion_acc["datasets"] += 1
            completion_acc["true_anomalies"] += len(truth)
            completion_acc["hard_tp"] += len(hard & truth)
            completion_acc["hard_fp"] += len(hard - truth)
            completion_acc["hard_misses"] += len(missed)
            score = frozen[dataset]
            pvalues = np.vstack(
                [np.asarray(score["sources"][source]["pvalue"], dtype=float) for source in core.SOURCE_NAMES]
            )
            ranks = np.vstack(
                [np.asarray(score["sources"][source]["rank"], dtype=int) for source in core.SOURCE_NAMES]
            )
            mean_rank = np.asarray(score["mean_rank"], dtype=float)
            for alpha in core.ALPHAS:
                support = np.sum(pvalues <= alpha, axis=0)
                candidates = set(np.flatnonzero(support >= 2).astype(int).tolist())
                acc = alpha_acc[alpha]
                missed_support = support[list(missed)] if missed else np.asarray([], dtype=int)
                acc["zero_source_support"] += int(np.sum(missed_support == 0))
                acc["one_source_support"] += int(np.sum(missed_support == 1))
                acc["two_or_more_source_support"] += int(np.sum(missed_support >= 2))
                acc["four_source_support"] += int(np.sum(missed_support == 4))
                acc["all_test_candidate_count"] += len(candidates)
                acc["all_test_tp"] += len(candidates & truth)
                acc["all_test_fp"] += len(candidates - truth)
                acc["hard_miss_recovered"] += len(candidates & missed)
                beyond_hard = candidates - hard
                beyond_all = candidates - all_lanes
                acc["new_candidates_beyond_hard"] += len(beyond_hard)
                acc["new_beyond_hard_tp"] += len(beyond_hard & truth)
                acc["new_beyond_hard_fp"] += len(beyond_hard - truth)
                acc["new_candidates_beyond_all_exp137_lanes"] += len(beyond_all)
                acc["new_beyond_all_exp137_lanes_tp"] += len(beyond_all & truth)
                acc["new_beyond_all_exp137_lanes_fp"] += len(beyond_all - truth)
                acc["candidate_nonzero_datasets"] += int(bool(candidates))
                acc["resolution_eligible_datasets"] += int(1.0 / (int(base["train_normal_count"]) + 1) <= alpha)
                acc["new_beyond_all_nonzero_datasets"] += int(bool(beyond_all))
                alpha_workloads[alpha].append(len(candidates))
                alpha_new_workloads[alpha].append(len(beyond_all))
            for depth in core.RANK_DEPTHS:
                candidate_sets = {
                    "any_source": set(np.flatnonzero(np.min(ranks, axis=0) <= depth).astype(int).tolist()),
                    "mean_rank": set(np.flatnonzero(mean_rank <= depth).astype(int).tolist()),
                }
                for method, candidates in candidate_sets.items():
                    acc = rank_acc[(method, depth)]
                    beyond_hard = candidates - hard
                    beyond_all = candidates - all_lanes
                    acc["hard_misses_visible"] += len(candidates & missed)
                    acc["all_test_candidate_count"] += len(candidates)
                    acc["all_test_tp"] += len(candidates & truth)
                    acc["all_test_fp"] += len(candidates - truth)
                    acc["hard_miss_recovered"] += len(candidates & missed)
                    acc["new_candidates_beyond_hard"] += len(beyond_hard)
                    acc["new_beyond_hard_tp"] += len(beyond_hard & truth)
                    acc["new_beyond_hard_fp"] += len(beyond_hard - truth)
                    acc["new_candidates_beyond_all_exp137_lanes"] += len(beyond_all)
                    acc["new_beyond_all_exp137_lanes_tp"] += len(beyond_all & truth)
                    acc["new_beyond_all_exp137_lanes_fp"] += len(beyond_all - truth)
                    rank_workloads[(method, depth)].append(len(candidates))

    alpha_published = {float(row["alpha"]): row for row in read_rows(output_dir / "06_alpha_visibility_summary.csv")}
    alpha_ok = True
    for alpha, acc in alpha_acc.items():
        row = alpha_published[alpha]
        alpha_ok &= row_matches(
            row,
            {"hard_misses": len(misses), "scope": "retrospective_diagnostic_only"},
            ("hard_misses", "scope"),
        )
        for field, value in acc.items():
            alpha_ok &= _same_number(row[field], value)
        workload = alpha_workloads[alpha]
        new_workload = alpha_new_workloads[alpha]
        alpha_ok &= _same_number(row["candidate_mean_per_dataset"], np.mean(workload))
        alpha_ok &= _same_number(
            row["all_test_precision"],
            acc["all_test_tp"] / max(1, acc["all_test_tp"] + acc["all_test_fp"]),
        )
        alpha_ok &= _same_number(row["candidate_p95_per_dataset"], np.percentile(workload, 95))
        alpha_ok &= _same_number(row["candidate_max_per_dataset"], max(workload))
        alpha_ok &= _same_number(row["new_beyond_all_p95_per_dataset"], np.percentile(new_workload, 95))
        alpha_ok &= _same_number(row["new_beyond_all_max_per_dataset"], max(new_workload))

    rank_published = {
        (row["method"], int(row["rank_depth_k"])): row
        for row in read_rows(output_dir / "07_rank_depth_summary.csv")
    }
    rank_ok = True
    for key, acc in rank_acc.items():
        row = rank_published[key]
        rank_ok &= row_matches(
            row,
            {"hard_misses": len(misses), "scope": "workload_sensitivity_not_policy_selection"},
            ("hard_misses", "scope"),
        )
        for field, value in acc.items():
            rank_ok &= _same_number(row[field], value)
        workload = rank_workloads[key]
        rank_ok &= _same_number(row["candidate_mean_per_dataset"], np.mean(workload))
        rank_ok &= _same_number(
            row["all_test_precision"],
            acc["all_test_tp"] / max(1, acc["all_test_tp"] + acc["all_test_fp"]),
        )
        rank_ok &= _same_number(row["candidate_p95_per_dataset"], np.percentile(workload, 95))
        rank_ok &= _same_number(row["candidate_max_per_dataset"], max(workload))

    cause_counts = Counter(row["cause_category"] for row in misses)
    cause_published = {row["cause_category"]: row for row in read_rows(output_dir / "08_cause_summary.csv")}
    cause_ok = all(
        int(cause_published[category]["hard_miss_count"]) == cause_counts[category]
        and _same_number(
            cause_published[category]["share_of_hard_misses"],
            cause_counts[category] / max(1, len(misses)),
        )
        for category in core.CAUSE_CATEGORIES
    )
    completion = json.loads((output_dir / "10_completion.json").read_text(encoding="utf-8"))
    expected_completion = {
        "experiment_ids": [
            "experiment_175_large_train_miss_structure_diagnostic",
            "experiment_176_budget_vs_representation_diagnostic",
        ],
        **dict(completion_acc),
        "cause_counts": dict(cause_counts),
        "retrospective": True,
        "prospective_validated": False,
        "test_metrics_used_for_policy_selection": False,
    }
    completion_ok = completion_matches(completion, expected_completion)

    manifest_ok = True
    manifest_path = output_dir / "MANIFEST.csv"
    if manifest_path.exists():
        for row in read_rows(manifest_path):
            if row["file"] in {"11_independent_reaggregation.json", "13_reaggregation_log.txt"}:
                continue
            manifest_ok &= sha256_file(output_dir / row["file"]) == row["sha256"]
    return {
        "alpha_summary_matches": bool(alpha_ok),
        "rank_summary_matches": bool(rank_ok),
        "cause_summary_matches": bool(cause_ok),
        "completion_summary_matches": bool(completion_ok),
        "manifest_hashes_match_non_circular_files": bool(manifest_ok),
    }


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
        checks.update(verify_published_summaries(output_dir, misses))
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
