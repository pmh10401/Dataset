from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp391_fn_reachability.fn_reachability_core import (
    ALPHAS,
    VOTE_THRESHOLDS,
    aggregate_recovery_rows,
    build_vote_candidates,
    classify_baseline,
    evaluate_recovery,
    load_verified_jsonl,
    sha256_file,
)


EXPERIMENT_ID = "experiment_391_fn_zero_reachability_audit"
EXPECTED_DATASETS = 1117
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP389_ROOT = DATA_ROOT / "exp389_390_dual_confidence_tiers"
EVIDENCE_ROOT = DATA_ROOT / "exp386_388_existing_evidence_alpha_sweep"
COVERAGE_PATH = (
    DATA_ROOT
    / "exp381_385_coverage_first_feature_lattice"
    / "exp381_dataset_coverage.csv"
)
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "exp391_fn_zero_reachability_audit"
EXPECTED_BASELINE = {
    "hard_count": 2218,
    "hard_tp": 1915,
    "hard_fp": 303,
    "hard_fn": 2003,
    "zero_f1_datasets": 356,
    "zero_no_hard_datasets": 229,
    "zero_wrong_hard_datasets": 127,
}


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.3f}".replace(".", "p")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_status(
    output_root: Path,
    status: str,
    done: int,
    total: int,
    errors: int = 0,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "exp391_status.json").write_text(
        json.dumps(
            {
                "id": EXPERIMENT_ID,
                "status": status,
                "phase": status,
                "datasets_done": done,
                "datasets_total": total,
                "progress_percent": 100.0 * done / total if total else 0.0,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_coverage() -> dict[str, dict]:
    with COVERAGE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["dataset"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Exp381 coverage contains duplicate datasets")
    return result


def load_test_labels(dataset_names: set[str]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = defaultdict(list)
    uri = f"file:{DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for dataset, label in connection.execute(
            """
            SELECT d.name, i.label
            FROM datasets AS d
            JOIN instances AS i ON i.dataset_id = d.id
            WHERE i.split = 'TEST'
            ORDER BY d.name, i.instance_index
            """
        ):
            if dataset in dataset_names:
                labels[str(dataset)].append(str(label))
    missing = dataset_names - set(labels)
    if missing:
        raise ValueError(f"missing TEST labels for {len(missing)} datasets")
    return dict(labels)


def load_frozen_inputs(
    dataset_limit: int | None,
) -> tuple[list[dict], dict[float, dict[str, dict]], list[dict]]:
    input_records: list[dict] = []
    hard_path = EXP389_ROOT / "exp389_frozen_predictions.jsonl"
    hard_manifest_path = EXP389_ROOT / "exp389_prediction_manifest.json"
    hard_rows, hard_manifest = load_verified_jsonl(hard_path, hard_manifest_path)
    input_records.append(
        {
            "role": "exp389_hard_and_lanes",
            "path": str(hard_path),
            "sha256": sha256_file(hard_path),
            "manifest_path": str(hard_manifest_path),
            "manifest_sha256": sha256_file(hard_manifest_path),
        }
    )
    if dataset_limit is not None:
        hard_rows = hard_rows[:dataset_limit]
    dataset_names = {str(row["dataset"]) for row in hard_rows}

    evidence_by_alpha: dict[float, dict[str, dict]] = {}
    for alpha in ALPHAS:
        root = EVIDENCE_ROOT / f"alpha_{alpha_slug(alpha)}"
        evidence_path = root / "exp357_frozen_evidence.jsonl"
        manifest_path = root / "exp357_prediction_manifest.json"
        rows, manifest = load_verified_jsonl(evidence_path, manifest_path)
        if abs(float(manifest.get("evidence_alpha", -1)) - alpha) > 1e-12:
            raise ValueError(f"evidence Alpha mismatch: {alpha}")
        mapping = {
            str(row["dataset"]): row
            for row in rows
            if str(row["dataset"]) in dataset_names
        }
        if set(mapping) != dataset_names:
            raise ValueError(f"evidence dataset coverage mismatch: {alpha}")
        evidence_by_alpha[alpha] = mapping
        input_records.append(
            {
                "role": f"exp357_evidence_alpha_{alpha:.3f}",
                "path": str(evidence_path),
                "sha256": sha256_file(evidence_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    input_records.append(
        {
            "role": "exp381_train_coverage",
            "path": str(COVERAGE_PATH),
            "sha256": sha256_file(COVERAGE_PATH),
            "manifest_path": "",
            "manifest_sha256": "",
        }
    )
    input_records.append(
        {
            "role": "read_only_test_labels",
            "path": str(DB_PATH),
            "sha256": sha256_file(DB_PATH),
            "manifest_path": "",
            "manifest_sha256": "",
        }
    )
    if hard_manifest.get("contains_test_labels") is not False:
        raise ValueError("Exp389 manifest contains TEST labels")
    return hard_rows, evidence_by_alpha, input_records


def baseline_metrics(hard: set[int], labels: list[str]) -> dict[str, int | float]:
    truth = {index for index, label in enumerate(labels) if label == "1"}
    tp = len(hard & truth)
    fp = len(hard - truth)
    fn = len(truth - hard)
    precision = tp / len(hard) if hard else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_anomaly_count": len(truth),
        "hard_count": len(hard),
        "hard_tp": tp,
        "hard_fp": fp,
        "hard_fn": fn,
        "hard_precision": precision,
        "hard_recall": recall,
        "hard_f1": f1,
    }


def population_names(row: dict) -> list[str]:
    names = ["all"]
    category = str(row["baseline_category"])
    if category.startswith("zero_"):
        names.extend(["zero_f1", category])
    else:
        names.append("hard_has_tp")
    if int(row["n_test"]) >= 100:
        names.append("test_ge_100")
        if category.startswith("zero_"):
            names.append("zero_f1_test_ge_100")
    if int(row["n_train_normal"]) < 5:
        names.append("train_lt_5")
    elif int(row["n_train_normal"]) < 20:
        names.append("train_5_19")
    else:
        names.append("train_ge_20")
    return names


def aggregate_detail_rows(detail_rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[float, int, str], list[dict]] = defaultdict(list)
    for row in detail_rows:
        for population in population_names(row):
            expanded = dict(row)
            expanded["population"] = population
            buckets[(float(row["alpha"]), int(row["vote_threshold"]), population)].append(
                expanded
            )
    return [
        aggregate_recovery_rows(rows)
        for _, rows in sorted(buckets.items(), key=lambda item: item[0])
    ]


def aggregate_source_rows(source_rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[float, str, str], list[dict]] = defaultdict(list)
    for row in source_rows:
        for population in population_names(row):
            buckets[(float(row["alpha"]), str(row["source"]), population)].append(row)
    output = []
    for (alpha, source, population), rows in sorted(
        buckets.items(), key=lambda item: item[0]
    ):
        candidate_count = sum(int(row["candidate_count"]) for row in rows)
        recovered = sum(int(row["recovered_fn"]) for row in rows)
        baseline_fn = sum(int(row["baseline_fn"]) for row in rows)
        output.append(
            {
                "alpha": alpha,
                "source": source,
                "population": population,
                "datasets": len({row["dataset"] for row in rows}),
                "available_datasets": sum(int(row["source_available"]) for row in rows),
                "baseline_fn": baseline_fn,
                "candidate_count": candidate_count,
                "recovered_fn": recovered,
                "added_fp": sum(int(row["added_fp"]) for row in rows),
                "candidate_precision": recovered / candidate_count if candidate_count else 0.0,
                "fn_recovery_rate": recovered / baseline_fn if baseline_fn else 0.0,
                "review_recovered_fn": sum(
                    int(row["review_recovered_fn"]) for row in rows
                ),
                "review_overlap_fp": sum(
                    int(row["review_overlap_fp"]) for row in rows
                ),
                "review_overlap_precision": (
                    sum(int(row["review_recovered_fn"]) for row in rows)
                    / sum(int(row["review_overlap_count"]) for row in rows)
                    if sum(int(row["review_overlap_count"]) for row in rows)
                    else 0.0
                ),
                "novel_recovered_fn": sum(
                    int(row["novel_recovered_fn"]) for row in rows
                ),
                "selected_for_deployment": False,
                "test_metrics_select_policy": False,
                "retrospective_counterfactual": True,
                "prospective_validated": False,
            }
        )
    return output


def verify_full_baseline(dataset_rows: list[dict]) -> dict:
    totals = {
        "hard_count": sum(int(row["hard_count"]) for row in dataset_rows),
        "hard_tp": sum(int(row["hard_tp"]) for row in dataset_rows),
        "hard_fp": sum(int(row["hard_fp"]) for row in dataset_rows),
        "hard_fn": sum(int(row["hard_fn"]) for row in dataset_rows),
        "zero_f1_datasets": sum(
            str(row["baseline_category"]).startswith("zero_") for row in dataset_rows
        ),
        "zero_no_hard_datasets": sum(
            row["baseline_category"] == "zero_no_hard" for row in dataset_rows
        ),
        "zero_wrong_hard_datasets": sum(
            row["baseline_category"] == "zero_wrong_hard" for row in dataset_rows
        ),
    }
    if totals != EXPECTED_BASELINE:
        raise ValueError(f"frozen baseline mismatch: {totals}")
    return totals


def render_report(
    baseline: dict,
    dataset_rows: list[dict],
    rule_summary: list[dict],
    source_summary: list[dict],
) -> str:
    fixed_rows = sorted(
        (
            row
            for row in rule_summary
            if row["population"] == "zero_f1_test_ge_100"
            and float(row["alpha"]) == 0.005
            and int(row["vote_threshold"]) in (1, 2, 3)
        ),
        key=lambda row: int(row["vote_threshold"]),
    )
    large_rows = [row for row in dataset_rows if int(row["n_test"]) >= 100]
    small_rows = [row for row in dataset_rows if int(row["n_test"]) < 100]
    large_fn = sum(int(row["hard_fn"]) for row in large_rows)
    small_fn = sum(int(row["hard_fn"]) for row in small_rows)
    large_zero = sum(
        str(row["baseline_category"]).startswith("zero_") for row in large_rows
    )
    small_zero = sum(
        str(row["baseline_category"]).startswith("zero_") for row in small_rows
    )
    ewma_signal = next(
        (
            row
            for row in source_summary
            if row["population"] == "zero_f1"
            and row["source"] == "ewma"
            and float(row["alpha"]) == 0.005
        ),
        None,
    )
    lines = [
        "# Exp391 FN/Zero-F1 회수 가능성 감사 결과",
        "",
        "## 기준 자동 경보",
        "",
        f"- 자동 경보: {baseline['hard_count']:,}건",
        f"- TP / FP / FN: {baseline['hard_tp']:,} / {baseline['hard_fp']:,} / {baseline['hard_fn']:,}",
        f"- Zero-F1: {baseline['zero_f1_datasets']:,}개",
        f"- 경보 없음 / 오경보만: {baseline['zero_no_hard_datasets']:,} / {baseline['zero_wrong_hard_datasets']:,}",
        "",
        "## 진단 경계",
        "",
        "- 기존 자동 경보와 검토 lane은 변경하지 않았다.",
        "- 여섯 Alpha와 1/2/3표 합의를 모두 보고하며 하나를 선택하지 않았다.",
        "- TEST 라벨은 동결 입력 해시 검증 후 사후 평가에만 사용했다.",
        "- 모든 결과는 retrospective counterfactual이며 prospective 검증이 아니다.",
        "",
        "## FN 집중 구간",
        "",
        (
            f"- 큰 데이터셋(TEST 100 이상): {len(large_rows):,}개, FN "
            f"{large_fn:,}건, Zero-F1 {large_zero:,}개"
        ),
        (
            f"- 작은 데이터셋(TEST 100 미만): {len(small_rows):,}개, FN "
            f"{small_fn:,}건, Zero-F1 {small_zero:,}개"
        ),
        (
            f"- 큰 데이터셋 FN 비중: "
            f"{100.0 * large_fn / baseline['hard_fn']:.2f}%"
        ),
    ]
    if fixed_rows:
        lines.extend(
            [
                "",
                "## 큰 Zero-F1 데이터셋 사전 고정 진단표",
                "",
                "| Alpha | 동의 수 | 후보 | FN 회수 | 추가 FP | 후보 정밀도 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        lines.extend(
            (
                f"| {float(row['alpha']):.3f} | "
                f"{int(row['vote_threshold'])} | "
                f"{int(row['candidate_count']):,} | "
                f"{int(row['recovered_fn']):,} | "
                f"{int(row['added_fp']):,} | "
                f"{100.0 * float(row['candidate_precision']):.2f}% |"
            )
            for row in fixed_rows
        )
        lines.append(
            "- 세 행은 사전 고정된 진단 조합이며, TEST 결과로 하나를 선택하지 않는다."
        )
    if ewma_signal:
        lines.extend(
            [
                "",
                "## 단일 근거 연구 신호",
                "",
                (
                    f"- EWMA / Alpha 0.005의 Zero-F1 사후 진단: 후보 "
                    f"{ewma_signal['candidate_count']:,}건, FN 회수 "
                    f"{ewma_signal['recovered_fn']:,}건, 추가 FP "
                    f"{ewma_signal['added_fp']:,}건"
                ),
                "- 소수 데이터셋에 집중된 사후 관찰이므로 자동 경보 규칙으로 선택하지 않는다.",
                "",
                "## 운영 결론",
                "",
                "- 새 근거를 자동 경보에 직접 추가하기에는 추가 FP가 너무 많다.",
                "- 기존 검토 후보 안에서 근거를 사용해 검토 순서를 개선할 가능성은 별도로 검증한다.",
                "- Exp391은 기존 자동 경보, 일반 검토, 우선 검토 결과를 변경하지 않았다.",
            ]
        )
    return "\n".join(lines) + "\n"


def run(output_root: Path, dataset_limit: int | None = None) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    write_status(output_root, "verifying_inputs", 0, dataset_limit or EXPECTED_DATASETS)
    hard_rows, evidence_by_alpha, input_records = load_frozen_inputs(dataset_limit)
    coverage = load_coverage()

    # TEST labels are opened only after every frozen prediction/evidence file passed
    # its manifest SHA256 and coverage checks above.
    labels_by_dataset = load_test_labels({str(row["dataset"]) for row in hard_rows})
    dataset_rows: list[dict] = []
    detail_rows: list[dict] = []
    source_rows: list[dict] = []
    for number, hard_row in enumerate(hard_rows, start=1):
        dataset = str(hard_row["dataset"])
        labels = labels_by_dataset[dataset]
        n_test = int(hard_row["n_test"])
        if len(labels) != n_test:
            raise ValueError(f"TEST label length mismatch: {dataset}")
        hard = set(hard_row["automatic_hard_indices"])
        exact_normal = set(hard_row["automatic_normal_exact_indices"])
        standard = set(hard_row["standard_review_indices"])
        priority = set(hard_row["priority_review_indices"])
        review = standard | priority
        metrics = baseline_metrics(hard, labels)
        category = classify_baseline(hard, labels)
        coverage_row = coverage.get(dataset)
        if coverage_row is None:
            raise ValueError(f"missing Exp381 coverage row: {dataset}")
        dataset_row = {
            "dataset": dataset,
            "n_train_normal": int(coverage_row["n_train_normal"]),
            "n_test": n_test,
            "test_size_ge_100": n_test >= 100,
            "baseline_category": category,
            **metrics,
            "standard_review_count": len(standard),
            "priority_review_count": len(priority),
            "exact_normal_count": len(exact_normal),
        }
        for alpha in ALPHAS:
            evidence_row = evidence_by_alpha[alpha][dataset]
            evidence = evidence_row["evidence"]
            available_count = sum(
                payload.get("available") is True for payload in evidence.values()
            )
            alpha_key = alpha_slug(alpha)
            dataset_row[f"available_sources_alpha_{alpha_key}"] = available_count
            for vote_threshold in VOTE_THRESHOLDS:
                candidates = build_vote_candidates(
                    evidence,
                    vote_threshold=vote_threshold,
                    n_test=n_test,
                    excluded_indices=hard | exact_normal,
                )
                recovery = evaluate_recovery(
                    candidate_indices=candidates,
                    hard_indices=hard,
                    review_indices=review,
                    labels=labels,
                )
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "n_train_normal": int(coverage_row["n_train_normal"]),
                        "n_test": n_test,
                        "baseline_category": category,
                        "baseline_fn": int(metrics["hard_fn"]),
                        "alpha": alpha,
                        "vote_threshold": vote_threshold,
                        "available_source_count": available_count,
                        **recovery,
                        "selected_for_deployment": False,
                        "test_metrics_select_policy": False,
                        "retrospective_counterfactual": True,
                        "prospective_validated": False,
                    }
                )
                dataset_row[
                    f"recovered_fn_alpha_{alpha_key}_{vote_threshold}vote"
                ] = recovery["recovered_fn"]
            for source, payload in sorted(evidence.items()):
                candidates = build_vote_candidates(
                    {source: payload},
                    vote_threshold=1,
                    n_test=n_test,
                    excluded_indices=hard | exact_normal,
                )
                source_recovery = evaluate_recovery(
                    candidate_indices=candidates,
                    hard_indices=hard,
                    review_indices=review,
                    labels=labels,
                )
                source_rows.append(
                    {
                        "dataset": dataset,
                        "n_train_normal": int(coverage_row["n_train_normal"]),
                        "n_test": n_test,
                        "baseline_category": category,
                        "baseline_fn": int(metrics["hard_fn"]),
                        "alpha": alpha,
                        "source": source,
                        "source_available": int(payload.get("available") is True),
                        **source_recovery,
                    }
                )
        dataset_rows.append(dataset_row)
        if number == len(hard_rows) or number % 100 == 0:
            write_status(output_root, "evaluating", number, len(hard_rows))
            print(
                f"Progress: [{number}/{len(hard_rows)}] rows={number} "
                f"dataset={dataset} errors=0",
                flush=True,
            )

    baseline = (
        verify_full_baseline(dataset_rows)
        if len(dataset_rows) == EXPECTED_DATASETS
        else {
            "hard_count": sum(int(row["hard_count"]) for row in dataset_rows),
            "hard_tp": sum(int(row["hard_tp"]) for row in dataset_rows),
            "hard_fp": sum(int(row["hard_fp"]) for row in dataset_rows),
            "hard_fn": sum(int(row["hard_fn"]) for row in dataset_rows),
            "zero_f1_datasets": sum(
                str(row["baseline_category"]).startswith("zero_")
                for row in dataset_rows
            ),
            "zero_no_hard_datasets": sum(
                row["baseline_category"] == "zero_no_hard" for row in dataset_rows
            ),
            "zero_wrong_hard_datasets": sum(
                row["baseline_category"] == "zero_wrong_hard" for row in dataset_rows
            ),
        }
    )
    rule_summary = aggregate_detail_rows(detail_rows)
    source_summary = aggregate_source_rows(source_rows)
    high_fn_rows = sorted(
        (row for row in dataset_rows if int(row["hard_fn"]) > 0),
        key=lambda row: (-int(row["hard_fn"]), str(row["dataset"])),
    )

    outputs = {
        "dataset_audit": output_root / "exp391_dataset_audit.csv",
        "rule_detail": output_root / "exp391_rule_detail.csv",
        "rule_summary": output_root / "exp391_rule_summary.csv",
        "source_summary": output_root / "exp391_source_summary.csv",
        "high_fn": output_root / "exp391_high_fn_datasets.csv",
        "report": output_root / "EXP391_RESULTS_KO.md",
    }
    write_csv(outputs["dataset_audit"], dataset_rows)
    write_csv(outputs["rule_detail"], detail_rows)
    write_csv(outputs["rule_summary"], rule_summary)
    write_csv(outputs["source_summary"], source_summary)
    write_csv(outputs["high_fn"], high_fn_rows)
    outputs["report"].write_text(
        render_report(baseline, dataset_rows, rule_summary, source_summary),
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "datasets": len(dataset_rows),
        "alphas": list(ALPHAS),
        "vote_thresholds": list(VOTE_THRESHOLDS),
        "baseline": baseline,
        "inputs": input_records,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
        },
        "frozen_inputs_verified_before_test_label_load": True,
        "automatic_hard_lane_changed": False,
        "review_lanes_changed": False,
        "selected_for_deployment": False,
        "test_metrics_select_policy": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    (output_root / "exp391_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_status(output_root, "completed", len(dataset_rows), len(dataset_rows))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    args = parser.parse_args()
    run(args.output_dir, args.dataset_limit)


if __name__ == "__main__":
    main()
