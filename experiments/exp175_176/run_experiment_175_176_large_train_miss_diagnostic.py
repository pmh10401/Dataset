from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.exp175_176 import diagnostic_core as core


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
BASELINE_PATH = ROOT / "results/exp137/experiment_137_operational_triage_results.csv"
CACHE_MANIFEST_PATH = ROOT / "outputs/exp153_157_train_only_improvements/01_score_cache_manifest.csv"
DEFAULT_CACHE_ROOT = Path(
    "/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements/cache"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/exp175_176_large_train_miss_diagnostic"
EXPECTED_DATASETS = 639
EXPECTED_ANOMALIES = 3440
EXPECTED_HARD_MISSES = 2107


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_indices(value: object) -> set[int]:
    text = str(value or "").replace(",", " ").strip()
    return {int(token) for token in text.split()} if text else set()


def eligible_baseline_rows() -> list[dict[str, str]]:
    rows = [row for row in read_rows(BASELINE_PATH) if int(row["train_normal_count"]) >= 20]
    rows.sort(key=lambda row: row["dataset_name"])
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError(f"eligible dataset count mismatch: {len(rows)} != {EXPECTED_DATASETS}")
    return rows


def load_cache_record(dataset: str, manifest_row: dict[str, str], cache_root: Path) -> tuple[dict[str, object], str]:
    cache_path = Path(cache_root) / manifest_row["cache_file"]
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    actual_sha = sha256_file(cache_path)
    expected_sha = manifest_row["cache_sha256"]
    if actual_sha != expected_sha:
        raise ValueError(f"cache SHA256 mismatch for {dataset}")
    with np.load(cache_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("dataset_name") != dataset:
            raise ValueError(f"cache dataset mismatch for {dataset}")
        n_train = int(metadata["n_train"])
        n_test = int(metadata["n_test"])
        sources: dict[str, dict[str, np.ndarray]] = {}
        for display_name, cache_key in core.SOURCE_CACHE_KEYS.items():
            sources[display_name] = {
                "train": np.asarray(archive[f"{cache_key}__train_oof"], dtype=float),
                "test": np.asarray(archive[f"{cache_key}__test"], dtype=float),
            }
    record: dict[str, object] = {
        "dataset": dataset,
        "n_train": n_train,
        "n_test": n_test,
        "sources": sources,
    }
    core.validate_frozen_record(record)
    return record, actual_sha


def vector_upper_tail_pvalues(train_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(train_scores, dtype=float))
    query = np.asarray(test_scores, dtype=float)
    ge_counts = len(reference) - np.searchsorted(reference, query, side="left")
    return (1.0 + ge_counts.astype(float)) / (len(reference) + 1.0)


def freeze_score_diagnostics(record: dict[str, object], cache_sha256: str) -> dict[str, object]:
    core.validate_frozen_record(record)
    sources = record["sources"]
    frozen_sources: dict[str, dict[str, object]] = {}
    test_scores: dict[str, np.ndarray] = {}
    for source_name in core.SOURCE_NAMES:
        source = sources[source_name]
        train = np.asarray(source["train"], dtype=float)
        test = np.asarray(source["test"], dtype=float)
        test_scores[source_name] = test
        ranks = core.descending_ranks(test)
        frozen_sources[source_name] = {
            "rank": ranks.tolist(),
            "rank_percentile": (ranks / max(1, len(test))).tolist(),
            "pvalue": vector_upper_tail_pvalues(train, test).tolist(),
            "train_score_sha256": array_sha256(train),
            "test_score_sha256": array_sha256(test),
        }
    payload = {
        "dataset": record["dataset"],
        "n_train": int(record["n_train"]),
        "n_test": int(record["n_test"]),
        "cache_sha256": cache_sha256,
        "sources": frozen_sources,
        "mean_rank": core.mean_rank(test_scores).tolist(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
        "prospective_validated": False,
    }
    core.validate_frozen_record(record)
    return payload


def freeze_all(output_dir: Path, cache_root: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["dataset_name"]: row for row in read_rows(CACHE_MANIFEST_PATH)}
    baseline = eligible_baseline_rows()
    missing = [row["dataset_name"] for row in baseline if row["dataset_name"] not in manifest]
    if missing:
        raise ValueError(f"score cache manifest missing {len(missing)} datasets")
    frozen_path = output_dir / "02_frozen_score_diagnostics.jsonl"
    with frozen_path.open("w", encoding="utf-8") as handle:
        for number, row in enumerate(baseline, 1):
            name = row["dataset_name"]
            record, cache_sha = load_cache_record(name, manifest[name], cache_root)
            if int(row["train_normal_count"]) != int(record["n_train"]):
                raise ValueError(f"n_train mismatch for {name}")
            payload = freeze_score_diagnostics(record, cache_sha)
            if int(row["test_size"]) != int(payload["n_test"]):
                raise ValueError(f"n_test mismatch for {name}")
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            if number % 50 == 0 or number == len(baseline):
                print(f"freeze {number}/{len(baseline)}", flush=True)
    prediction_sha = sha256_file(frozen_path)
    manifest_payload = {
        "experiment_id": "experiment_175_large_train_miss_structure_diagnostic",
        "dataset_count": len(baseline),
        "eligible_rule": "train_normal_count >= 20",
        "frozen_prediction_path": str(frozen_path),
        "frozen_prediction_sha256": prediction_sha,
        "baseline_path": str(BASELINE_PATH),
        "baseline_sha256": sha256_file(BASELINE_PATH),
        "cache_manifest_path": str(CACHE_MANIFEST_PATH),
        "cache_manifest_sha256": sha256_file(CACHE_MANIFEST_PATH),
        "database_path": str(DB_PATH),
        "database_sha256": sha256_file(DB_PATH),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "alphas": list(core.ALPHAS),
        "rank_depths": list(core.RANK_DEPTHS),
        "source_names": list(core.SOURCE_NAMES),
        "retrospective": True,
        "prospective_validated": False,
    }
    (output_dir / "03_frozen_prediction_manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_payload


def load_frozen(output_dir: Path) -> dict[str, dict[str, object]]:
    manifest_path = output_dir / "03_frozen_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_path = Path(manifest["frozen_prediction_path"])
    if sha256_file(frozen_path) != manifest["frozen_prediction_sha256"]:
        raise ValueError("frozen prediction SHA256 mismatch")
    rows: dict[str, dict[str, object]] = {}
    with frozen_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("contains_test_labels") is not False:
                raise ValueError("frozen payload contains TEST labels")
            if payload.get("contains_evaluation_metrics") is not False:
                raise ValueError("frozen payload contains evaluation metrics")
            rows[payload["dataset"]] = payload
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError(f"frozen coverage mismatch: {len(rows)}")
    return rows


def load_test_labels(connection: sqlite3.Connection, dataset: str) -> np.ndarray:
    cursor = connection.execute(
        """
        SELECT CAST(i.label AS INTEGER)
        FROM instances i
        JOIN datasets d ON d.id = i.dataset_id
        WHERE d.name = ? AND i.split = 'TEST'
        ORDER BY i.instance_index
        """,
        (dataset,),
    )
    return np.asarray([int(row[0]) for row in cursor.fetchall()], dtype=np.int8)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def evaluate_all(output_dir: Path) -> dict[str, object]:
    frozen = load_frozen(output_dir)
    baseline = eligible_baseline_rows()
    dataset_rows: list[dict[str, object]] = []
    anomaly_rows: list[dict[str, object]] = []
    with sqlite3.connect(DB_PATH) as connection:
        for number, base in enumerate(baseline, 1):
            name = base["dataset_name"]
            score_row = frozen[name]
            labels = load_test_labels(connection, name)
            if len(labels) != int(score_row["n_test"]):
                raise ValueError(f"label length mismatch for {name}")
            true_indices = set(np.flatnonzero(labels != 0).astype(int).tolist())
            hard = parse_indices(base["hard_alert_indices"])
            standard = parse_indices(base["standard_review_indices"])
            priority = parse_indices(base["priority_review_indices"])
            if hard & standard or hard & priority or standard & priority:
                raise ValueError(f"lane overlap for {name}")
            missed = sorted(true_indices - hard)
            local_causes: Counter[str] = Counter()
            source_missed_ranks: dict[str, list[int]] = defaultdict(list)
            for index in missed:
                support_counts: dict[float, int] = {}
                for alpha in core.ALPHAS:
                    support_counts[alpha] = sum(
                        float(score_row["sources"][source]["pvalue"][index]) <= alpha
                        for source in core.SOURCE_NAMES
                    )
                cause = core.classify_miss(index in standard, index in priority, support_counts)
                local_causes[cause] += 1
                item: dict[str, object] = {
                    "dataset": name,
                    "family": base["family"],
                    "candidate_index": index,
                    "true_label": int(labels[index]),
                    "n_train_normal": int(base["train_normal_count"]),
                    "n_test": len(labels),
                    "dataset_anomaly_count": len(true_indices),
                    "in_standard_review": int(index in standard),
                    "in_priority_review": int(index in priority),
                    "cause_category": cause,
                    "mean_rank": float(score_row["mean_rank"][index]),
                    "any_source_min_rank": min(
                        int(score_row["sources"][source]["rank"][index])
                        for source in core.SOURCE_NAMES
                    ),
                    "retrospective_label_used_for_diagnosis_only": 1,
                }
                for alpha in core.ALPHAS:
                    item[f"support_count_alpha_{alpha:g}"] = support_counts[alpha]
                for source in core.SOURCE_NAMES:
                    source_data = score_row["sources"][source]
                    rank = int(source_data["rank"][index])
                    source_missed_ranks[source].append(rank)
                    item[f"{source}_rank"] = rank
                    item[f"{source}_rank_percentile"] = float(source_data["rank_percentile"][index])
                    item[f"{source}_pvalue"] = float(source_data["pvalue"][index])
                for depth in core.RANK_DEPTHS:
                    item[f"any_source_top_{depth}"] = int(item["any_source_min_rank"] <= depth)
                    item[f"mean_rank_top_{depth}"] = int(item["mean_rank"] <= depth)
                anomaly_rows.append(item)
            dataset_item: dict[str, object] = {
                "dataset": name,
                "family": base["family"],
                "n_train_normal": int(base["train_normal_count"]),
                "n_test": len(labels),
                "test_size_bin": core.test_size_bin(len(labels)),
                "anomaly_count": len(true_indices),
                "anomaly_multiplicity_bin": core.multiplicity_bin(len(true_indices)),
                "hard_alert_count": len(hard),
                "hard_tp": len(hard & true_indices),
                "hard_fp": len(hard - true_indices),
                "hard_fn": len(missed),
                "hard_recall": len(hard & true_indices) / max(1, len(true_indices)),
                "hard_miss_share": len(missed) / max(1, len(true_indices)),
                "standard_review_count": len(standard),
                "priority_review_count": len(priority),
            }
            for category in core.CAUSE_CATEGORIES:
                dataset_item[category] = local_causes[category]
            for source in core.SOURCE_NAMES:
                ranks = source_missed_ranks[source]
                dataset_item[f"{source}_miss_rank_median"] = percentile(ranks, 50)
                dataset_item[f"{source}_miss_rank_p90"] = percentile(ranks, 90)
            dataset_rows.append(dataset_item)
            if number % 100 == 0 or number == len(baseline):
                print(f"evaluate {number}/{len(baseline)}", flush=True)

    if len(anomaly_rows) != EXPECTED_HARD_MISSES:
        raise ValueError(f"Hard miss count mismatch: {len(anomaly_rows)} != {EXPECTED_HARD_MISSES}")
    total_anomalies = sum(int(row["anomaly_count"]) for row in dataset_rows)
    if total_anomalies != EXPECTED_ANOMALIES:
        raise ValueError(f"anomaly count mismatch: {total_anomalies} != {EXPECTED_ANOMALIES}")
    write_rows(output_dir / "04_exp175_dataset_diagnostic.csv", dataset_rows)
    write_rows(output_dir / "05_exp176_missed_anomaly_diagnostic.csv", anomaly_rows)

    alpha_rows = []
    for alpha in core.ALPHAS:
        counts = [int(row[f"support_count_alpha_{alpha:g}"]) for row in anomaly_rows]
        alpha_rows.append(
            {
                "alpha": alpha,
                "hard_misses": len(counts),
                "zero_source_support": sum(count == 0 for count in counts),
                "one_source_support": sum(count == 1 for count in counts),
                "two_or_more_source_support": sum(count >= 2 for count in counts),
                "four_source_support": sum(count == 4 for count in counts),
                "scope": "retrospective_diagnostic_only",
            }
        )
    write_rows(output_dir / "06_alpha_visibility_summary.csv", alpha_rows)

    rank_rows = []
    for depth in core.RANK_DEPTHS:
        rank_rows.append(
            {
                "rank_depth_k": depth,
                "hard_misses": len(anomaly_rows),
                "visible_in_any_source_top_k": sum(int(row[f"any_source_top_{depth}"]) for row in anomaly_rows),
                "visible_in_mean_rank_top_k": sum(int(row[f"mean_rank_top_{depth}"]) for row in anomaly_rows),
                "scope": "workload_sensitivity_not_policy_selection",
            }
        )
    write_rows(output_dir / "07_rank_depth_summary.csv", rank_rows)

    cause_counts = Counter(row["cause_category"] for row in anomaly_rows)
    cause_rows = [
        {
            "cause_category": category,
            "hard_miss_count": cause_counts[category],
            "share_of_hard_misses": cause_counts[category] / len(anomaly_rows),
        }
        for category in core.CAUSE_CATEGORIES
    ]
    write_rows(output_dir / "08_cause_summary.csv", cause_rows)
    summary = {
        "experiment_ids": [
            "experiment_175_large_train_miss_structure_diagnostic",
            "experiment_176_budget_vs_representation_diagnostic",
        ],
        "datasets": len(dataset_rows),
        "true_anomalies": total_anomalies,
        "hard_tp": sum(int(row["hard_tp"]) for row in dataset_rows),
        "hard_fp": sum(int(row["hard_fp"]) for row in dataset_rows),
        "hard_misses": len(anomaly_rows),
        "cause_counts": dict(cause_counts),
        "retrospective": True,
        "prospective_validated": False,
        "test_metrics_used_for_policy_selection": False,
    }
    (output_dir / "10_completion.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_contract(output_dir: Path) -> None:
    contract = {
        "experiment_ids": [
            "experiment_175_large_train_miss_structure_diagnostic",
            "experiment_176_budget_vs_representation_diagnostic",
        ],
        "eligible_rule": "train_normal_count >= 20",
        "expected_datasets": EXPECTED_DATASETS,
        "expected_anomalies": EXPECTED_ANOMALIES,
        "expected_hard_misses": EXPECTED_HARD_MISSES,
        "alphas": list(core.ALPHAS),
        "rank_depths": list(core.RANK_DEPTHS),
        "source_names": list(core.SOURCE_NAMES),
        "prediction_reads_test_labels": False,
        "evaluation_reads_test_labels_after_frozen_hash": True,
        "test_metric_policy_selection": False,
        "autonomous_policy_changed": False,
        "retrospective": True,
        "prospective_validated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "01_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "MANIFEST.csv":
            continue
        rows.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "retrospective_or_prospective": "retrospective",
                "autonomous_or_human_assisted": "diagnostic_only",
                "known_limitations": "TEST labels used only after frozen score hash for retrospective diagnosis",
            }
        )
    write_rows(output_dir / "MANIFEST.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "evaluate", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()
    write_contract(args.output_dir)
    if args.phase in {"freeze", "all"}:
        freeze_all(args.output_dir, args.cache_root)
    if args.phase in {"evaluate", "all"}:
        evaluate_all(args.output_dir)
    write_manifest(args.output_dir)


if __name__ == "__main__":
    main()
