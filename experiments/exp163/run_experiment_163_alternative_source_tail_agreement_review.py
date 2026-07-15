#!/usr/bin/env python3
"""Run Exp163 alternative-source tail agreement review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXP153_DIR = PROJECT_DIR / "experiments" / "exp153_157"
for import_path in (PROJECT_DIR, EXP153_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from experiments.exp163.alternative_tail_core import (
    ALPHAS,
    VARIANTS,
    build_prediction_row,
    evaluate_prediction_rows,
    parse_indices,
    validate_lane_contract,
)
from experiments.exp163.exp163_io import (
    FORBIDDEN_PREDICTION_KEYS,
    freeze_prediction_file,
    get_sha256,
    load_dataset_scores,
    load_exp160_predictions,
    load_exp162_contract,
    load_exp162_manifest,
    verify_prediction_manifest,
    write_predictions_atomic,
)

EXP160_PRED_PATH = (
    PROJECT_DIR
    / "outputs"
    / "exp160_candidate_zero_similarity_recovery"
    / "02_exp160_predictions.jsonl"
)
EXP160_EXPECTED_HASH = "ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d"
EXP162_MANIFEST_PATH = (
    PROJECT_DIR
    / "outputs"
    / "exp162_full_coverage_alternative_score_cache"
    / "02_cache_manifest.csv"
)
EXP162_MANIFEST_EXPECTED_HASH = "ec8e1a1c617aa2216a81da1959bd1aec529ad7aadb908ababe3f7e05b7f288c1"
EXP162_CONTRACT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "exp162_full_coverage_alternative_score_cache"
    / "01_source_contract.json"
)
EXP162_CONTRACT_EXPECTED_HASH = "0a80d8a26387d81cd7b2e18ae36f3116c829d20742dd1fa79bb8a3c5bc3d6461"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "exp163_alternative_source_tail_agreement_review"
SMOKE_OUTPUT_DIR = PROJECT_DIR / "outputs" / "exp163_alternative_source_tail_agreement_review_smoke"


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    normalized = []
    fields: list[str] = []
    for source_row in rows:
        row = {}
        for key, value in source_row.items():
            if isinstance(value, (dict, list, tuple, set)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
            row[key] = value
            if key not in fields:
                fields.append(key)
        normalized.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_config(source_config_hash: str) -> dict:
    return {
        "experiment_id": "experiment_163_alternative_source_tail_agreement_review",
        "alphas": list(ALPHAS),
        "variants": list(VARIANTS),
        "sources": ["spectral", "shapelet", "interval"],
        "threshold": "np.quantile(finite_train_scores, 1-alpha, method=higher)",
        "comparison": "test_score > threshold",
        "target_gate": "n_train>=5 and frozen_exp160_candidate_count==0",
        "destination_lane": "standard_review_only",
        "source_config_hash": source_config_hash,
        "uses_test_candidate_empty_gate": 1,
        "uses_test_length_for_policy": 0,
        "uses_family_name": 0,
        "uses_dataset_name_for_policy": 0,
        "retrospective_counterfactual": 1,
        "prospective_validated": 0,
    }


def prediction_config_hash(source_config_hash: str) -> str:
    payload = json.dumps(
        prediction_config(source_config_hash), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_and_validate_inputs() -> tuple[list[dict], dict[str, dict], dict, list[str]]:
    exp160_rows = load_exp160_predictions(EXP160_PRED_PATH, EXP160_EXPECTED_HASH)
    manifest = load_exp162_manifest(
        EXP162_MANIFEST_PATH, EXP162_MANIFEST_EXPECTED_HASH
    )
    contract = load_exp162_contract(
        EXP162_CONTRACT_PATH, EXP162_CONTRACT_EXPECTED_HASH
    )

    keys = []
    alpha_by_dataset: dict[str, set[float]] = defaultdict(set)
    for row in exp160_rows:
        key = (str(row["dataset_name"]), float(row["alpha"]))
        keys.append(key)
        alpha_by_dataset[key[0]].add(key[1])
    if len(exp160_rows) != 1117 * len(ALPHAS):
        raise ValueError(f"Exp160 row count mismatch: {len(exp160_rows)}")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate dataset/alpha rows in Exp160 prediction")
    datasets = sorted(alpha_by_dataset)
    if len(datasets) != 1117:
        raise ValueError(f"Exp160 population must contain 1117 datasets, got {len(datasets)}")
    expected_alphas = set(ALPHAS)
    incomplete = [name for name, values in alpha_by_dataset.items() if values != expected_alphas]
    if incomplete:
        raise ValueError(f"Exp160 alpha coverage mismatch: {incomplete[:10]}")

    manifest_names = set(manifest)
    dataset_names = set(datasets)
    if manifest_names != dataset_names:
        missing = sorted(dataset_names - manifest_names)
        extra = sorted(manifest_names - dataset_names)
        raise ValueError(f"Exp162 coverage mismatch missing={missing[:10]} extra={extra[:10]}")
    bad_status = sorted(
        name
        for name, entry in manifest.items()
        if entry.get("status") not in {"success", "reused"}
    )
    if bad_status:
        raise ValueError(f"Exp162 manifest has unsuccessful rows: {bad_status[:10]}")
    if int(contract.get("population_size", -1)) != 1117:
        raise ValueError("Exp162 contract population_size mismatch")
    if contract.get("contains_test_labels") is not False:
        raise ValueError("Exp162 contract contains_test_labels must be false")
    if contract.get("contains_evaluation_metrics") is not False:
        raise ValueError("Exp162 contract contains_evaluation_metrics must be false")

    return exp160_rows, manifest, contract, datasets


def _baseline_index(rows: list[dict]) -> dict[tuple[str, float], dict]:
    index = {}
    for row in rows:
        key = (str(row["dataset_name"]), float(row["alpha"]))
        if key in index:
            raise ValueError(f"duplicate Exp160 key: {key}")
        index[key] = row
    return index


def run_audit(
    datasets: list[str],
    exp160_rows: list[dict],
    manifest: dict[str, dict],
    contract: dict,
    output_dir: Path,
) -> list[dict]:
    print(f"Auditing {len(datasets)} datasets...", flush=True)
    baseline = _baseline_index(exp160_rows)
    audit_rows = []
    for done, dataset in enumerate(datasets, 1):
        scores = load_dataset_scores(
            dataset, manifest[dataset], str(contract["source_config_hash"]), PROJECT_DIR
        )
        n_train = len(scores["spectral_train"])
        n_test = len(scores["spectral_test"])
        for alpha in ALPHAS:
            row = baseline[(dataset, alpha)]
            if int(row["n_train"]) != n_train or int(row["n_test"]) != n_test:
                raise ValueError(f"Exp160/cache row count mismatch dataset={dataset}")
        audit_rows.append(
            {
                "dataset": dataset,
                "n_train": n_train,
                "n_test": n_test,
                "cache_file": manifest[dataset]["file_path"],
                "cache_sha256": manifest[dataset]["file_hash"],
                "cache_valid": 1,
            }
        )
        if done % 100 == 0 or done == len(datasets):
            print(f"Audit {done}/{len(datasets)}", flush=True)
    _write_csv(output_dir / "01_input_audit.csv", audit_rows)
    _write_json(
        output_dir / "01_input_audit_summary.json",
        {
            "datasets": len(audit_rows),
            "errors": 0,
            "exp160_sha256": EXP160_EXPECTED_HASH,
            "exp162_manifest_sha256": EXP162_MANIFEST_EXPECTED_HASH,
            "exp162_contract_sha256": EXP162_CONTRACT_EXPECTED_HASH,
            "source_config_hash": contract["source_config_hash"],
        },
    )
    print("Audit PASSED.", flush=True)
    return audit_rows


def run_predict(
    datasets: list[str],
    exp160_rows: list[dict],
    manifest: dict[str, dict],
    contract: dict,
    output_dir: Path,
) -> tuple[Path, Path]:
    run_audit(datasets, exp160_rows, manifest, contract, output_dir)
    baseline = _baseline_index(exp160_rows)
    predictions = []
    for done, dataset in enumerate(datasets, 1):
        scores = load_dataset_scores(
            dataset, manifest[dataset], str(contract["source_config_hash"]), PROJECT_DIR
        )
        for alpha in ALPHAS:
            baseline_row = baseline[(dataset, alpha)]
            for variant in VARIANTS:
                predictions.append(
                    build_prediction_row(
                        baseline_row=baseline_row,
                        scores=scores,
                        alpha=alpha,
                        variant=variant,
                    )
                )
        if done % 100 == 0 or done == len(datasets):
            print(f"Prediction {done}/{len(datasets)}", flush=True)

    predictions.sort(key=lambda row: (row["dataset_name"], row["alpha"], row["variant"]))
    expected_rows = len(datasets) * len(ALPHAS) * len(VARIANTS)
    if len(predictions) != expected_rows:
        raise ValueError(
            f"prediction row count mismatch expected={expected_rows} actual={len(predictions)}"
        )
    pred_path = output_dir / "02_exp163_predictions.jsonl"
    manifest_path = output_dir / "03_prediction_manifest.json"
    write_predictions_atomic(pred_path, predictions)
    freeze_prediction_file(
        pred_path,
        manifest_path,
        row_count=expected_rows,
        input_hashes={
            "exp160_predictions": EXP160_EXPECTED_HASH,
            "exp162_cache_manifest": EXP162_MANIFEST_EXPECTED_HASH,
            "exp162_source_contract": EXP162_CONTRACT_EXPECTED_HASH,
        },
        config_hash=prediction_config_hash(str(contract["source_config_hash"])),
    )
    print(f"Frozen {expected_rows} prediction rows: {get_sha256(pred_path)}", flush=True)
    return pred_path, manifest_path


def _aggregate_group(rows: list[dict]) -> dict:
    def total(field: str) -> int:
        return int(sum(int(row[field]) for row in rows))

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows])) if rows else 0.0

    hard_tp, hard_fp = total("hard_tp"), total("hard_fp")
    standard_tp, standard_fp = total("standard_review_tp"), total("standard_review_fp")
    priority_tp, priority_fp = total("priority_review_tp"), total("priority_review_fp")
    added_tp, added_fp = total("added_standard_tp"), total("added_standard_fp")
    workload = np.asarray([int(row["candidate_count"]) for row in rows], dtype=float)
    return {
        "datasets": len(rows),
        "hard_alerts": total("hard_count"),
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
        "mean_hard_recall": mean("hard_recall"),
        "mean_hard_f1": mean("hard_f1"),
        "standard_review_count": total("standard_review_count"),
        "standard_review_tp": standard_tp,
        "standard_review_fp": standard_fp,
        "standard_review_precision": standard_tp / max(1, standard_tp + standard_fp),
        "priority_review_count": total("priority_review_count"),
        "priority_review_tp": priority_tp,
        "priority_review_fp": priority_fp,
        "priority_review_precision": priority_tp / max(1, priority_tp + priority_fp),
        "added_standard_count": total("added_standard_count"),
        "added_standard_tp": added_tp,
        "added_standard_fp": added_fp,
        "added_standard_precision": added_tp / max(1, added_tp + added_fp),
        "mean_combined_f1": mean("combined_f1"),
        "mean_combined_f1_scope": "human_assisted_diagnostic_only",
        "candidate_zero_datasets": sum(int(row["candidate_count"]) == 0 for row in rows),
        "candidate_zero_targets": total("candidate_zero_target"),
        "abstained_datasets": total("abstained"),
        "workload_mean": float(np.mean(workload)),
        "workload_median": float(np.median(workload)),
        "workload_p90": float(np.percentile(workload, 90)),
        "workload_p95": float(np.percentile(workload, 95)),
        "workload_max": int(np.max(workload)),
    }


def run_evaluate(output_dir: Path, expected_row_count: int) -> list[dict]:
    pred_path = output_dir / "02_exp163_predictions.jsonl"
    manifest_path = output_dir / "03_prediction_manifest.json"
    verify_prediction_manifest(
        pred_path, manifest_path, expected_row_count=expected_row_count
    )
    predictions = _read_jsonl(pred_path)
    for row in predictions:
        forbidden = set(row) & FORBIDDEN_PREDICTION_KEYS
        if forbidden:
            raise ValueError(f"prediction contains forbidden evaluation fields: {forbidden}")

    from experiments.exp153_157.exp153_157_bridge import load_test_labels

    results = evaluate_prediction_rows(predictions, label_loader=load_test_labels)
    result_path = output_dir / "04_exp163_results.csv"
    _write_csv(result_path, results)

    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in results:
        groups[(str(row["variant"]), float(row["alpha"]))].append(row)
    summaries = []
    for (variant, alpha), rows in sorted(groups.items()):
        summary = {"variant": variant, "alpha": alpha}
        summary.update(_aggregate_group(rows))
        summaries.append(summary)
    _write_csv(output_dir / "05_exp163_summary.csv", summaries)
    (output_dir / "06_exp163_summary.md").write_text(
        "# Exp163 evaluation summary\n\n"
        "Hard metrics are autonomous. Standard/Priority and combined F1 are "
        "human-assisted diagnostics. All variants and alphas are retrospective "
        "sensitivities; TEST metrics do not select a policy.\n\n```json\n"
        + json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    diff_fields = (
        "dataset_name",
        "alpha",
        "variant",
        "n_train",
        "n_test",
        "candidate_zero_target",
        "source_thresholds",
        "source_candidate_counts",
        "agreement_count",
        "agreement_indices",
        "added_standard_count",
        "added_standard_indices",
        "added_standard_tp",
        "added_standard_fp",
        "candidate_count",
    )
    _write_csv(
        output_dir / "07_candidate_level_diff.csv",
        [{field: row.get(field) for field in diff_fields} for row in results if int(row["candidate_zero_target"])],
    )
    _write_csv(
        output_dir / "08_workload_summary.csv",
        [
            {
                "dataset_name": row["dataset_name"],
                "alpha": row["alpha"],
                "variant": row["variant"],
                "candidate_count": row["candidate_count"],
                "added_standard_count": row["added_standard_count"],
                "candidate_zero_after": int(row["candidate_count"] == 0),
            }
            for row in results
        ],
    )
    print(f"Evaluation complete: {len(results)} rows", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        choices=("audit", "predict", "evaluate", "manifest", "all"),
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    exp160_rows, manifest, contract, full_datasets = load_and_validate_inputs()
    datasets = full_datasets
    if args.smoke:
        datasets = datasets[:10]
    if args.dataset_limit is not None:
        datasets = datasets[: args.dataset_limit]
    if not datasets:
        raise ValueError("empty dataset selection")
    output_dir = Path(args.output_dir) if args.output_dir else (SMOKE_OUTPUT_DIR if args.smoke else DEFAULT_OUTPUT_DIR)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "00_prediction_config.json", prediction_config(str(contract["source_config_hash"])))
    stages = list(args.stage)
    if "all" in stages:
        stages = ["audit", "predict", "evaluate", "manifest"]
    expected_rows = len(datasets) * len(ALPHAS) * len(VARIANTS)
    for stage in stages:
        if stage == "audit":
            run_audit(datasets, exp160_rows, manifest, contract, output_dir)
        elif stage == "predict":
            run_predict(datasets, exp160_rows, manifest, contract, output_dir)
        elif stage == "evaluate":
            run_evaluate(output_dir, expected_rows)
        elif stage == "manifest":
            pred_path = output_dir / "02_exp163_predictions.jsonl"
            freeze_prediction_file(
                pred_path,
                output_dir / "03_prediction_manifest.json",
                row_count=expected_rows,
                input_hashes={
                    "exp160_predictions": EXP160_EXPECTED_HASH,
                    "exp162_cache_manifest": EXP162_MANIFEST_EXPECTED_HASH,
                    "exp162_source_contract": EXP162_CONTRACT_EXPECTED_HASH,
                },
                config_hash=prediction_config_hash(str(contract["source_config_hash"])),
            )


if __name__ == "__main__":
    main()
