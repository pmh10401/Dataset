#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp201_205.core import binary_metrics
from experiments.exp206_210.artifacts import (
    EXP137_CSV,
    EXP206_ROOT,
    freeze_predictions,
    load_dataset_series_only,
    load_population_values_only,
    load_test_labels_for_evaluation,
    load_verified_predictions,
)
from experiments.exp206_210.core import ALPHAS, SEED, empirical_upper_pvalues, sha256_file
from experiments.exp211_214.core import (
    STABILITY_CONTRACT,
    aggregate_fold_pvalues,
    pooled_cross_conformal_pvalues,
    source_stability,
    transport_metrics,
)
from experiments.exp211_214.fold_transport import (
    SOURCE_NAMES,
    build_fold_transport_artifact,
    fold_transport_config_hash,
)


OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp211_214_calibration_transport")
STAGE_META = {
    "211": ("experiment_211_calibration_transport_audit", 251),
    "212": ("experiment_212_fold_ensemble_query_scoring", 251),
    "213": ("experiment_213_train_only_source_stability_guard", 251),
    "214": ("experiment_214_calibration_transport_reaggregation", 251),
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_exp206_vectors(dataset: str) -> dict[str, np.ndarray]:
    path = EXP206_ROOT / "cache" / f"{dataset}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {
            f"{source}_{name}": np.asarray(archive[f"{source}_{name}"], dtype=float)
            for source in SOURCE_NAMES
            for name in ("train", "train_full", "test")
        }


def _fold_cache_path(output_root: Path, dataset: str) -> Path:
    return output_root / "fold_cache" / f"{dataset}.npz"


def _save_fold_artifact(path: Path, artifact: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, metadata_json=json.dumps(metadata, sort_keys=True), **artifact)
    temporary.replace(path)


def _load_fold_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {key: np.asarray(archive[key]) for key in archive.files if key != "metadata_json"}
    return arrays, metadata


def _load_verified_fold_artifact(
    path: Path, expected_sha256: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"fold cache SHA256 mismatch path={path} expected={expected_sha256} actual={actual}"
        )
    return _load_fold_artifact(path)


def _fold_manifest_map(
    output_root: Path, expected_datasets: set[str]
) -> dict[str, dict[str, str]]:
    path = output_root / "exp212_fold_cache_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["dataset"]: row for row in csv.DictReader(handle)}
    if set(rows) != expected_datasets:
        missing = sorted(expected_datasets - set(rows))
        extra = sorted(set(rows) - expected_datasets)
        raise ValueError(f"fold cache coverage mismatch missing={missing[:3]} extra={extra[:3]}")
    return rows


def _process_fold_dataset(entry: dict[str, object], output_root: Path) -> dict[str, object]:
    dataset = str(entry["dataset"])
    path = _fold_cache_path(output_root, dataset)
    config_hash = fold_transport_config_hash()
    if path.exists():
        try:
            _, metadata = _load_fold_artifact(path)
            if (
                metadata.get("config_hash") == config_hash
                and int(metadata.get("n_train", -1)) == int(entry["n_train"])
                and int(metadata.get("n_test", -1)) == int(entry["n_test"])
            ):
                return {"dataset": dataset, "cache_file": path.name, "cache_sha256": sha256_file(path), "status": "reused"}
        except Exception:
            pass
    x_train, x_test, target_length = load_dataset_series_only(dataset)
    artifact = build_fold_transport_artifact(x_train, x_test, seed=SEED)
    metadata = {
        "dataset": dataset,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "target_length": target_length,
        "config_hash": config_hash,
        "seed": SEED,
        "contains_test_labels": False,
    }
    _save_fold_artifact(path, artifact, metadata)
    return {"dataset": dataset, "cache_file": path.name, "cache_sha256": sha256_file(path), "status": "created"}


def _publish(done: int, total: int, rows: int, dataset: str, errors: int = 0) -> None:
    print(f"Progress: [{done} / {total}] rows={rows} dataset={dataset} errors={errors}", flush=True)


def run_stage_211(limit: int | None, output_root: Path) -> None:
    datasets = load_population_values_only(limit)
    rows: list[dict[str, object]] = []
    for number, entry in enumerate(datasets, 1):
        dataset = str(entry["dataset"])
        vectors = _load_exp206_vectors(dataset)
        for source in SOURCE_NAMES:
            oof = vectors[f"{source}_train"]
            full = vectors[f"{source}_train_full"]
            test = vectors[f"{source}_test"]
            metrics = transport_metrics(oof, full, test)
            row: dict[str, object] = {
                "dataset": dataset,
                "source": source,
                "n_train": int(entry["n_train"]),
                "n_test": int(entry["n_test"]),
                **metrics,
                "stable_train_only": int(source_stability(metrics)),
                "contains_test_labels": 0,
                "retrospective_diagnostic": 1,
            }
            for alpha in ALPHAS:
                row[f"oof_reference_test_tail_rate_{alpha}"] = float(np.mean(empirical_upper_pvalues(oof, test) <= alpha))
                row[f"full_reference_test_tail_rate_{alpha}"] = float(np.mean(empirical_upper_pvalues(full, test) <= alpha))
            rows.append(row)
        if number % 5 == 0 or number == len(datasets):
            _publish(number, len(datasets), len(rows), dataset)
    result = output_root / f"{STAGE_META['211'][0]}_results.csv"
    _write_csv(result, rows)
    summary = []
    for source in SOURCE_NAMES:
        selected = [row for row in rows if row["source"] == source]
        summary.append({
            "source": source,
            "datasets": len(selected),
            "stable_datasets": sum(int(row["stable_train_only"]) for row in selected),
            "median_rank_corr": float(np.median([row["oof_full_rank_corr"] for row in selected])),
            "p10_rank_corr": float(np.quantile([row["oof_full_rank_corr"] for row in selected], 0.1)),
            "median_oof_full_ks": float(np.median([row["oof_full_ks"] for row in selected])),
            "median_oof_test_ks": float(np.median([row["oof_test_ks"] for row in selected])),
            "metric_scope": "label_free_transport_diagnostic",
        })
    _write_csv(output_root / f"{STAGE_META['211'][0]}_summary.csv", summary)
    _complete(output_root, "211", len(datasets))


def _source_pvalues(artifact: dict[str, np.ndarray], aggregation: str) -> dict[str, np.ndarray]:
    if aggregation == "pooled_cross_conformal":
        return {
            source: pooled_cross_conformal_pvalues(
                artifact[f"{source}_fold_test_exceedance_counts"],
                artifact[f"{source}_fold_calibration_sizes"],
            )
            for source in SOURCE_NAMES
        }
    return {
        source: aggregate_fold_pvalues(
            artifact[f"{source}_fold_test_pvalues"], aggregation
        )
        for source in SOURCE_NAMES
    }


def _source_flags(artifact: dict[str, np.ndarray], alpha: float, aggregation: str = "median") -> dict[str, np.ndarray]:
    return {source: values <= alpha for source, values in _source_pvalues(artifact, aggregation).items()}


def build_prediction_rows(
    datasets: list[dict[str, object]], output_root: Path, stage: str
) -> list[dict[str, object]]:
    stability: dict[tuple[str, str], bool] = {}
    if stage == "213":
        audit_path = output_root / f"{STAGE_META['211'][0]}_results.csv"
        with audit_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stability[(row["dataset"], row["source"])] = bool(int(row["stable_train_only"]))
    manifest = _fold_manifest_map(
        output_root, {str(entry["dataset"]) for entry in datasets}
    )
    rows: list[dict[str, object]] = []
    for number, entry in enumerate(datasets, 1):
        dataset = str(entry["dataset"])
        artifact, metadata = _load_verified_fold_artifact(
            _fold_cache_path(output_root, dataset), manifest[dataset]["cache_sha256"]
        )
        excluded = set(entry["hard_indices"]) | set(entry["standard_indices"]) | set(entry["priority_indices"])
        for alpha in ALPHAS:
            if stage == "212":
                variants: dict[str, np.ndarray] = {}
                for aggregation in ("pooled_cross_conformal", "median", "conservative_max"):
                    flags = _source_flags(artifact, alpha, aggregation)
                    for source in SOURCE_NAMES:
                        variants[f"{source}_{aggregation}"] = flags[source]
                    support = np.column_stack([flags[source] for source in SOURCE_NAMES]).sum(axis=1)
                    variants[f"alternative_2of3_{aggregation}"] = support >= 2
                    variants[f"alternative_3of3_{aggregation}"] = support == 3
                    variants[f"spectral_interval_2of2_{aggregation}"] = flags["spectral"] & flags["interval"]
            else:
                flags = _source_flags(artifact, alpha, "pooled_cross_conformal")
                stable_sources = [source for source in SOURCE_NAMES if stability.get((dataset, source), False)]
                if len(stable_sources) < 2:
                    variants = {
                        "stable_2ofn_median": np.zeros(int(entry["n_test"]), dtype=bool),
                        "all_stable_median": np.zeros(int(entry["n_test"]), dtype=bool),
                    }
                    abstained = 1
                else:
                    support = np.column_stack([flags[source] for source in stable_sources]).sum(axis=1)
                    variants = {
                        "stable_2ofn_median": support >= 2,
                        "all_stable_median": support == len(stable_sources),
                    }
                    abstained = 0
            for variant, mask in variants.items():
                indices = set(np.flatnonzero(mask).tolist()) - excluded
                rows.append({
                    "experiment_id": STAGE_META[stage][0],
                    "dataset": dataset,
                    "variant": f"{variant}__{alpha}",
                    "alpha": alpha,
                    "n_train": int(entry["n_train"]),
                    "n_test": int(entry["n_test"]),
                    "candidate_indices": sorted(indices),
                    "candidate_count": len(indices),
                    "final_lane": "research_sensitivity_only",
                    "autonomous_alert": False,
                    "abstained": abstained if stage == "213" else 0,
                    "resolution_blocked": int(alpha < _minimum_attainable_p(
                        artifact,
                        "pooled_cross_conformal" if stage == "213" else (
                            "pooled_cross_conformal" if "pooled_cross_conformal" in variant
                            else "conservative_max" if "conservative_max" in variant
                            else "median"
                        ),
                    )),
                    "stable_sources": stable_sources if stage == "213" else list(SOURCE_NAMES),
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "uses_family_name": 0,
                    "uses_dataset_name_for_policy": 0,
                    "uses_test_length_for_policy": 0,
                    "retrospective_counterfactual": 1,
                })
        if number % 5 == 0 or number == len(datasets):
            _publish(number, len(datasets), len(rows), dataset)
    return sorted(rows, key=lambda row: (row["dataset"], row["variant"]))


def _minimum_attainable_p(artifact: dict[str, np.ndarray], aggregation: str) -> float:
    sizes = np.asarray(artifact["spectral_fold_calibration_sizes"], dtype=float)
    floors = 1.0 / (sizes + 1.0)
    if aggregation == "pooled_cross_conformal":
        return 1.0 / (float(np.sum(sizes)) + 1.0)
    if aggregation == "median":
        return float(np.median(floors))
    if aggregation == "conservative_max":
        return float(np.max(floors))
    raise ValueError(f"unknown aggregation for resolution: {aggregation}")


def _evaluate_predictions(
    frozen: list[dict[str, object]], datasets: list[dict[str, object]], stage: str
) -> list[dict[str, object]]:
    entries = {str(entry["dataset"]): entry for entry in datasets}
    rows = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pred in frozen:
        grouped[str(pred["dataset"])].append(pred)
    for dataset, predictions in grouped.items():
        labels = load_test_labels_for_evaluation(dataset)
        baseline_hard = set(entries[dataset]["hard_indices"])
        baseline = binary_metrics(labels, baseline_hard)
        for pred in predictions:
            candidates = set(int(value) for value in pred["candidate_indices"])
            added = binary_metrics(labels, candidates)
            combined = binary_metrics(labels, baseline_hard | candidates)
            rows.append({
            "experiment_id": STAGE_META[stage][0],
            "dataset_name": dataset,
            "variant": pred["variant"],
            "n_train": pred["n_train"],
            "n_test": pred["n_test"],
            "anomaly_count": int(np.sum(labels)),
            "candidate_indices": " ".join(map(str, sorted(candidates))),
            "candidate_count": added["count"],
            "candidate_tp": added["tp"],
            "candidate_fp": added["fp"],
            "candidate_precision": added["precision"],
            "baseline_hard_tp": baseline["tp"],
            "baseline_hard_fp": baseline["fp"],
            "baseline_hard_f1": baseline["f1"],
            "counterfactual_tp": combined["tp"],
            "counterfactual_fp": combined["fp"],
            "counterfactual_f1": combined["f1"],
            "zero_f1_before": int(baseline["f1"] == 0.0),
            "zero_f1_after": int(combined["f1"] == 0.0),
            "zero_f1_fixed": int(baseline["f1"] == 0.0 and combined["f1"] > 0.0),
            "metric_scope": "retrospective_research_candidate",
            "autonomous_exp137_unchanged": 1,
                "selected_for_operation": 0,
            })
    return rows


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    summaries = []
    for variant, selected in sorted(grouped.items()):
        tp = sum(int(row["candidate_tp"]) for row in selected)
        fp = sum(int(row["candidate_fp"]) for row in selected)
        counts = np.asarray([int(row["candidate_count"]) for row in selected])
        summaries.append({
            "variant": variant,
            "datasets": len(selected),
            "active_datasets": int(np.sum(counts > 0)),
            "candidate_count": int(np.sum(counts)),
            "candidate_tp": tp,
            "candidate_fp": fp,
            "candidate_precision": tp / max(1, tp + fp),
            "baseline_mean_hard_f1": float(np.mean([row["baseline_hard_f1"] for row in selected])),
            "counterfactual_mean_f1": float(np.mean([row["counterfactual_f1"] for row in selected])),
            "zero_f1_before": sum(int(row["zero_f1_before"]) for row in selected),
            "zero_f1_after": sum(int(row["zero_f1_after"]) for row in selected),
            "zero_f1_fixed": sum(int(row["zero_f1_fixed"]) for row in selected),
            "workload_median": float(np.median(counts)),
            "workload_p95": float(np.quantile(counts, 0.95)),
            "workload_max": int(np.max(counts)),
            "metric_scope": "retrospective_research_candidate",
            "autonomous_exp137_unchanged": 1,
            "selected_for_operation": 0,
        })
    return summaries


def _prediction_input_hashes(output_root: Path, stage: str) -> dict[str, str]:
    paths = [
        EXP137_CSV,
        EXP206_ROOT / "01_score_cache_manifest.csv",
        output_root / "exp212_fold_cache_manifest.csv",
    ]
    if stage == "213":
        paths.append(output_root / f"{STAGE_META['211'][0]}_results.csv")
    return {str(path): sha256_file(path) for path in paths}


def run_stage_212(limit: int | None, output_root: Path, workers: int) -> None:
    datasets = load_population_values_only(limit)
    error_path = output_root / "exp212_errors.csv"
    error_path.unlink(missing_ok=True)
    manifest_rows, errors = [], []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_process_fold_dataset, entry, output_root): str(entry["dataset"]) for entry in datasets}
        for number, future in enumerate(as_completed(futures), 1):
            dataset = futures[future]
            try:
                manifest_rows.append(future.result())
            except Exception as error:
                errors.append({"dataset": dataset, "error": repr(error)})
            if number % 5 == 0 or number == len(futures):
                _publish(number, len(datasets), len(manifest_rows), dataset, len(errors))
    if errors:
        _write_csv(error_path, errors)
        raise SystemExit(2)
    manifest_rows.sort(key=lambda row: row["dataset"])
    manifest_path = output_root / "exp212_fold_cache_manifest.csv"
    _write_csv(manifest_path, manifest_rows)
    predictions = build_prediction_rows(datasets, output_root, "212")
    prediction_path, prediction_manifest = freeze_predictions(
        "212", STAGE_META["212"][0], predictions, output_root, _prediction_input_hashes(output_root, "212")
    )
    frozen = load_verified_predictions(prediction_path, prediction_manifest)
    results = _evaluate_predictions(frozen, datasets, "212")
    _write_csv(output_root / f"{STAGE_META['212'][0]}_results.csv", results)
    _write_csv(output_root / f"{STAGE_META['212'][0]}_summary.csv", _summarize(results))
    _complete(output_root, "212", len(datasets))


def run_stage_213(limit: int | None, output_root: Path) -> None:
    datasets = load_population_values_only(limit)
    predictions = build_prediction_rows(datasets, output_root, "213")
    prediction_path, prediction_manifest = freeze_predictions(
        "213", STAGE_META["213"][0], predictions, output_root, _prediction_input_hashes(output_root, "213")
    )
    frozen = load_verified_predictions(prediction_path, prediction_manifest)
    results = _evaluate_predictions(frozen, datasets, "213")
    _write_csv(output_root / f"{STAGE_META['213'][0]}_results.csv", results)
    _write_csv(output_root / f"{STAGE_META['213'][0]}_summary.csv", _summarize(results))
    (output_root / "exp213_stability_contract.json").write_text(
        json.dumps(STABILITY_CONTRACT, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _complete(output_root, "213", len(datasets))


def run_stage_214(limit: int | None, output_root: Path) -> None:
    datasets = load_population_values_only(limit)
    rows = []
    for stage in ("212", "213"):
        path = output_root / f"{STAGE_META[stage][0]}_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({"source_experiment": STAGE_META[stage][0], **row})
    result = output_root / f"{STAGE_META['214'][0]}_results.csv"
    _write_csv(result, rows)
    summary = [{
        "experiment_id": STAGE_META["214"][0],
        "datasets": len(datasets),
        "compared_variants": len(rows),
        "autonomous_exp137_unchanged": 1,
        "metric_scope": "retrospective_research_reaggregation",
        "selected_for_operation": 0,
    }]
    _write_csv(output_root / f"{STAGE_META['214'][0]}_summary.csv", summary)
    _complete(output_root, "214", len(datasets))


def _complete(output_root: Path, stage: str, datasets: int) -> None:
    (output_root / f"exp{stage}_completion.json").write_text(
        json.dumps({"experiment_id": STAGE_META[stage][0], "status": "completed", "datasets": datasets}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp211-214 calibration transport runner")
    parser.add_argument("--stage", required=True, choices=STAGE_META)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.stage == "211":
        run_stage_211(args.dataset_limit, args.output_dir)
    elif args.stage == "212":
        run_stage_212(args.dataset_limit, args.output_dir, args.workers)
    elif args.stage == "213":
        run_stage_213(args.dataset_limit, args.output_dir)
    else:
        run_stage_214(args.dataset_limit, args.output_dir)
    print(f"Stage {args.stage} elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
