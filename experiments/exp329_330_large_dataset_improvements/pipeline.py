from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.exp309_316_fn_recovery.reporting import write_csv
from experiments.exp317_318_train_reliability_fusion.contracts import freeze_jsonl, verify_frozen_jsonl
from experiments.exp325_327_full_source_reliability.pipeline import _load_record_without_labels

from .config import (
    ALPHAS,
    DB_PATH,
    EXPECTED_DATASETS,
    EXP328_ROOT,
    EXP329_ID,
    EXP329_ROOT,
    EXP330_ID,
    EXP330_ROOT,
    PCA_DIMENSIONS,
    RANDOM_STATE,
)


VERIFIER_VARIANTS = (
    "review_independent_2",
    "review_total_support_3",
    "outside_strict_4of8",
    "combined_strict",
    "combined_balanced",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _status(root: Path, experiment_id: str, state: str, phase: str, done: int, total: int, started: float) -> None:
    updated = time.time()
    _atomic_json(root / "status.json", {
        "id": experiment_id,
        "experiment_id": experiment_id,
        "status": state,
        "phase": phase,
        "done": done,
        "expected": total,
        "datasets_done": done,
        "datasets_total": total,
        "progress_percent": 100.0 * done / total if total else 0.0,
        "elapsed_seconds": updated - started,
        "updated_unix": updated,
        "updated_at_unix": updated,
        "retrospective": True,
        "automatic_policy_changed": False,
        "research_shadow_only": True,
    })


def evidence_for_index(row: dict, index: int) -> dict[str, object]:
    primary = row.get("primary_sources", {})
    independent = row.get("independent_sources", {})
    primary_names = [name for name, source in primary.items() if index in set(source.get("candidate_indices", []))]
    independent_names = [name for name, indices in independent.items() if index in set(indices)]
    ranks = []
    for source in primary.values():
        order = list(source.get("order", []))
        if index in order:
            ranks.append(order.index(index) + 1)
    return {
        "in_existing_review": index in set(row.get("default_review_indices", [])),
        "primary_count": len(primary_names),
        "independent_count": len(independent_names),
        "total_support": len(primary_names) + len(independent_names),
        "best_primary_rank": min(ranks) if ranks else 10**9,
        "primary_sources": primary_names,
        "independent_sources": independent_names,
    }


def select_verifier_variants(evidence: dict[str, object]) -> dict[str, bool]:
    review = bool(evidence["in_existing_review"])
    independent = int(evidence["independent_count"])
    total = int(evidence["total_support"])
    rank = int(evidence["best_primary_rank"])
    review_independent = review and independent >= 2
    review_support = review and independent >= 1 and total >= 3
    outside_strict = (not review) and independent >= 2 and total >= 4 and rank <= 12
    combined_strict = review_support or outside_strict
    combined_balanced = (
        (review and independent >= 1 and total >= 2)
        or ((not review) and independent >= 1 and total >= 3 and rank <= 5)
    )
    return {
        "review_independent_2": review_independent,
        "review_total_support_3": review_support,
        "outside_strict_4of8": outside_strict,
        "combined_strict": combined_strict,
        "combined_balanced": combined_balanced,
    }


def _exp328_rows() -> list[dict]:
    rows = verify_frozen_jsonl(
        EXP328_ROOT / "exp328_frozen_reachability.jsonl",
        EXP328_ROOT / "exp328_prediction_manifest.json",
    )
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError(f"Exp328 coverage mismatch: {len(rows)}/{EXPECTED_DATASETS}")
    return rows


def run_exp329_predict(root: Path = EXP329_ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_rows = _exp328_rows()
    _status(root, EXP329_ID, "running", "frozen_train_only_evidence_verifier", 0, len(source_rows), started)
    output = []
    for done, row in enumerate(source_rows, 1):
        hard = set(row["automatic_hard_indices"])
        review = set(row["default_review_indices"])
        source_candidates = set().union(
            *(set(source.get("candidate_indices", [])) for source in row["primary_sources"].values()),
            *(set(indices) for indices in row["independent_sources"].values()),
        )
        universe = sorted((review | source_candidates) - hard)
        selected = {variant: [] for variant in VERIFIER_VARIANTS}
        evidence_rows = []
        for index in universe:
            evidence = evidence_for_index(row, index)
            decisions = select_verifier_variants(evidence)
            for variant, accepted in decisions.items():
                if accepted:
                    selected[variant].append(index)
            evidence_rows.append({
                "index": index,
                "in_existing_review": evidence["in_existing_review"],
                "primary_count": evidence["primary_count"],
                "independent_count": evidence["independent_count"],
                "total_support": evidence["total_support"],
                "best_primary_rank": evidence["best_primary_rank"],
            })
        output.append({
            "experiment_id": EXP329_ID,
            "dataset": row["dataset"],
            "n_train_normal": row["n_train_normal"],
            "n_test": row["n_test"],
            "automatic_hard_indices": sorted(hard),
            "existing_review_indices": sorted(review),
            "candidate_universe_indices": universe,
            "selected_indices": selected,
            "candidate_evidence": evidence_rows,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "uses_test_metrics_for_selection": False,
            "automatic_policy_changed": False,
            "research_shadow_only": True,
        })
        if done % 20 == 0 or done == len(source_rows):
            _status(root, EXP329_ID, "running", "frozen_train_only_evidence_verifier", done, len(source_rows), started)
    freeze_jsonl(output, root / "exp329_frozen_predictions.jsonl", root / "exp329_prediction_manifest.json",
                 experiment_id=EXP329_ID, metadata={"datasets": len(output), "variants": list(VERIFIER_VARIANTS),
                 "rules_frozen_before_label_evaluation": True})
    _status(root, EXP329_ID, "completed", "prediction", len(output), len(output), started)


def _labels(dataset: str) -> np.ndarray:
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute("""SELECT i.label FROM instances i JOIN datasets d ON i.dataset_id=d.id
            WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index""", (dataset,)).fetchall()
    return np.asarray([int(row[0]) for row in rows], dtype=np.int64)


def _binary_metrics(indices: set[int], truth: set[int]) -> dict[str, float | int]:
    tp = len(indices & truth)
    fp = len(indices - truth)
    fn = len(truth - indices)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(indices), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def run_exp329_evaluate(root: Path = EXP329_ROOT) -> None:
    started = time.time()
    rows = verify_frozen_jsonl(root / "exp329_frozen_predictions.jsonl", root / "exp329_prediction_manifest.json")
    _status(root, EXP329_ID, "running", "retrospective_evaluation", 0, len(rows), started)
    detail = []
    for done, row in enumerate(rows, 1):
        truth = set(np.flatnonzero(_labels(row["dataset"]) == 1).astype(int).tolist())
        hard = set(row["automatic_hard_indices"])
        for variant in VERIFIER_VARIANTS:
            selected = set(row["selected_indices"][variant])
            added = _binary_metrics(selected, truth)
            combined = _binary_metrics(hard | selected, truth)
            detail.append({
                "experiment_id": EXP329_ID, "dataset": row["dataset"], "variant": variant,
                "n_train_normal": row["n_train_normal"], "n_test": row["n_test"],
                "shadow_count": added["count"], "shadow_tp": added["tp"], "shadow_fp": added["fp"],
                "shadow_precision": added["precision"], "combined_hard_count": combined["count"],
                "combined_hard_tp": combined["tp"], "combined_hard_fp": combined["fp"],
                "combined_hard_precision": combined["precision"], "combined_hard_recall": combined["recall"],
                "combined_hard_f1": combined["f1"], "metric_scope": "retrospective_shadow_only",
            })
        if done % 20 == 0 or done == len(rows):
            _status(root, EXP329_ID, "running", "retrospective_evaluation", done, len(rows), started)
    write_csv(root / "exp329_results.csv", detail)
    summary = []
    for variant in VERIFIER_VARIANTS:
        group = [item for item in detail if item["variant"] == variant]
        shadow_tp = sum(int(item["shadow_tp"]) for item in group)
        shadow_fp = sum(int(item["shadow_fp"]) for item in group)
        summary.append({
            "experiment_id": EXP329_ID, "variant": variant, "datasets": len(group),
            "shadow_count": shadow_tp + shadow_fp, "shadow_tp": shadow_tp, "shadow_fp": shadow_fp,
            "shadow_precision": shadow_tp / (shadow_tp + shadow_fp) if shadow_tp + shadow_fp else 0.0,
            "combined_hard_tp": sum(int(item["combined_hard_tp"]) for item in group),
            "combined_hard_fp": sum(int(item["combined_hard_fp"]) for item in group),
            "mean_combined_hard_f1": float(np.mean([float(item["combined_hard_f1"]) for item in group])),
            "automatic_policy_changed": False,
        })
    write_csv(root / "exp329_summary.csv", summary)
    _status(root, EXP329_ID, "completed", "evaluation", len(rows), len(rows), started)


def _resample(series: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(length, dtype=np.float64)
    if values.size == 1:
        return np.full(length, values[0], dtype=np.float64)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, values)


def multiscale_features(series_rows: list[np.ndarray]) -> np.ndarray:
    output = []
    for series in series_rows:
        raw = _resample(series, 64)
        median = float(np.median(raw))
        scale = float(np.median(np.abs(raw - median))) * 1.4826
        if not np.isfinite(scale) or scale < 1e-8:
            scale = float(np.std(raw))
        scale = scale if np.isfinite(scale) and scale >= 1e-8 else 1.0
        raw = (raw - median) / scale
        diff = _resample(np.diff(raw), 32)
        spectrum = np.log1p(np.abs(np.fft.rfft(raw)))[1:33]
        blocks = raw.reshape(16, 4)
        feature = np.concatenate([raw, diff, spectrum, blocks.mean(axis=1), blocks.std(axis=1)])
        output.append(feature)
    return np.asarray(output, dtype=np.float32)


def empirical_upper_pvalues(calibration_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    calibration = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    query = np.asarray(query_scores, dtype=np.float64).reshape(-1)
    return (1.0 + (calibration[None, :] >= query[:, None]).sum(axis=1)) / (len(calibration) + 1.0)


def _knn_scores(train_embedding: np.ndarray, test_embedding: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    effective = min(k, len(train_embedding) - 1)
    if effective < 1:
        raise ValueError("at least two TRAIN-normal rows are required")
    train_nn = NearestNeighbors(n_neighbors=effective + 1, n_jobs=1).fit(train_embedding)
    train_scores = train_nn.kneighbors(train_embedding, return_distance=True)[0][:, -1]
    test_nn = NearestNeighbors(n_neighbors=effective, n_jobs=1).fit(train_embedding)
    test_scores = test_nn.kneighbors(test_embedding, return_distance=True)[0][:, -1]
    return train_scores, test_scores


def predict_learned_dataset(record: dict, *, pca_dimensions: tuple[int, ...] = PCA_DIMENSIONS) -> dict:
    train_features = multiscale_features(record["train_series"])
    test_features = multiscale_features(record["test_series"])
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    variants = {}
    with threadpool_limits(limits=1):
        for requested in pca_dimensions:
            dimensions = min(requested, train_scaled.shape[0] - 1, train_scaled.shape[1])
            pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=RANDOM_STATE)
            train_embedding = pca.fit_transform(train_scaled)
            test_embedding = pca.transform(test_scaled)
            train_scores, test_scores = _knn_scores(train_embedding, test_embedding, k=5)
            pvalues = empirical_upper_pvalues(train_scores, test_scores)
            variants[f"multiscale_pca{requested}_knn5"] = {
                "actual_dimensions": int(dimensions),
                "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
                "train_score_median": float(np.median(train_scores)),
                "train_score_p95": float(np.quantile(train_scores, 0.95)),
                "candidate_indices": {
                    str(alpha): np.flatnonzero(pvalues <= alpha).astype(int).tolist() for alpha in ALPHAS
                },
            }
    return {
        "experiment_id": EXP330_ID,
        "dataset": record["dataset_name"],
        "n_train_normal": len(record["train_series"]),
        "n_test": len(record["test_series"]),
        "feature_dimensions": int(train_features.shape[1]),
        "variants": variants,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_test_metrics_for_selection": False,
        "automatic_policy_changed": False,
        "research_shadow_only": True,
    }


def _predict_dataset_name(dataset: str) -> dict:
    return predict_learned_dataset(_load_record_without_labels(dataset))


def run_exp330_predict(root: Path = EXP330_ROOT, *, workers: int = 7, limit: int | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    names = [row["dataset"] for row in _exp328_rows()]
    if limit:
        names = names[:limit]
    _status(root, EXP330_ID, "running", "train_normal_multiscale_representation", 0, len(names), started)
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_predict_dataset_name, name): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"dataset": name, "error": repr(exc)})
                print(f"ERROR dataset={name} error={exc!r}", flush=True)
            if done % 5 == 0 or done == len(names):
                _status(root, EXP330_ID, "running", "train_normal_multiscale_representation", done, len(names), started)
                print(f"Progress: [{done}/{len(names)}] errors={len(errors)} last={name}", flush=True)
    if errors or len(rows) != len(names):
        _atomic_json(root / "errors.json", {"errors": errors})
        _status(root, EXP330_ID, "failed", "prediction", len(rows), len(names), started)
        raise SystemExit(2)
    rows.sort(key=lambda item: item["dataset"])
    freeze_jsonl(rows, root / "exp330_frozen_predictions.jsonl", root / "exp330_prediction_manifest.json",
                 experiment_id=EXP330_ID, metadata={"datasets": len(rows), "alphas": list(ALPHAS),
                 "pca_dimensions": list(PCA_DIMENSIONS), "random_state": RANDOM_STATE,
                 "rules_frozen_before_label_evaluation": True})
    _status(root, EXP330_ID, "completed", "prediction", len(rows), len(rows), started)


def run_exp330_evaluate(root: Path = EXP330_ROOT) -> None:
    started = time.time()
    rows = verify_frozen_jsonl(root / "exp330_frozen_predictions.jsonl", root / "exp330_prediction_manifest.json")
    baseline = {row["dataset"]: row for row in _exp328_rows()}
    fn_category_rows = {}
    fn_path = EXP328_ROOT / "exp328_fn_detail.csv"
    if fn_path.exists():
        import csv
        with fn_path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                fn_category_rows[(item["dataset"], int(item["fn_index"]))] = item.get("category", "")
    _status(root, EXP330_ID, "running", "retrospective_evaluation", 0, len(rows), started)
    detail = []
    for done, row in enumerate(rows, 1):
        truth = set(np.flatnonzero(_labels(row["dataset"]) == 1).astype(int).tolist())
        hard = set(baseline[row["dataset"]]["automatic_hard_indices"])
        review = set(baseline[row["dataset"]]["default_review_indices"])
        for variant, payload in row["variants"].items():
            for alpha in ALPHAS:
                candidates = set(payload["candidate_indices"][str(alpha)])
                novel = candidates - hard - review
                metrics = _binary_metrics(candidates, truth)
                novel_metrics = _binary_metrics(novel, truth)
                outside_recovered = sum(
                    1 for index in novel & truth
                    if fn_category_rows.get((row["dataset"], index)) == "outside_audited_top64"
                )
                detail.append({
                    "experiment_id": EXP330_ID, "dataset": row["dataset"], "variant": variant, "alpha": alpha,
                    "n_train_normal": row["n_train_normal"], "n_test": row["n_test"],
                    "candidate_count": metrics["count"], "candidate_tp": metrics["tp"], "candidate_fp": metrics["fp"],
                    "candidate_precision": metrics["precision"], "novel_count": novel_metrics["count"],
                    "novel_tp": novel_metrics["tp"], "novel_fp": novel_metrics["fp"],
                    "novel_precision": novel_metrics["precision"], "outside_top64_fn_recovered": outside_recovered,
                    "explained_variance_ratio": payload["explained_variance_ratio"],
                    "metric_scope": "retrospective_shadow_only",
                })
        if done % 10 == 0 or done == len(rows):
            _status(root, EXP330_ID, "running", "retrospective_evaluation", done, len(rows), started)
    write_csv(root / "exp330_results.csv", detail)
    summary = []
    for variant in sorted({item["variant"] for item in detail}):
        for alpha in ALPHAS:
            group = [item for item in detail if item["variant"] == variant and float(item["alpha"]) == alpha]
            tp = sum(int(item["candidate_tp"]) for item in group); fp = sum(int(item["candidate_fp"]) for item in group)
            novel_tp = sum(int(item["novel_tp"]) for item in group); novel_fp = sum(int(item["novel_fp"]) for item in group)
            summary.append({
                "experiment_id": EXP330_ID, "variant": variant, "alpha": alpha, "datasets": len(group),
                "candidate_count": tp + fp, "candidate_tp": tp, "candidate_fp": fp,
                "candidate_precision": tp / (tp + fp) if tp + fp else 0.0,
                "novel_count": novel_tp + novel_fp, "novel_tp": novel_tp, "novel_fp": novel_fp,
                "novel_precision": novel_tp / (novel_tp + novel_fp) if novel_tp + novel_fp else 0.0,
                "outside_top64_fn_recovered": sum(int(item["outside_top64_fn_recovered"]) for item in group),
                "automatic_policy_changed": False,
            })
    write_csv(root / "exp330_summary.csv", summary)
    _status(root, EXP330_ID, "completed", "evaluation", len(rows), len(rows), started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=(329, 330), required=True)
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--dataset-limit", type=int)
    args = parser.parse_args()
    root = args.output_root or (EXP329_ROOT if args.experiment == 329 else EXP330_ROOT)
    if args.experiment == 329:
        if args.phase in {"predict", "all"}: run_exp329_predict(root)
        if args.phase in {"evaluate", "all"}: run_exp329_evaluate(root)
    else:
        if args.phase in {"predict", "all"}: run_exp330_predict(root, workers=args.workers, limit=args.dataset_limit)
        if args.phase in {"evaluate", "all"}: run_exp330_evaluate(root)


if __name__ == "__main__":
    main()

