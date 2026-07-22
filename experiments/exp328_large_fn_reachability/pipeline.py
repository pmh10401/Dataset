from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.exp292_standard_review_evidence_tiering.sources import load_all_source_support
from experiments.exp309_316_fn_recovery.reporting import write_csv
from experiments.exp317_318_train_reliability_fusion.contracts import freeze_jsonl, verify_frozen_jsonl
from experiments.exp325_327_full_source_reliability.pipeline import _load_record_without_labels, _score_bundle

from .config import DB_PATH, EXPECTED_DATASETS, EXP291_ROOT, EXPERIMENT_ID, MAX_TOP, MIN_TRAIN_NORMAL, OUTPUT_ROOT


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _status(root: Path, status: str, phase: str, done: int, total: int, started: float) -> None:
    updated = time.time()
    _atomic_json(root / "status.json", {
        "id": EXPERIMENT_ID, "experiment_id": EXPERIMENT_ID, "status": status, "phase": phase,
        "done": done, "expected": total, "datasets_done": done, "datasets_total": total,
        "progress_percent": 100.0 * done / total if total else 0.0,
        "elapsed_seconds": updated - started, "updated_unix": updated, "updated_at_unix": updated,
        "retrospective": True, "automatic_policy_changed": False,
    })


def classify_false_negative(index: int, *, review: set[int],
                            primary_candidates: dict[str, set[int]],
                            primary_orders: dict[str, list[int]],
                            independent: dict[str, set[int]]) -> dict[str, object]:
    if index in review:
        return {"category": "already_in_review_lane", "candidate_source_count": 0, "rank_bucket": "review"}
    candidate_sources = [name for name, indices in {**primary_candidates, **independent}.items() if index in indices]
    if len(candidate_sources) >= 2:
        return {"category": "multi_source_candidate_not_routed", "candidate_source_count": len(candidate_sources), "rank_bucket": "candidate"}
    if len(candidate_sources) == 1:
        return {"category": "single_source_candidate", "candidate_source_count": 1, "rank_bucket": "candidate"}
    ranks = [order.index(index) + 1 for order in primary_orders.values() if index in order]
    if ranks:
        best = min(ranks)
        bucket = "top12" if best <= 12 else "top32" if best <= 32 else "top64"
        return {"category": f"{bucket}_below_threshold", "candidate_source_count": 0, "rank_bucket": bucket, "best_rank": best}
    return {"category": "outside_audited_top64", "candidate_source_count": 0, "rank_bucket": "outside_top64"}


def _load_large_names() -> list[str]:
    baseline = {row["dataset"] for row in verify_frozen_jsonl(
        EXP291_ROOT / "exp291_frozen_operational_routing.jsonl",
        EXP291_ROOT / "exp291_prediction_manifest.json",
    )}
    with sqlite3.connect(str(DB_PATH)) as conn:
        names = [row[0] for row in conn.execute(
            "SELECT name FROM datasets WHERE train_normal_count >= ? ORDER BY name", (MIN_TRAIN_NORMAL,)
        ) if row[0] in baseline]
    if len(names) != EXPECTED_DATASETS:
        raise ValueError(f"large-dataset coverage mismatch: {len(names)}/{EXPECTED_DATASETS}")
    return names


def _build_prediction(args) -> dict:
    dataset, exp291_row, independent = args
    record = _load_record_without_labels(dataset)
    primary = _score_bundle(record, max_top=MAX_TOP)
    return {
        "experiment_id": EXPERIMENT_ID, "dataset": dataset,
        "n_train_normal": len(record["train_series"]), "n_test": len(record["test_series"]),
        "automatic_hard_indices": sorted(map(int, exp291_row["automatic_hard_indices"])),
        "default_review_indices": sorted(map(int, exp291_row["default_review_indices"])),
        "primary_sources": primary,
        "independent_sources": {name: sorted(map(int, support.get(dataset, set()))) for name, support in independent.items()},
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "uses_test_metrics_for_routing": False, "automatic_policy_changed": False,
        "retrospective": True,
    }


def run_predict(root: Path, *, workers: int = 7, limit: int | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True); started = time.time()
    names = _load_large_names();
    if limit: names = names[:limit]
    baseline = {row["dataset"]: row for row in verify_frozen_jsonl(
        EXP291_ROOT / "exp291_frozen_operational_routing.jsonl", EXP291_ROOT / "exp291_prediction_manifest.json")}
    independent, manifests = load_all_source_support()
    _status(root, "running", "label_free_reachability_cache", 0, len(names), started)
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_build_prediction, (name, baseline[name], independent)): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try: rows.append(future.result())
            except Exception as exc: errors.append((name, repr(exc))); print(f"ERROR dataset={name} error={exc!r}", flush=True)
            if done % 5 == 0 or done == len(names):
                _status(root, "running", "label_free_reachability_cache", done, len(names), started)
                print(f"Progress: [{done}/{len(names)}] errors={len(errors)} last={name}", flush=True)
    if errors or len(rows) != len(names):
        _atomic_json(root / "errors.json", {"errors": errors}); _status(root, "failed", "prediction", len(rows), len(names), started); raise SystemExit(2)
    rows.sort(key=lambda row: row["dataset"])
    freeze_jsonl(rows, root / "exp328_frozen_reachability.jsonl", root / "exp328_prediction_manifest.json",
                 experiment_id=EXPERIMENT_ID, metadata={"datasets": len(rows), "source_manifests": manifests,
                 "population": "train_normal_count>=100", "automatic_policy_changed": False})
    _status(root, "completed", "prediction", len(rows), len(rows), started)


def _labels(dataset: str) -> np.ndarray:
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute("""SELECT i.label FROM instances i JOIN datasets d ON i.dataset_id=d.id
            WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index""", (dataset,)).fetchall()
    return np.asarray([int(row[0]) for row in rows], dtype=np.int64)


def run_evaluate(root: Path) -> None:
    started = time.time()
    predictions = verify_frozen_jsonl(root / "exp328_frozen_reachability.jsonl", root / "exp328_prediction_manifest.json")
    _status(root, "running", "retrospective_fn_classification", 0, len(predictions), started)
    fn_rows, dataset_rows = [], []
    for done, row in enumerate(predictions, 1):
        y = _labels(row["dataset"]); hard = set(row["automatic_hard_indices"]); review = set(row["default_review_indices"])
        true_indices = set(np.flatnonzero(y == 1).astype(int).tolist()); false_negatives = sorted(true_indices - hard)
        primary_candidates = {name: set(data["candidate_indices"]) for name, data in row["primary_sources"].items()}
        primary_orders = {name: list(data["order"]) for name, data in row["primary_sources"].items()}
        independent = {name: set(indices) for name, indices in row["independent_sources"].items()}
        categories = Counter()
        for index in false_negatives:
            result = classify_false_negative(index, review=review, primary_candidates=primary_candidates,
                                             primary_orders=primary_orders, independent=independent)
            categories[result["category"]] += 1
            fn_rows.append({"experiment_id": EXPERIMENT_ID, "dataset": row["dataset"], "n_train_normal": row["n_train_normal"],
                            "n_test": row["n_test"], "fn_index": index, **result,
                            "primary_candidate_sources": " ".join(name for name, values in primary_candidates.items() if index in values),
                            "independent_candidate_sources": " ".join(name for name, values in independent.items() if index in values),
                            "metric_scope": "retrospective_diagnostic_only"})
        tp = len(true_indices & hard); fp = len(hard - true_indices)
        dataset_rows.append({"experiment_id": EXPERIMENT_ID, "dataset": row["dataset"], "n_train_normal": row["n_train_normal"],
                             "n_test": row["n_test"], "true_anomalies": len(true_indices), "hard_count": len(hard),
                             "hard_tp": tp, "hard_fp": fp, "hard_fn": len(false_negatives),
                             **{f"fn_{name}": count for name, count in sorted(categories.items())}})
        if done % 10 == 0 or done == len(predictions): _status(root, "running", "retrospective_fn_classification", done, len(predictions), started)
    write_csv(root / "exp328_fn_detail.csv", fn_rows); write_csv(root / "exp328_results.csv", dataset_rows)
    totals = Counter(row["category"] for row in fn_rows)
    summary_rows = [{"experiment_id": EXPERIMENT_ID, "category": category, "fn_count": count,
                     "fn_share": count / len(fn_rows) if fn_rows else 0.0} for category, count in totals.most_common()]
    write_csv(root / "exp328_summary.csv", summary_rows)
    _atomic_json(root / "exp328_completion.json", {"experiment_id": EXPERIMENT_ID, "datasets": len(dataset_rows),
                 "false_negatives": len(fn_rows), "categories": dict(totals), "automatic_policy_changed": False,
                 "retrospective": True, "prospective_validated": False})
    _status(root, "completed", "evaluation", len(predictions), len(predictions), started)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--dataset-limit", type=int); args = parser.parse_args()
    if args.phase in {"predict", "all"}: run_predict(args.output_root, workers=args.workers, limit=args.dataset_limit)
    if args.phase in {"evaluate", "all"}: run_evaluate(args.output_root)


if __name__ == "__main__": main()
