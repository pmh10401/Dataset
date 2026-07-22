from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.exp309_316_fn_recovery.reporting import write_csv
from experiments.exp317_318_train_reliability_fusion.contracts import freeze_jsonl, sha256_file, verify_frozen_jsonl
from run_experiment_40_original_score_normalization_sweep import CONFIGS as EXP40_CONFIGS, count_cap_threshold, score_pair_for_config as rocket_score_pair
from run_experiment_29_train_normal_threshold_calibration import train_false_positive_stats
from run_experiment_60_62_rocket_imaging_selector_variants import IMAGING_CONFIGS
from run_model_hard_research_experiments import prepare_series_pair_for_scale, score_pair_for_config as imaging_score_pair
from run_original_improvement_experiment import parse_family
from run_rank_ensemble_calibration import align_series_lengths, sanitize_series, z_normalize
from run_experiment_93_nonpos_candidate_reranker import candidate_pool, rank_map, score_candidates, sorted_candidates, weak_base_indices

from .config import (
    DB_PATH, EXPECTED_DATASETS, EXP87_CONFIG, EXP87_RESULTS, EXP90_RESULTS,
    EXP90_SELECTOR, EXP119A_RESULTS, EXP119A_SELECTOR, EXP291_ROOT,
    EXPERIMENT_IDS, FIXED_PRIORS, MAX_TOP, OUTPUT_ROOTS, POOL_TOP,
)
from .scoring import adaptive_weights, choose_replacement, wilson_reliability


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _status(root: Path, number: int, state: str, phase: str, done: int, total: int, started: float) -> None:
    updated = time.time()
    _atomic_json(root / "status.json", {
        "id": EXPERIMENT_IDS[number], "experiment_id": EXPERIMENT_IDS[number],
        "status": state, "phase": phase, "datasets_done": done, "datasets_total": total,
        "progress_percent": 100.0 * done / total if total else 0.0,
        "done": done, "expected": total,
        "elapsed_seconds": updated - started, "updated_at_unix": updated, "updated_unix": updated,
        "contains_test_labels": False if number != 327 else None,
        "retrospective": True, "automatic_policy_changed": False,
    })


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_indices(value) -> list[int]:
    text = str(value or "").strip()
    return [int(float(item)) for item in text.split()] if text else []


def _load_record_without_labels(dataset: str) -> dict:
    with sqlite3.connect(str(DB_PATH)) as conn:
        meta = conn.execute("SELECT series_length FROM datasets WHERE name=?", (dataset,)).fetchone()
        train = conn.execute("""SELECT i.values_blob FROM instances i JOIN datasets d ON i.dataset_id=d.id
            WHERE d.name=? AND i.split='TRAIN' ORDER BY i.instance_index""", (dataset,)).fetchall()
        test = conn.execute("""SELECT i.values_blob FROM instances i JOIN datasets d ON i.dataset_id=d.id
            WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index""", (dataset,)).fetchall()
    train_series = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in train]
    test_series = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in test]
    lengths = [len(x) for x in train_series + test_series]
    test_lengths = [len(x) for x in test_series]
    metadata_len = int(meta[0]) if meta and meta[0] else 0
    return {
        "dataset_name": dataset, "family": parse_family(dataset), "metadata_len": metadata_len,
        "actual_len_median": float(np.median(lengths)) if lengths else float(metadata_len),
        "actual_len_max": float(np.max(lengths)) if lengths else float(metadata_len),
        "test_actual_len_median": float(np.median(test_lengths)) if test_lengths else float(metadata_len),
        "train_series": train_series, "test_series": test_series,
    }


def _score_bundle(record: dict) -> dict[str, dict]:
    target = int(record["metadata_len"] or round(record["actual_len_median"]))
    target = max(8, target)
    x_train = align_series_lengths(record["train_series"], target)
    x_test = align_series_lengths(record["test_series"], target)
    x_train_z = z_normalize(x_train).astype(np.float32)
    x_test_z = z_normalize(x_test).astype(np.float32)
    rocket_config = next(c for c in EXP40_CONFIGS if c["name"] == "rocket_256_knn3_local_gap")
    rocket_train, rocket_test = rocket_score_pair(x_train_z, x_test_z, target, rocket_config, {})
    bundles = {"rocket": (rocket_train, rocket_test)}

    image_target = min(max(8, int(round(record["actual_len_median"]))), 2048)
    raw_train = align_series_lengths(record["train_series"], image_target)
    raw_test = align_series_lengths(record["test_series"], image_target)
    z_train = z_normalize(raw_train).astype(np.float32)
    z_test = z_normalize(raw_test).astype(np.float32)
    for source, config_key in (("frequency_image", "exp55_best"), ("texture_shape", "exp56_best")):
        config = IMAGING_CONFIGS[config_key]
        pair = prepare_series_pair_for_scale(config.get("series_scale", "per_series_z"), raw_train, raw_test, z_train, z_test)
        bundles[source] = imaging_score_pair(pair[0], pair[1], image_target, config, record)

    output = {}
    for name, (train_scores, test_scores) in bundles.items():
        threshold, _, _ = count_cap_threshold(train_scores, 0.015)
        exceed_count, _ = train_false_positive_stats(train_scores, threshold)
        order = np.argsort(-np.asarray(test_scores), kind="stable")[:MAX_TOP].astype(int).tolist()
        output[name] = {
            "order": order, "threshold": float(threshold),
            **wilson_reliability(int(exceed_count), len(train_scores)),
        }
    return output


def _exp87_maps() -> dict[tuple[str, str], dict[str, str]]:
    return {(row["dataset_name"], row["threshold_method"]): row for row in _read_csv(EXP87_RESULTS)
            if row.get("config_name") == EXP87_CONFIG}


def _dataset_names() -> list[str]:
    names = sorted({
        row["dataset_name"] for row in _read_csv(EXP90_RESULTS)
        if row.get("selector_name") == EXP90_SELECTOR
    })
    if len(names) != EXPECTED_DATASETS:
        raise ValueError(f"Exp90 lineage coverage mismatch: {len(names)}")
    return names


def _build_cache_row(args) -> dict:
    dataset, aux = args
    record = _load_record_without_labels(dataset)
    sources = _score_bundle(record)
    for threshold_method, source in (("family_guard_v1", "aux_family_guard"), ("count_cap_3pct", "aux_count_cap")):
        row = aux.get((dataset, threshold_method))
        if row is None:
            sources[source] = {
                "order": [], "threshold": None, "available": False,
                **wilson_reliability(0, 0),
            }
            continue
        safe_limit = 0.015 if threshold_method == "family_guard_v1" else 0.02
        train_rate = float(row.get("train_exceed_rate") or 1.0)
        order = _parse_indices(row.get("top_score_indices"))[:MAX_TOP] if train_rate <= safe_limit else []
        sources[source] = {
            "order": order, "threshold": float(row.get("threshold") or 0.0), "available": True,
            **wilson_reliability(int(float(row.get("train_exceed_count") or 0)), int(float(row.get("train_score_count") or 0))),
        }
    return {
        "experiment_id": EXPERIMENT_IDS[325], "dataset": dataset,
        "train_count": len(record["train_series"]), "test_count": len(record["test_series"]),
        "sources": sources, "contains_test_labels": False, "contains_evaluation_metrics": False,
        "uses_test_metrics_for_weights": False, "five_source_architecture": True,
        "all_five_sources_available": all(data.get("available", True) for data in sources.values()),
        "retrospective": True,
    }


def run_exp325(root: Path, *, limit: int | None = None, workers: int = 7) -> None:
    root.mkdir(parents=True, exist_ok=True); started = time.time(); names = _dataset_names()
    if limit: names = names[:limit]
    aux = _exp87_maps(); _status(root, 325, "running", "five_source_cache", 0, len(names), started)
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_build_cache_row, (name, aux)): name for name in names}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try: rows.append(future.result())
            except Exception as exc: errors.append((name, repr(exc))); print(f"ERROR dataset={name} error={exc!r}", flush=True)
            if done % 10 == 0 or done == len(names):
                _status(root, 325, "running", "five_source_cache", done, len(names), started)
                print(f"Progress: [{done}/{len(names)}] errors={len(errors)} last={name}", flush=True)
    if errors or len(rows) != len(names):
        _atomic_json(root / "errors.json", {"errors": errors}); raise SystemExit(2)
    rows.sort(key=lambda row: row["dataset"])
    freeze_jsonl(rows, root / "exp325_frozen_source_cache.jsonl", root / "exp325_cache_manifest.json",
                 experiment_id=EXPERIMENT_IDS[325], metadata={
                     "datasets": len(rows), "five_source_architecture": True,
                     "all_five_sources_available_datasets": sum(bool(row["all_five_sources_available"]) for row in rows),
                     "auxiliary_source_coverage_is_partial": True,
                 })
    flat = []
    for row in rows:
        flat.append({"experiment_id": EXPERIMENT_IDS[325], "dataset": row["dataset"], "train_count": row["train_count"],
                     **{f"{name}_reliability": data["reliability"] for name, data in row["sources"].items()}})
    write_csv(root / "exp325_results.csv", flat)
    write_csv(root / "exp325_summary.csv", [{"experiment_id": EXPERIMENT_IDS[325], "datasets": len(rows), "errors": 0,
                                               "five_source_architecture": True,
                                               "all_five_sources_available_datasets": sum(bool(row["all_five_sources_available"]) for row in rows),
                                               "auxiliary_source_coverage_is_partial": True,
                                               "contains_test_labels": False}])
    _status(root, 325, "completed", "five_source_cache", len(names), len(names), started)


def _load_selector_maps() -> tuple[dict, dict]:
    exp90 = {row["dataset_name"]: row for row in _read_csv(EXP90_RESULTS) if row.get("selector_name") == EXP90_SELECTOR}
    validated = {row["dataset_name"]: row for row in _read_csv(EXP119A_RESULTS) if row.get("selector_name") == EXP119A_SELECTOR}
    if len(exp90) != EXPECTED_DATASETS or set(exp90) != set(validated):
        raise ValueError("Exp90/Exp119a coverage mismatch")
    return exp90, validated


def _select(cache: dict, base_row: dict, mode: str) -> dict:
    sources = cache["sources"]
    reliabilities = {name: float(data["reliability"]) for name, data in sources.items()}
    weights = adaptive_weights(reliabilities, mode)
    orders = {name: list(data["order"]) for name, data in sources.items()}
    ranks = {name: rank_map(order) for name, order in orders.items()}
    base = set(_parse_indices(base_row.get("selected_indices")))
    pool = candidate_pool(*(order[:POOL_TOP] for order in orders.values()))
    scored = score_candidates(pool | base, ranks, weights, max_rank=MAX_TOP)
    ranked = sorted_candidates(scored)
    top = next((idx for idx in ranked if idx not in base), None)
    weak = weak_base_indices(base, scored)
    info = scored.get(top, {}) if top is not None else {}
    base_best = max([float(scored.get(idx, {}).get("score", 0.0)) for idx in base] or [0.0])
    gain = float(info.get("score", 0.0)) - base_best
    selected = choose_replacement(base_indices=base, weak_indices=weak, top_candidate=top,
                                  top_support=int(info.get("support", 0)), top_best_rank=int(info.get("best_rank", 99)),
                                  score_gain=gain, train_count=int(cache["train_count"]))
    return {"selected_indices": sorted(selected), "top_candidate": top, "score_gain": gain,
            "top_support": int(info.get("support", 0)), "top_best_rank": int(info.get("best_rank", 99)),
            "weights": weights, "reliabilities": reliabilities}


def run_exp326(root: Path, cache_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True); started = time.time()
    cache_rows = verify_frozen_jsonl(cache_root / "exp325_frozen_source_cache.jsonl", cache_root / "exp325_cache_manifest.json")
    exp90, validated = _load_selector_maps(); _status(root, 326, "running", "five_source_replay", 0, len(cache_rows), started)
    outputs = []
    for done, cache in enumerate(cache_rows, 1):
        dataset = cache["dataset"]
        variants = {mode: _select(cache, exp90[dataset], mode) for mode in ("fixed_priors", "prior_reliability", "pure_reliability")}
        expected = sorted(_parse_indices(validated[dataset].get("selected_indices")))
        outputs.append({"experiment_id": EXPERIMENT_IDS[326], "dataset": dataset, "expected_validated_indices": expected,
                        "fixed_control_match": variants["fixed_priors"]["selected_indices"] == expected,
                        "variants": variants, "contains_test_labels": False, "contains_evaluation_metrics": False,
                        "uses_test_metrics_for_weights": False, "all_five_sources_weighted": True, "retrospective": True})
        if done % 25 == 0 or done == len(cache_rows): _status(root, 326, "running", "five_source_replay", done, len(cache_rows), started)
    mismatch = sum(not row["fixed_control_match"] for row in outputs)
    freeze_jsonl(outputs, root / "exp326_frozen_predictions.jsonl", root / "exp326_prediction_manifest.json",
                 experiment_id=EXPERIMENT_IDS[326], metadata={"datasets": len(outputs), "fixed_control_mismatch": mismatch})
    detail = []
    for row in outputs:
        fixed = row["variants"]["fixed_priors"]["selected_indices"]
        for mode, variant in row["variants"].items():
            detail.append({"experiment_id": EXPERIMENT_IDS[326], "dataset": row["dataset"], "variant": mode,
                           "selected_indices": " ".join(map(str, variant["selected_indices"])),
                           "changed_from_fixed": variant["selected_indices"] != fixed,
                           "fixed_control_match": row["fixed_control_match"],
                           **{f"weight_{name}": weight for name, weight in variant["weights"].items()}})
    write_csv(root / "exp326_results.csv", detail)
    write_csv(root / "exp326_summary.csv", [{"experiment_id": EXPERIMENT_IDS[326], "datasets": len(outputs),
                                               "variant": mode, "fixed_control_mismatch": mismatch,
                                               "changed_from_fixed_datasets": sum(row["variants"][mode]["selected_indices"] != row["variants"]["fixed_priors"]["selected_indices"] for row in outputs)}
                                              for mode in ("fixed_priors", "prior_reliability", "pure_reliability")])
    _status(root, 326, "completed", "five_source_replay", len(outputs), len(outputs), started)


def _test_labels(dataset: str) -> np.ndarray:
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute("""SELECT i.label FROM instances i JOIN datasets d ON i.dataset_id=d.id
            WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index""", (dataset,)).fetchall()
    return np.asarray([int(row[0]) for row in rows], dtype=np.int64)


def run_exp327(root: Path, replay_root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True); started = time.time()
    rows = verify_frozen_jsonl(replay_root / "exp326_frozen_predictions.jsonl", replay_root / "exp326_prediction_manifest.json")
    exp291 = {row["dataset"]: row for row in verify_frozen_jsonl(EXP291_ROOT / "exp291_frozen_operational_routing.jsonl", EXP291_ROOT / "exp291_prediction_manifest.json")}
    _status(root, 327, "running", "retrospective_evaluation", 0, len(rows), started)
    detail = []
    for done, row in enumerate(rows, 1):
        y = _test_labels(row["dataset"]); current = exp291[row["dataset"]]
        hard = set(current["automatic_hard_indices"]); review = set(current["default_review_indices"])
        for mode, variant in row["variants"].items():
            selected = set(variant["selected_indices"]); tp = sum(int(y[i] == 1) for i in selected); fp = len(selected) - tp
            detail.append({"experiment_id": EXPERIMENT_IDS[327], "dataset": row["dataset"], "variant": mode,
                           "selected_count": len(selected), "tp": tp, "fp": fp,
                           "selected_indices": " ".join(map(str, sorted(selected))),
                           "overlap_current_exp291_hard": len(selected & hard), "overlap_current_exp291_review": len(selected & review),
                           "exact_downstream_route_replayed": False, "metric_scope": "retrospective_upstream_selector_only"})
        if done % 25 == 0 or done == len(rows): _status(root, 327, "running", "retrospective_evaluation", done, len(rows), started)
    write_csv(root / "exp327_results.csv", detail)
    summary = []
    for mode in ("fixed_priors", "prior_reliability", "pure_reliability"):
        subset = [row for row in detail if row["variant"] == mode]; tp = sum(row["tp"] for row in subset); fp = sum(row["fp"] for row in subset)
        f1s = []
        for row in subset:
            y = _test_labels(row["dataset"]); fn = int(y.sum()) - row["tp"]; denominator = 2 * row["tp"] + row["fp"] + fn
            f1s.append(2 * row["tp"] / denominator if denominator else 0.0)
        summary.append({"experiment_id": EXPERIMENT_IDS[327], "variant": mode, "datasets": len(subset), "tp": tp, "fp": fp,
                        "precision": tp / (tp + fp) if tp + fp else 0.0, "mean_f1": float(np.mean(f1s)),
                        "exact_downstream_route_replayed": False, "automatic_exp291_change_claimed": False,
                        "interpretation": "upstream Exp93 selector counterfactual only; full downstream replay required for Exp291 policy change"})
    write_csv(root / "exp327_summary.csv", summary)
    _status(root, 327, "completed", "retrospective_evaluation", len(rows), len(rows), started)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", type=int, choices=(325, 326, 327), required=True)
    parser.add_argument("--output-root", type=Path); parser.add_argument("--dataset-limit", type=int); parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--cache-root", type=Path, default=OUTPUT_ROOTS[325]); parser.add_argument("--replay-root", type=Path, default=OUTPUT_ROOTS[326])
    args = parser.parse_args(); root = args.output_root or OUTPUT_ROOTS[args.experiment]
    if args.experiment == 325: run_exp325(root, limit=args.dataset_limit, workers=args.workers)
    elif args.experiment == 326: run_exp326(root, args.cache_root)
    else: run_exp327(root, args.replay_root)


if __name__ == "__main__": main()
